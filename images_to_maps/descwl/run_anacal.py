#!/usr/bin/env python
"""AnaCal processing script for weak lensing simulations.

Runs AnaCal on the same test set used by NPE for fair comparison.

Usage:
    python run_anacal.py

Configure settings in config_run_anacal.yaml
"""

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import anacal
import galsim
import hydra
import numpy as np
import lightning as L
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from numpy.random import SeedSequence
from omegaconf import DictConfig
from tqdm import tqdm

from descwl_shear_sims.psfs import make_ps_psf
from descwl_shear_sims.sim import get_se_dim


# =============================================================================
# AnaCal Utility Classes
# =============================================================================


class GalSimPsfWrapper(anacal.psf.BasePsf):
    """Wrapper to make GalSim PSF objects compatible with anacal."""

    def __init__(self, galsim_psf, pixel_scale, npix=64, is_variable=False):
        self.galsim_psf = galsim_psf
        self.pixel_scale = pixel_scale
        self.npix = npix
        self.is_variable = is_variable
        self.shape = None

    def draw(self, x, y):
        """Draw PSF at position (x, y)."""
        if self.is_variable:
            pos = galsim.PositionD(x, y)
            psf_at_pos = self.galsim_psf.getPSF(pos)
        else:
            psf_at_pos = self.galsim_psf

        return psf_at_pos.drawImage(
            nx=self.npix, ny=self.npix, scale=self.pixel_scale, method="auto"
        ).array.astype(np.float64)


# =============================================================================
# PSF Reconstruction Functions
# =============================================================================


def reconstruct_variable_psf(file_path, setting_config):
    """
    Reconstruct variable PSF from filename for backward compatibility.

    Parameters
    ----------
    file_path : str
        Path to data file (e.g., "dataset_123_size_1.pt")
    setting_config : dict
        Config with keys: seed, num_images, variation_factor, coadd_dim, rotate, pixel_scale, npix

    Returns
    -------
    GalSimPsfWrapper with is_variable=True
    """
    # Parse image index from filename
    match = re.search(r"dataset_(\d+)_size", os.path.basename(file_path))
    if not match:
        raise ValueError(f"Cannot parse index from filename: {file_path}")
    global_idx = int(match.group(1))

    # Reconstruct the same seed sequence used during generation
    ss = SeedSequence(setting_config["seed"])
    child_seeds = ss.spawn(setting_config["num_images"])
    child_seed = int(child_seeds[global_idx].generate_state(1)[0])
    psf_seed = (child_seed + 1000000) % (2**32)

    # Compute se_dim
    se_dim = get_se_dim(
        coadd_dim=setting_config["coadd_dim"],
        rotate=setting_config.get("rotate", False),
    )

    # Reconstruct PSF
    rng = np.random.RandomState(psf_seed)
    psf_obj = make_ps_psf(
        rng=rng,
        dim=se_dim,
        variation_factor=setting_config.get("variation_factor", 1.0),
    )

    return GalSimPsfWrapper(
        psf_obj,
        pixel_scale=setting_config.get("pixel_scale", 0.2),
        npix=setting_config.get("npix", 64),
        is_variable=True,
    )


# =============================================================================
# Band Combination Functions
# =============================================================================


def combine_multiband_images(
    images_tensor, band_variances=None, method="inverse_variance"
):
    """Combine multi-band images into single image for anacal processing."""
    if method == "inverse_variance":
        if band_variances is not None:
            weights = []
            band_names = list(band_variances.keys())  # Use actual band names from data
            for band in band_names:
                variance_value = band_variances.get(band, 0.354)
                weights.append(1.0 / variance_value)
            weights = torch.tensor(weights, dtype=torch.float32)
        else:
            typical_variances = torch.tensor([0.099, 0.138, 0.354, 1.344])
            typical_variances = typical_variances[: images_tensor.shape[0]]
            weights = 1.0 / typical_variances

        total_weight = weights.sum()
        normalized_weights = weights / total_weight
        combined = torch.sum(images_tensor * normalized_weights.view(-1, 1, 1), dim=0)
        combined_variance = 1.0 / total_weight.item()

    elif method == "mean":
        combined = torch.mean(images_tensor, dim=0)
        combined_variance = 0.354

    else:
        raise ValueError(f"Unknown combination method: {method}")

    return combined, combined_variance


def combine_multiband_masks(masks_dict, method="union"):
    """Combine masks from multiple bands into a single mask."""
    if not masks_dict:
        return None

    band_names = list(masks_dict.keys())
    first_mask = masks_dict[band_names[0]]

    if method == "union":
        combined_mask = np.zeros_like(first_mask, dtype=np.int16)
        for mask in masks_dict.values():
            combined_mask = np.logical_or(combined_mask, mask).astype(np.int16)
    elif method == "first_only":
        combined_mask = first_mask.copy()
    elif method == "intersection":
        combined_mask = np.ones_like(first_mask, dtype=np.int16)
        for mask in masks_dict.values():
            combined_mask = np.logical_and(combined_mask, mask).astype(np.int16)
    else:
        raise ValueError(f"Unknown combination method: {method}")

    return combined_mask


# =============================================================================
# AnaCal Processing Functions
# =============================================================================


def anacal_multiband_combined(
    images_tensor,
    psf_input,
    masks_dict=None,
    combine_method="inverse_variance",
    mask_combine_method="union",
    star_catalog=None,
    npix=64,
    sigma_arcsec=0.52,
    mag_zero=30.0,
    pixel_scale=0.2,
    band_variances=None,
):
    """Process multi-band images by combining them first, then running anacal."""
    combined_image, combined_variance = combine_multiband_images(
        images_tensor, band_variances=band_variances, method=combine_method
    )
    gal_array = combined_image.numpy()

    combined_mask = None
    if masks_dict is not None:
        combined_mask = combine_multiband_masks(masks_dict, method=mask_combine_method)

    noise_array = np.random.normal(
        0, np.sqrt(combined_variance), gal_array.shape
    ).astype(np.float64)

    fpfs_config = anacal.fpfs.FpfsConfig(npix=npix, sigma_arcsec=sigma_arcsec)

    if isinstance(psf_input, np.ndarray):
        out = anacal.fpfs.process_image(
            fpfs_config=fpfs_config,
            mag_zero=mag_zero,
            gal_array=gal_array,
            psf_array=psf_input,
            pixel_scale=pixel_scale,
            noise_variance=combined_variance,
            noise_array=noise_array,
            mask_array=combined_mask,
            star_catalog=star_catalog,
            detection=None,
        )
    else:
        center_psf = psf_input.draw(gal_array.shape[1] // 2, gal_array.shape[0] // 2)
        out = anacal.fpfs.process_image(
            fpfs_config=fpfs_config,
            mag_zero=mag_zero,
            gal_array=gal_array,
            psf_array=center_psf,
            psf_object=psf_input,
            pixel_scale=pixel_scale,
            noise_variance=combined_variance,
            noise_array=noise_array,
            mask_array=combined_mask,
            star_catalog=star_catalog,
            detection=None,
        )

    e1 = out["fpfs_w"] * out["fpfs_e1"]
    e1g1 = out["fpfs_dw_dg1"] * out["fpfs_e1"] + out["fpfs_w"] * out["fpfs_de1_dg1"]
    e2 = out["fpfs_w"] * out["fpfs_e2"]
    e2g2 = out["fpfs_dw_dg2"] * out["fpfs_e2"] + out["fpfs_w"] * out["fpfs_de2_dg2"]

    return np.sum(e1), np.sum(e1g1), np.sum(e2), np.sum(e2g2), len(e1)


# =============================================================================
# Processing Functions
# =============================================================================


def process_sample(sample, config, file_path=None):
    """Process a single sample with AnaCal."""
    images = sample["images"]
    catalog = sample["tile_catalog"]
    anacal_data = sample["anacal_data"]

    masks_dict = anacal_data["masks"]
    band_variances = anacal_data["variances"]
    bright_star_catalog = anacal_data.get("bright_star_catalog")
    pixel_scale = anacal_data.get("pixel_scale", config["pixel_scale"])

    # Determine PSF: reconstruct variable PSF if setting_config provided
    setting_config = config.get("setting_config")
    if setting_config and setting_config.get("variable_psf", False) and file_path:
        psf_input = reconstruct_variable_psf(file_path, setting_config)
    else:
        psf_input = anacal_data["psf_image"]

    e1_sum, e1g1_sum, e2_sum, e2g2_sum, num_detections = anacal_multiband_combined(
        images,
        psf_input,
        masks_dict=masks_dict,
        combine_method=config["combine_method"],
        mask_combine_method=config["mask_combine_method"],
        star_catalog=bright_star_catalog,
        npix=config["npix"],
        sigma_arcsec=config["sigma_arcsec"],
        mag_zero=config["mag_zero"],
        pixel_scale=pixel_scale,
        band_variances=band_variances,
    )

    return {
        "e1_sum": float(e1_sum),
        "e1g1_sum": float(e1g1_sum),
        "e2_sum": float(e2_sum),
        "e2g2_sum": float(e2g2_sum),
        "num_detections": int(num_detections),
        "shear_1": float(catalog["shear_1"]),
        "shear_2": float(catalog["shear_2"]),
    }


def process_file(args):
    """Worker function for multiprocessing."""
    file_path, config = args
    try:
        data_list = torch.load(file_path, weights_only=False)
        results = []
        for sample in data_list:
            results.append(process_sample(sample, config, file_path))
        return results, None
    except Exception as e:
        return None, str(e)


@hydra.main(version_base=None, config_path=".", config_name="config_run_anacal")
def main(cfg: DictConfig) -> None:
    print("=== ANACAL PROCESSING ===")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"Output: {cfg.output_file}")

    # Clear global hydra to allow re-initialization for loading training config
    GlobalHydra.instance().clear()

    # Use NPE's hydra config for data source with cached_data_path override
    with initialize(config_path=".", version_base=None):
        hydra_cfg = compose(
            "config_train_npe",
            overrides=[f"paths.cached_data={cfg.cached_data_path}"],
        )

    # Same seed as NPE for identical test set
    L.seed_everything(hydra_cfg.train.seed)
    print(f"Seed: {hydra_cfg.train.seed}")

    # Get test files from NPE's data source
    print("Setting up data source...")
    data_source = instantiate(hydra_cfg.train.data_source)
    data_source.setup("test")
    test_files = data_source.test_dataset.file_paths
    print(f"Processing {len(test_files)} test files...")

    # Initialize results
    results = {
        "e1_sum": [],
        "e1g1_sum": [],
        "e2_sum": [],
        "e2g2_sum": [],
        "num_detections": [],
        "shear_1": [],
        "shear_2": [],
    }

    start_time = time.time()
    failed_count = 0
    n_workers = cfg.get("n_workers", 1)
    cfg_dict = dict(cfg)  # Convert to dict for passing to workers

    def append_results(file_results):
        for r in file_results:
            results["e1_sum"].append(r["e1_sum"])
            results["e1g1_sum"].append(r["e1g1_sum"])
            results["e2_sum"].append(r["e2_sum"])
            results["e2g2_sum"].append(r["e2g2_sum"])
            results["num_detections"].append(r["num_detections"])
            results["shear_1"].append(r["shear_1"])
            results["shear_2"].append(r["shear_2"])

    if n_workers > 1:
        print(f"Using {n_workers} workers...")
        args_list = [(f, cfg_dict) for f in test_files]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Use map() to preserve order (as_completed returns in completion order)
            for file_results, error in tqdm(
                executor.map(process_file, args_list),
                total=len(args_list),
                desc="Processing",
            ):
                if error:
                    print(f"Error: {error}")
                    failed_count += 1
                else:
                    append_results(file_results)
    else:
        for file_path in tqdm(test_files, desc="Processing"):
            file_results, error = process_file((file_path, cfg_dict))
            if error:
                print(f"Error processing {file_path}: {error}")
                failed_count += 1
            else:
                append_results(file_results)

    # Save results
    elapsed_time = time.time() - start_time
    output = {
        "e1_sum": np.array(results["e1_sum"]),
        "e1g1_sum": np.array(results["e1g1_sum"]),
        "e2_sum": np.array(results["e2_sum"]),
        "e2g2_sum": np.array(results["e2g2_sum"]),
        "num_detections": np.array(results["num_detections"]),
        "shear_1": np.array(results["shear_1"]),
        "shear_2": np.array(results["shear_2"]),
        "total_e1_sum": float(np.sum(results["e1_sum"])),
        "total_e1g1_sum": float(np.sum(results["e1g1_sum"])),
        "total_e2_sum": float(np.sum(results["e2_sum"])),
        "total_e2g2_sum": float(np.sum(results["e2g2_sum"])),
        "total_detections": int(np.sum(results["num_detections"])),
        "n_samples": len(results["e1_sum"]),
        "processing_time": elapsed_time,
        "seed": hydra_cfg.train.seed,
    }

    torch.save(output, cfg.output_file)
    print(f"\nSaved to: {cfg.output_file}")

    # Summary
    print("\n=== COMPLETE ===")
    print(f"Samples processed: {output['n_samples']}")
    if failed_count > 0:
        print(f"Failed files: {failed_count}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Total detections: {output['total_detections']}")


if __name__ == "__main__":
    main()
