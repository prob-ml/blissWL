"""Lensing DC2 data module - fully standalone implementation.

This module is self-contained and has no dependencies on bliss core modules.
"""

import copy
import logging
import math
import os
import re
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

from images_to_maps.cached_dataset import CachedSimulatedDataModule
from images_to_maps.dc2.utils import get_bands_flux_and_psf, plocs_from_ra_dec

PADDED_CROP_RE = re.compile(
    r"^tract_(?P<tract>\d+)_patch_(?P<patch_x>\d+)_(?P<patch_y>\d+)"
    r"_padded_crop_(?P<crop_row>\d+)_(?P<crop_col>\d+)_"
)
# Padded crops step by their 4x4-tile interior over a 16x16-tile patch.
CROPS_PER_PATCH_SIDE = 4
HELD_OUT_SPLITS = ("test", "val")  # earlier entries take precedence


def padded_crop_position(file_name: str) -> tuple[int, int, int]:
    """Tract and crop row/column in the tract's crop grid (row 0 is the south edge)."""
    match = PADDED_CROP_RE.match(file_name)
    if match is None:
        raise ValueError(f"Not a padded crop file name: {file_name}")
    crop_row = int(match.group("crop_row"))
    crop_col = int(match.group("crop_col"))
    if crop_row >= CROPS_PER_PATCH_SIDE or crop_col >= CROPS_PER_PATCH_SIDE:
        raise ValueError(f"Unexpected crop index in {file_name}")
    row = CROPS_PER_PATCH_SIDE * int(match.group("patch_y")) + crop_row
    col = CROPS_PER_PATCH_SIDE * int(match.group("patch_x")) + crop_col
    return int(match.group("tract")), row, col


def assign_crop_splits(
    file_names: List[str], split_regions: dict, train_buffer_crops: int = 0
) -> dict[str, str]:
    """Assign padded crops to "train", "val", "test", or "buffer" (dropped).

    ``split_regions`` maps "test" and "val" to lists of rectangles
    ``{tract, rows: [first, last], cols: [first, last]}`` (inclusive) in each
    tract's crop grid.  Test takes precedence over val, and every other crop is
    train.  Train crops within ``train_buffer_crops`` rows/columns (including
    diagonals) of a val or test crop in the same tract are marked "buffer", so
    that no training crop's border context covers held-out sky.  The buffer is
    only applied within a tract.
    """
    unknown = set(split_regions) - set(HELD_OUT_SPLITS)
    if unknown:
        raise ValueError(f"split_regions keys must be in {HELD_OUT_SPLITS}: {unknown}")

    positions = {name: padded_crop_position(name) for name in file_names}
    assignment = dict.fromkeys(file_names, "train")
    for split in reversed(HELD_OUT_SPLITS):  # assign val first so test overrides it
        for region in split_regions.get(split) or []:
            (row0, row1), (col0, col1) = region["rows"], region["cols"]
            matched = [
                name
                for name, (tract, row, col) in positions.items()
                if tract == int(region["tract"])
                and row0 <= row <= row1
                and col0 <= col <= col1
            ]
            if not matched:
                raise ValueError(f"{split} region matches no cached crops: {region}")
            for name in matched:
                assignment[name] = split

    held_out = {positions[n] for n, split in assignment.items() if split != "train"}
    offsets = range(-train_buffer_crops, train_buffer_crops + 1)
    for name, split in assignment.items():
        if split != "train":
            continue
        tract, row, col = positions[name]
        if any(
            (tract, row + dr, col + dc) in held_out for dr in offsets for dc in offsets
        ):
            assignment[name] = "buffer"
    return assignment


class LensingDC2DataModule(CachedSimulatedDataModule):
    """Data module for lensing DC2 survey data."""

    BANDS = ("u", "g", "r", "i", "z", "y")

    def __init__(
        self,
        tile_slen: int,
        splits: str | None,
        avg_ellip_kernel_size: int,
        avg_ellip_kernel_sigma: int,
        redshift_quantiles: List[float],
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        train_transforms: List,
        shuffle_file_order: bool,
        splits_type: str = "percent",
        exclude_file_list_path: str | None = None,
        tracts: List[int] | None = None,
        split_regions: dict | None = None,
        train_buffer_crops: int = 0,
        **kwargs,
    ):
        super().__init__(
            splits=splits,
            batch_size=batch_size,
            num_workers=num_workers,
            cached_data_path=cached_data_path,
            train_transforms=train_transforms,
            nontrain_transforms=[],
            subset_fraction=None,
            shuffle_file_order=shuffle_file_order,
            splits_type=splits_type,
            exclude_file_list_path=exclude_file_list_path,
        )

        self.tile_slen = tile_slen
        self.avg_ellip_kernel_size = avg_ellip_kernel_size
        self.avg_ellip_kernel_sigma = avg_ellip_kernel_sigma
        self.redshift_quantiles = redshift_quantiles
        self.tracts = None if tracts is None else {int(t) for t in tracts}
        if OmegaConf.is_config(split_regions):
            split_regions = OmegaConf.to_container(split_regions, resolve=True)
        self.split_regions = split_regions
        self.train_buffer_crops = int(train_buffer_crops)
        self.split_assignment = None
        if (splits is None) == (split_regions is None):
            raise ValueError("Set exactly one of splits and split_regions")
        if split_regions is not None:
            if shuffle_file_order:
                raise ValueError("split_regions requires shuffle_file_order=false")
            region_tracts = {
                int(region["tract"])
                for regions in split_regions.values()
                for region in regions or []
            }
            if self.tracts is not None and not region_tracts <= self.tracts:
                raise ValueError(
                    f"split_regions use tracts {sorted(region_tracts - self.tracts)} "
                    f"that are not in tracts {sorted(self.tracts)}"
                )

    def _filter_file_names(self, file_names: List[str]) -> List[str]:
        """Keep only cache files named ``tract_<tract>_...`` for the selected tracts."""
        if self.tracts is None:
            return file_names
        selected = []
        for file_name in file_names:
            match = re.match(r"tract_(\d+)_", file_name)
            if match is not None and int(match.group(1)) in self.tracts:
                selected.append(file_name)
        logging.getLogger("LensingDC2DataModule").info(
            "Selected %d/%d cached files from tracts %s",
            len(selected),
            len(file_names),
            sorted(self.tracts),
        )
        return selected

    def _load_file_paths_and_slices(self):
        if self.split_regions is None:
            super()._load_file_paths_and_slices()
            return

        file_names = self._list_file_names()
        self.split_assignment = assign_crop_splits(
            file_names, self.split_regions, self.train_buffer_crops
        )
        by_split = {
            split: [n for n in file_names if self.split_assignment[n] == split]
            for split in ("train", "val", "test", "buffer")
        }
        for split in ("train", "val", "test"):
            if not by_split[split]:
                raise ValueError(f"split_regions leave the {split} split empty")
        logging.getLogger("LensingDC2DataModule").info(
            "Region split: train=%d val=%d test=%d buffer (dropped)=%d",
            *(len(by_split[s]) for s in ("train", "val", "test", "buffer")),
        )

        # Lay files out as train | val | test so the base class slices apply.
        ordered = by_split["train"] + by_split["val"] + by_split["test"]
        self.file_paths = [os.path.join(str(self.cached_data_path), f) for f in ordered]
        n_train, n_val = len(by_split["train"]), len(by_split["val"])
        self.slices = [
            slice(0, n_train),
            slice(n_train, n_train + n_val),
            slice(n_train + n_val, len(ordered)),
        ]

    @staticmethod
    def squeeze_tile_dict(tile_dict):
        """Remove batch dimension from tile dict values."""
        tile_dict_copy = copy.copy(tile_dict)
        return {k: v.squeeze(0) for k, v in tile_dict_copy.items()}

    def to_tile_catalog(self, full_catalog, height, width):
        """Convert full catalog to tile-level catalog.

        Returns a plain dict (not BaseTileCatalog).
        """
        plocs = full_catalog["plocs"].reshape(1, -1, 2)
        source_tile_coords = torch.div(plocs, self.tile_slen, rounding_mode="trunc").to(
            torch.int64
        )
        n_tiles_h = math.ceil(height / self.tile_slen)
        n_tiles_w = math.ceil(width / self.tile_slen)
        stti = source_tile_coords[:, :, 0] * n_tiles_w + source_tile_coords[:, :, 1]
        source_to_tile_indices = stti.unsqueeze(-1).to(dtype=torch.int64)

        tile_cat = {}

        num_tiles = n_tiles_h * n_tiles_w

        redshift_bin = torch.zeros(full_catalog["redshift_nobin"].shape[1])

        for b, q in enumerate(self.redshift_quantiles):
            redshift_bin[full_catalog["redshift_nobin"][0, :, 0] > q] = b

        for k, v in full_catalog.items():
            if k in {"plocs", "plocs_binned", "redshift_nobin", "mag_mask"}:
                continue

            v = v.reshape(self.batch_size, plocs.shape[1], -1)

            v_sum = torch.zeros(self.batch_size, num_tiles, v.shape[-1], dtype=v.dtype)
            v_count = torch.zeros(
                self.batch_size, num_tiles, v.shape[-1], dtype=v.dtype
            )
            add_pos = torch.ones_like(v)

            nanmask = ~torch.isnan(v).any(dim=-1, keepdim=True).expand(
                -1, -1, v.shape[-1]
            )
            v = torch.where(nanmask, v, torch.tensor(0.0))
            add_pos = nanmask.float().to(v.dtype)

            if k == "psf":
                v_sum = v_sum.scatter_add(
                    1, source_to_tile_indices.expand(-1, -1, v.shape[-1]), v
                )
                v_count = v_count.scatter_add(
                    1, source_to_tile_indices.expand(-1, -1, v.shape[-1]), add_pos
                )
            else:
                for b, q in enumerate(self.redshift_quantiles):
                    mask = redshift_bin == b
                    v_sum_bin = (
                        v_sum[..., b]
                        .unsqueeze(-1)
                        .scatter_add(
                            1,
                            source_to_tile_indices[:, mask, :],
                            v[..., b].unsqueeze(-1),
                        )
                    )
                    v_sum[..., b] = v_sum_bin[..., 0]
                    v_count_bin = (
                        v_count[..., b]
                        .unsqueeze(-1)
                        .scatter_add(
                            1,
                            source_to_tile_indices[:, mask, :],
                            add_pos[..., b].unsqueeze(-1),
                        )
                    )
                    v_count[..., b] = v_count_bin[..., 0]

            tile_cat[f"{k}_sum"] = v_sum.reshape(
                self.batch_size, n_tiles_w, n_tiles_h, v.shape[-1]
            )
            tile_cat[f"{k}_count"] = v_count.reshape(
                self.batch_size, n_tiles_w, n_tiles_h, v.shape[-1]
            )

        return tile_cat  # Return plain dict


class LensingDC2Catalog:
    """Standalone catalog class for lensing DC2 data."""

    def __init__(self, height, width, data):
        self.height = height
        self.width = width
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    @classmethod
    def from_dataframe(cls, catalog, wcs, height, width, **kwargs):
        """Read and process catalog from an in-memory dataframe."""
        galid = torch.from_numpy(catalog["galaxy_id"].values)
        ra = torch.from_numpy(catalog["ra"].values)
        dec = torch.from_numpy(catalog["dec"].values)

        shear1 = torch.from_numpy(catalog["shear_1"].values)
        shear2 = torch.from_numpy(catalog["shear_2"].values)
        complex_shear = shear1 + shear2 * 1j
        convergence = torch.from_numpy(catalog["convergence"].values)
        reduced_shear = complex_shear / (1.0 - convergence)

        ellip1_intrinsic = torch.from_numpy(catalog["ellipticity_1_true_dc2"].values)
        ellip2_intrinsic = torch.from_numpy(catalog["ellipticity_2_true_dc2"].values)
        complex_ellip_intrinsic = ellip1_intrinsic + ellip2_intrinsic * 1j
        complex_ellip_lensed = (complex_ellip_intrinsic + reduced_shear) / (
            1.0 + reduced_shear.conj() * complex_ellip_intrinsic
        )
        ellip1_lensed = torch.view_as_real(complex_ellip_lensed)[..., 0]
        ellip2_lensed = torch.view_as_real(complex_ellip_lensed)[..., 1]

        ixx = torch.from_numpy(catalog["Iyy_pixel"].values)  # align coordinate system
        iyy = torch.from_numpy(catalog["Ixx_pixel"].values)  # align coordinate system
        ixy = torch.from_numpy(catalog["Ixy_pixel"].values)
        ellip_lsst = (ixx - iyy + 2j * ixy) / (
            ixx + iyy + 2 * np.sqrt(ixx * iyy - (ixy**2))
        )
        ellip1_lsst = torch.view_as_real(ellip_lsst)[..., 0]
        ellip2_lsst = torch.view_as_real(ellip_lsst)[..., 1]

        redshift = torch.from_numpy(catalog["redshift"].values)

        _, psf_params = get_bands_flux_and_psf(kwargs["bands"], catalog, median=False)

        plocs = plocs_from_ra_dec(ra, dec, wcs).squeeze(0)
        x0_mask = (plocs[:, 0] > 0) & (plocs[:, 0] < height)
        x1_mask = (plocs[:, 1] > 0) & (plocs[:, 1] < width)
        plocs_mask = x0_mask * x1_mask

        galid = galid[plocs_mask]
        ra = ra[plocs_mask]
        dec = dec[plocs_mask]
        plocs = plocs[plocs_mask]

        shear1 = shear1[plocs_mask]
        shear2 = shear2[plocs_mask]
        convergence = convergence[plocs_mask]
        ellip1_lensed = ellip1_lensed[plocs_mask]
        ellip2_lensed = ellip2_lensed[plocs_mask]
        ellip1_lsst = ellip1_lsst[plocs_mask]
        ellip2_lsst = ellip2_lsst[plocs_mask]

        redshift = redshift[plocs_mask]

        psf_params = psf_params[:, :, :, plocs_mask].permute(
            0, 3, 1, 2
        )  # 1, n_obj, 6, 4

        catalog_dict = {
            "galid": galid,
            "ra": ra,
            "dec": dec,
            "plocs": plocs,
            "shear1": shear1,
            "shear2": shear2,
            "convergence": convergence,
            "ellip1_lensed": ellip1_lensed,
            "ellip2_lensed": ellip2_lensed,
            "ellip1_lsst": ellip1_lsst,
            "ellip2_lsst": ellip2_lsst,
            "redshift": redshift,
            "psf_params": psf_params,
        }

        return catalog_dict

    @classmethod
    def bin_catalog_by_redshift(cls, catalog_dict, height, width, **kwargs):
        """Bin catalog by redshift quantiles."""
        nobj = catalog_dict["galid"].shape[0]

        redshift_bin = torch.zeros_like(catalog_dict["redshift"])
        num_redshift_bins = len(kwargs["redshift_quantiles"])

        for i in range(num_redshift_bins):
            redshift_bin[catalog_dict["redshift"] > kwargs["redshift_quantiles"][i]] = i

        plocs_redbin = torch.zeros(catalog_dict["plocs"].shape[0], 2, num_redshift_bins)
        shear1_redbin = torch.zeros(catalog_dict["shear1"].shape[0], num_redshift_bins)
        shear2_redbin = torch.zeros(catalog_dict["shear2"].shape[0], num_redshift_bins)
        convergence_redbin = torch.zeros(
            catalog_dict["convergence"].shape[0], num_redshift_bins
        )
        ra_redbin = torch.zeros(catalog_dict["ra"].shape[0], num_redshift_bins)
        dec_redbin = torch.zeros(catalog_dict["dec"].shape[0], num_redshift_bins)
        ellip1_lensed_redbin = torch.zeros(
            catalog_dict["ellip1_lensed"].shape[0], num_redshift_bins
        )
        ellip2_lensed_redbin = torch.zeros(
            catalog_dict["ellip2_lensed"].shape[0], num_redshift_bins
        )
        ellip1_lsst_redbin = torch.zeros(
            catalog_dict["ellip1_lsst"].shape[0], num_redshift_bins
        )
        ellip2_lsst_redbin = torch.zeros(
            catalog_dict["ellip2_lsst"].shape[0], num_redshift_bins
        )
        redshift_redbin = torch.zeros(
            catalog_dict["redshift"].shape[0], num_redshift_bins
        )

        for i in range(num_redshift_bins):
            num = torch.sum(redshift_bin == i)
            plocs_redbin[:num, :, i] = catalog_dict["plocs"][redshift_bin == i]
            shear1_redbin[:num, i] = catalog_dict["shear1"][redshift_bin == i]
            shear2_redbin[:num, i] = catalog_dict["shear2"][redshift_bin == i]
            convergence_redbin[:num, i] = catalog_dict["convergence"][redshift_bin == i]
            ra_redbin[:num, i] = catalog_dict["ra"][redshift_bin == i]
            dec_redbin[:num, i] = catalog_dict["dec"][redshift_bin == i]
            ellip1_lensed_redbin[:num, i] = catalog_dict["ellip1_lensed"][
                redshift_bin == i
            ]
            ellip2_lensed_redbin[:num, i] = catalog_dict["ellip2_lensed"][
                redshift_bin == i
            ]
            ellip1_lsst_redbin[:num, i] = catalog_dict["ellip1_lsst"][redshift_bin == i]
            ellip2_lsst_redbin[:num, i] = catalog_dict["ellip2_lsst"][redshift_bin == i]
            redshift_redbin[:num, i] = catalog_dict["redshift"][redshift_bin == i]

        d = {
            "ra": ra_redbin.reshape(1, nobj, num_redshift_bins),
            "dec": dec_redbin.reshape(1, nobj, num_redshift_bins),
            "plocs": catalog_dict["plocs"].reshape(1, nobj, 2),
            "plocs_binned": plocs_redbin.reshape(1, nobj, 2, num_redshift_bins),
            "shear1": shear1_redbin.reshape(1, nobj, num_redshift_bins),
            "shear2": shear2_redbin.reshape(1, nobj, num_redshift_bins),
            "convergence": convergence_redbin.reshape(1, nobj, num_redshift_bins),
            "ellip1_lensed": ellip1_lensed_redbin.reshape(1, nobj, num_redshift_bins),
            "ellip2_lensed": ellip2_lensed_redbin.reshape(1, nobj, num_redshift_bins),
            "ellip1_lsst": ellip1_lsst_redbin.reshape(1, nobj, num_redshift_bins),
            "ellip2_lsst": ellip2_lsst_redbin.reshape(1, nobj, num_redshift_bins),
            "redshift": redshift_redbin.reshape(1, nobj, num_redshift_bins),
            "redshift_nobin": catalog_dict["redshift"].reshape(1, nobj, 1),
            "psf": catalog_dict["psf_params"],
        }

        return cls(height, width, d)
