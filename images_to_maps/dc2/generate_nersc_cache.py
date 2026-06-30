"""Generate NPE-ready DC2 cache files from the NERSC DC2 layout.

This converter reads coadd images and tract-level truth/object parquet files from
``/nfs/turbo/lsa-regier/dc2_nersc`` and joins the truth rows to CosmoDC2 lensing
fields.  It writes the same cached datum schema consumed by the current
``images_to_maps`` NPE and flow-matching trainers.

Example:
    python -u images_to_maps/dc2/generate_nersc_cache.py \
        --tracts 3828 \
        --patches 0,5

    # Use another training config for generation defaults:
    python -u images_to_maps/dc2/generate_nersc_cache.py \
        --config images_to_maps/dc2/config_train_flowmatching.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import GCRCatalogs
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from astropy.io import fits
from GCRCatalogs import GCRQuery
from omegaconf import OmegaConf

from images_to_maps.dc2.dc2 import LensingDC2Catalog, LensingDC2DataModule
from images_to_maps.dc2.utils import (
    map_nested_dicts,
    plocs_from_ra_dec,
    unpack_dict,
    wcs_from_wcs_header_str,
)
from images_to_maps.utils.weighted_avg_ellip import compute_weighted_avg_ellip

BANDS = ("u", "g", "r", "i", "z", "y")
DEFAULT_DC2_NERSC_ROOT = Path("/nfs/turbo/lsa-regier/dc2_nersc")
DEFAULT_COSMODC2_ROOT = Path(
    "/nfs/turbo/lsa-regier/lsstdesc-public/dc2/cosmoDC2_v1.1.4"
)
DEFAULT_OUTPUT = Path("/nfs/turbo/lsa-regier/dc2_wl")
DEFAULT_CONFIG = Path(__file__).with_name("config_train_npe.yaml")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def format_eta(start_time: float, completed: int, total: int) -> str:
    if completed <= 0 or total <= completed:
        return "unknown" if completed <= 0 else "0s"
    elapsed = time.monotonic() - start_time
    seconds_per_item = elapsed / completed
    return format_duration(seconds_per_item * (total - completed))


def setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


TRUTH_COLUMNS = [
    "truth_type",
    "ra",
    "dec",
    "redshift",
    "flux_u",
    "flux_g",
    "flux_r",
    "flux_i",
    "flux_z",
    "flux_y",
    "tract",
    "patch",
    "cosmodc2_hp",
    "cosmodc2_id",
    "match_objectId",
    "is_unique_truth_entry",
]
OBJECT_COLUMNS = [
    "objectId",
    "Ixx_pixel",
    "Iyy_pixel",
    "Ixy_pixel",
    "IxxPSF_pixel_u",
    "IxxPSF_pixel_g",
    "IxxPSF_pixel_r",
    "IxxPSF_pixel_i",
    "IxxPSF_pixel_z",
    "IxxPSF_pixel_y",
    "IyyPSF_pixel_u",
    "IyyPSF_pixel_g",
    "IyyPSF_pixel_r",
    "IyyPSF_pixel_i",
    "IyyPSF_pixel_z",
    "IyyPSF_pixel_y",
    "IxyPSF_pixel_u",
    "IxyPSF_pixel_g",
    "IxyPSF_pixel_r",
    "IxyPSF_pixel_i",
    "IxyPSF_pixel_z",
    "IxyPSF_pixel_y",
    "psf_fwhm_u",
    "psf_fwhm_g",
    "psf_fwhm_r",
    "psf_fwhm_i",
    "psf_fwhm_z",
    "psf_fwhm_y",
]
COSMO_COLUMNS = [
    "galaxy_id",
    "ra",
    "dec",
    "ellipticity_1_true",
    "ellipticity_2_true",
    "ellipticity_1_true_dc2",
    "ellipticity_2_true_dc2",
    "shear_1",
    "shear_2",
    "convergence",
]


def cfg_select(cfg, key: str, default=None):
    value = OmegaConf.select(cfg, key, default=default)
    if value is None:
        return default
    return (
        OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
    )


def apply_config_defaults(args) -> None:
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if args.dc2_root is None:
        args.dc2_root = Path(cfg_select(cfg, "paths.dc2", DEFAULT_DC2_NERSC_ROOT))
    if args.cosmodc2_root is None:
        args.cosmodc2_root = Path(
            cfg_select(cfg, "paths.cosmodc2", DEFAULT_COSMODC2_ROOT)
        )
    if args.output is None:
        args.output = Path(
            cfg_select(
                cfg,
                "surveys.dc2.cached_data_path",
                cfg_select(cfg, "paths.dc2_cache", DEFAULT_OUTPUT),
            )
        )
    if args.image_slen is None:
        args.image_slen = int(cfg_select(cfg, "surveys.dc2.image_slen", 4096))
    if args.n_image_split is None:
        args.n_image_split = int(cfg_select(cfg, "surveys.dc2.n_image_split", 2))
    if args.tile_slen is None:
        args.tile_slen = int(cfg_select(cfg, "surveys.dc2.tile_slen", 256))
    if args.avg_ellip_kernel_size is None:
        args.avg_ellip_kernel_size = int(
            cfg_select(cfg, "surveys.dc2.avg_ellip_kernel_size", 15)
        )
    if args.avg_ellip_kernel_sigma is None:
        args.avg_ellip_kernel_sigma = int(
            cfg_select(cfg, "surveys.dc2.avg_ellip_kernel_sigma", 15)
        )
    if args.redshift_quantiles is None:
        args.redshift_quantiles = list(
            cfg_select(
                cfg,
                "surveys.dc2.redshift_quantiles",
                [0.00, 0.762988, 1.120420, 1.592735],
            )
        )


def parse_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_patches(value: str | None) -> set[str] | None:
    if not value:
        return None
    patches = set()
    for raw in value.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        if "," not in raw and "_" in raw:
            raw = raw.replace("_", ",")
        patches.add(raw)
    return patches


def band_root(dc2_root: Path, band: str) -> Path:
    if band == "u":
        return dc2_root / "run2.2i-dr6-v2-u" / "deepCoadd" / band
    return dc2_root / "run2.2i-dr6-v2-grizy" / "deepCoadd-results" / band


def coadd_path(dc2_root: Path, band: str, tract: int, patch: str) -> Path:
    if band == "u":
        return band_root(dc2_root, band) / str(tract) / f"{patch}.fits"
    return (
        band_root(dc2_root, band)
        / str(tract)
        / patch
        / f"calexp-{band}-{tract}-{patch}.fits"
    )


def iter_patch_manifest(dc2_root: Path) -> Iterable[tuple[int, str]]:
    # The u-band export uses deepCoadd/{band}/{tract}/{patch}.fits, but grizy
    # uses deepCoadd-results/{band}/{tract}/{patch}/calexp-...fits.  Use r as
    # the manifest source because it follows the populated grizy layout.
    r_root = band_root(dc2_root, "r")
    for path in sorted(r_root.glob("*/*/calexp-r-*-*.fits")):
        tract = int(path.parents[1].name)
        patch = path.parent.name
        if all(coadd_path(dc2_root, band, tract, patch).exists() for band in BANDS):
            yield tract, patch


def read_patch_image(dc2_root: Path, tract: int, patch: str, image_slen: int):
    images = []
    wcs_header_str = None
    for band in BANDS:
        path = coadd_path(dc2_root, band, tract, patch)
        with fits.open(path, memmap=False) as hdul:
            data = hdul[1].data
            if data.shape[0] < image_slen or data.shape[1] < image_slen:
                raise ValueError(
                    f"{path} has shape {data.shape}, smaller than image_slen={image_slen}"
                )
            if wcs_header_str is None:
                wcs_header_str = hdul[1].header.tostring()
            crop = np.array(data[:image_slen, :image_slen], copy=True)
            images.append(torch.nan_to_num(torch.from_numpy(crop)))
    return torch.stack(images), wcs_header_str


def read_truth_tract(dc2_root: Path, tract: int) -> pd.DataFrame:
    truth = pd.read_parquet(truth_path(dc2_root, tract), columns=TRUTH_COLUMNS)
    return truth[(truth["truth_type"] == 1) & (truth["cosmodc2_id"] >= 0)].copy()


def select_truth_patch(truth: pd.DataFrame, patch: str) -> pd.DataFrame:
    truth = truth[truth["patch"] == patch].copy()
    truth.drop_duplicates(subset=["cosmodc2_id"], inplace=True)
    return truth


def read_truth_patch(dc2_root: Path, tract: int, patch: str) -> pd.DataFrame:
    return select_truth_patch(read_truth_tract(dc2_root, tract), patch)


def read_object_tract(dc2_root: Path, tract: int) -> pd.DataFrame:
    path = dc2_root / "object_dpdd_only" / f"object_dpdd_tract{tract}.parquet"
    return pd.read_parquet(path, columns=OBJECT_COLUMNS)


def truth_path(dc2_root: Path, tract: int) -> Path:
    return (
        dc2_root
        / "truth_merged_summary_v1-0-0"
        / "match_dr6_v2"
        / f"truth_tract{tract}.parquet"
    )


def truth_tract_bounds(dc2_root: Path) -> dict[int, tuple[float, float, float, float]]:
    bounds = {}
    truth_dir = dc2_root / "truth_merged_summary_v1-0-0" / "match_dr6_v2"
    for path in sorted(truth_dir.glob("truth_tract*.parquet")):
        tract = int(path.stem.replace("truth_tract", ""))
        metadata = pq.ParquetFile(path).metadata
        column_indices = {
            metadata.schema.column(i).name: i for i in range(metadata.num_columns)
        }
        ra_idx = column_indices["ra"]
        dec_idx = column_indices["dec"]
        ra_min = dec_min = float("inf")
        ra_max = dec_max = float("-inf")
        for row_group_idx in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_idx)
            ra_stats = row_group.column(ra_idx).statistics
            dec_stats = row_group.column(dec_idx).statistics
            if ra_stats is None or dec_stats is None:
                raise ValueError(f"Missing RA/Dec parquet statistics for {path}")
            ra_min = min(ra_min, float(ra_stats.min))
            ra_max = max(ra_max, float(ra_stats.max))
            dec_min = min(dec_min, float(dec_stats.min))
            dec_max = max(dec_max, float(dec_stats.max))
        bounds[tract] = (ra_min, ra_max, dec_min, dec_max)
    return bounds


def patch_sky_bounds(
    wcs_header_str: str, height: int, width: int
) -> tuple[float, float, float, float]:
    wcs = wcs_from_wcs_header_str(wcs_header_str)
    xs = np.array([0, width, 0, width, width / 2])
    ys = np.array([0, 0, height, height, height / 2])
    ra, dec = wcs.pixel_to_world_values(xs, ys)
    return (
        float(np.nanmin(ra)),
        float(np.nanmax(ra)),
        float(np.nanmin(dec)),
        float(np.nanmax(dec)),
    )


def bounds_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    padding: float = 0.0,
) -> bool:
    a_ra_min, a_ra_max, a_dec_min, a_dec_max = a
    b_ra_min, b_ra_max, b_dec_min, b_dec_max = b
    return (
        a_ra_min <= b_ra_max + padding
        and a_ra_max >= b_ra_min - padding
        and a_dec_min <= b_dec_max + padding
        and a_dec_max >= b_dec_min - padding
    )


def overlapping_truth_tracts(
    patch_bounds: tuple[float, float, float, float],
    all_truth_bounds: dict[int, tuple[float, float, float, float]],
    padding: float,
) -> list[int]:
    return [
        tract
        for tract, tract_bounds in sorted(all_truth_bounds.items())
        if bounds_overlap(patch_bounds, tract_bounds, padding=padding)
    ]


def put_nominal_tract_first(tracts: list[int], nominal_tract: int) -> list[int]:
    if nominal_tract not in tracts:
        return tracts
    return [nominal_tract] + [tract for tract in tracts if tract != nominal_tract]


def load_cached_tract(
    cache: OrderedDict[int, pd.DataFrame],
    tract: int,
    max_tracts: int,
    label: str,
    load_fn,
) -> pd.DataFrame:
    cached = cache.get(tract)
    if cached is not None:
        cache.move_to_end(tract)
        return cached

    load_start = time.monotonic()
    logging.info("Loading %s parquet for tract=%s", label, tract)
    value = load_fn(tract)
    cache[tract] = value
    logging.info(
        "Loaded %s parquet for tract=%s rows=%d elapsed=%s",
        label,
        tract,
        len(value),
        format_duration(time.monotonic() - load_start),
    )
    while len(cache) > max_tracts:
        evicted_tract, _ = cache.popitem(last=False)
        logging.info("Evicted %s parquet for tract=%s from cache", label, evicted_tract)
    return value


def filter_truth_to_bounds(
    truth: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    padding: float,
) -> pd.DataFrame:
    ra_min, ra_max, dec_min, dec_max = bounds
    return truth[
        (truth["ra"] >= ra_min - padding)
        & (truth["ra"] <= ra_max + padding)
        & (truth["dec"] >= dec_min - padding)
        & (truth["dec"] <= dec_max + padding)
    ].copy()


def filter_catalog_to_wcs(
    catalog: pd.DataFrame,
    wcs_header_str: str,
    height: int,
    width: int,
) -> pd.DataFrame:
    if catalog.empty:
        return catalog
    wcs = wcs_from_wcs_header_str(wcs_header_str)
    ra = torch.from_numpy(catalog["ra"].values)
    dec = torch.from_numpy(catalog["dec"].values)
    plocs = plocs_from_ra_dec(ra, dec, wcs).squeeze(0)
    mask = (
        (plocs[:, 0] > 0)
        & (plocs[:, 0] < height)
        & (plocs[:, 1] > 0)
        & (plocs[:, 1] < width)
    )
    return catalog.loc[mask.numpy()].copy()


def load_cosmodc2_catalog(cosmodc2_root: Path):
    # The catalog name is resolved through lsstdesc-gcr-catalogs, while the HDF5 data
    # root is supplied explicitly so this works with the relocated public archive.
    GCRCatalogs.set_root_dir(str(cosmodc2_root.parent))
    return GCRCatalogs.load_catalog(
        "desc_cosmodc2", {"catalog_root_dir": str(cosmodc2_root)}
    )


def read_cosmo_rows(cosmo_cat, truth: pd.DataFrame) -> pd.DataFrame:
    ids = truth["cosmodc2_id"].dropna().astype(np.int64).unique()
    hpix = truth["cosmodc2_hp"].dropna().astype(np.int64).unique()
    hpix = hpix[hpix >= 0]
    if len(ids) == 0:
        return pd.DataFrame(columns=COSMO_COLUMNS)

    filters = [GCRQuery((lambda x, selected=ids: np.isin(x, selected), "galaxy_id"))]
    native_filters = []
    if len(hpix) > 0:
        native_filters = [
            GCRQuery((lambda x, selected=hpix: np.isin(x, selected), "healpix_pixel"))
        ]

    cosmo = cosmo_cat.get_quantities(
        quantities=COSMO_COLUMNS,
        filters=filters,
        native_filters=native_filters,
    )
    return pd.DataFrame(cosmo)


def build_lensing_dataframe(
    dc2_root: Path,
    cosmo_cat,
    truth_by_tract: dict[int, pd.DataFrame],
    object_by_tract: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    # Build from all truth/object tracts that overlap the patch footprint.  LSST
    # coadd tract images include overlap sky; using only the nominal coadd tract
    # leaves artificial zero target rows/columns at tract boundaries.
    truth_frames = []
    for tract, truth_df in truth_by_tract.items():
        if not truth_df.empty:
            tmp = truth_df.copy()
            tmp["source_truth_tract"] = tract
            truth_frames.append(tmp)
    if not truth_frames:
        return pd.DataFrame()
    truth = pd.concat(truth_frames, ignore_index=True)
    truth.drop_duplicates(subset=["cosmodc2_id"], inplace=True)
    if truth.empty:
        return pd.DataFrame()

    object_frames = []
    for tract, object_df in object_by_tract.items():
        if not object_df.empty:
            tmp = object_df.drop_duplicates(subset=["objectId"]).copy()
            tmp["source_object_tract"] = tract
            object_frames.append(tmp)
    object_df = (
        pd.concat(object_frames, ignore_index=True)
        if object_frames
        else pd.DataFrame(columns=OBJECT_COLUMNS)
    )
    object_df.drop_duplicates(subset=["objectId"], inplace=True)

    merged = truth.merge(
        object_df,
        left_on="match_objectId",
        right_on="objectId",
        how="left",
        suffixes=("", "_object"),
    )

    cosmo = read_cosmo_rows(cosmo_cat, truth)
    merged = merged.merge(
        cosmo,
        left_on="cosmodc2_id",
        right_on="galaxy_id",
        how="left",
        suffixes=("", "_cosmo"),
    )
    merged = merged[~merged["galaxy_id"].isna()].copy()
    return merged


def safe_avg(tile_dict: dict, name: str) -> torch.Tensor:
    total = tile_dict[f"{name}_sum"]
    count = tile_dict[f"{name}_count"]
    return total / (count + (count == 0) * torch.ones_like(count))


def finalize_tile_dict(tile_dict: dict, avg_kernel_size: int, avg_kernel_sigma: int):
    shear1 = safe_avg(tile_dict, "shear1")
    shear2 = safe_avg(tile_dict, "shear2")
    convergence = safe_avg(tile_dict, "convergence")
    ellip1_lensed = safe_avg(tile_dict, "ellip1_lensed")
    ellip2_lensed = safe_avg(tile_dict, "ellip2_lensed")
    ellip1_lsst = safe_avg(tile_dict, "ellip1_lsst")
    ellip2_lsst = safe_avg(tile_dict, "ellip2_lsst")

    tile_dict["shear_1"] = shear1
    tile_dict["shear_2"] = shear2
    tile_dict["shear1_shear2"] = torch.cat([shear1, shear2], dim=-1)
    tile_dict["convergence"] = convergence
    tile_dict["shear1_shear2_convergence"] = torch.cat(
        [shear1, shear2, convergence], dim=-1
    )
    tile_dict["ellip_lensed"] = torch.stack(
        (ellip1_lensed.squeeze(-1), ellip2_lensed.squeeze(-1)), dim=-1
    )
    tile_dict["ellip_lsst"] = torch.stack(
        (ellip1_lsst.squeeze(-1), ellip2_lsst.squeeze(-1)), dim=-1
    )
    tile_dict["ellip_lsst_wavg"] = compute_weighted_avg_ellip(
        tile_dict, avg_kernel_size, avg_kernel_sigma
    )
    tile_dict["redshift"] = safe_avg(tile_dict, "redshift")
    tile_dict["ra"] = safe_avg(tile_dict, "ra")
    tile_dict["dec"] = safe_avg(tile_dict, "dec")
    return tile_dict


def make_data_module(args) -> LensingDC2DataModule:
    return LensingDC2DataModule(
        dc2_image_dir="",
        dc2_cat_path="",
        image_slen=args.image_slen,
        n_image_split=args.n_image_split,
        tile_slen=args.tile_slen,
        splits="0:1/1:1/1:1",
        avg_ellip_kernel_size=args.avg_ellip_kernel_size,
        avg_ellip_kernel_sigma=args.avg_ellip_kernel_sigma,
        redshift_quantiles=args.redshift_quantiles,
        batch_size=1,
        num_workers=0,
        cached_data_path=str(args.output),
        train_transforms=[],
        shuffle_file_order=False,
    )


def write_training_examples(
    args,
    data_module: LensingDC2DataModule,
    image: torch.Tensor,
    wcs_header_str: str,
    catalog: pd.DataFrame,
    tract: int,
    patch: str,
) -> int:
    wcs = wcs_from_wcs_header_str(wcs_header_str)
    height, width = image[0].shape
    catalog_dict = LensingDC2Catalog.from_dataframe(
        catalog,
        wcs,
        height,
        width,
        bands=BANDS,
        n_bands=len(BANDS),
    )
    full_cat = LensingDC2Catalog.bin_catalog_by_redshift(
        catalog_dict,
        height,
        width,
        redshift_quantiles=args.redshift_quantiles,
    )
    tile_cat = data_module.to_tile_catalog(full_cat, height, width)
    n_psf_params = full_cat["psf"].shape[-1]
    # Match the historical dc2_lensing_splits behavior, including NaNs where a
    # split has no finite PSF contributors.
    psf_params = (tile_cat["psf_sum"] / tile_cat["psf_count"]).view(
        *tile_cat["psf_sum"].shape[:-1], len(BANDS), n_psf_params
    )
    del tile_cat["psf_sum"]
    del tile_cat["psf_count"]
    tile_dict = data_module.squeeze_tile_dict(tile_cat)
    tile_dict = finalize_tile_dict(
        tile_dict, args.avg_ellip_kernel_size, args.avg_ellip_kernel_sigma
    )

    data_splits = data_module.split_image_and_tile_cat(
        image, tile_dict, tile_dict.keys(), psf_params.squeeze(0)
    )
    data_to_cache = unpack_dict(data_splits)
    patch_slug = patch.replace(",", "_")
    count = 0
    for datum in data_to_cache:
        row = datum["image_height_index"]
        col = datum["image_width_index"]
        out = (
            args.output
            / f"tract_{tract}_patch_{patch_slug}_split_{row}_{col}_size_1.pt"
        )
        if out.exists() and not args.overwrite:
            continue
        datum_clone = map_nested_dicts(
            datum, lambda x: x.clone() if isinstance(x, torch.Tensor) else x
        )
        with out.open("wb") as f:
            torch.save([datum_clone], f)
        count += 1
    return count


def output_exists(args, tract: int, patch: str) -> bool:
    patch_slug = patch.replace(",", "_")
    expected = args.n_image_split**2
    return (
        len(
            list(
                args.output.glob(f"tract_{tract}_patch_{patch_slug}_split_*_size_1.pt")
            )
        )
        == expected
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Training config used for generation defaults. CLI options override it.",
    )
    parser.add_argument("--dc2-root", type=Path, default=None)
    parser.add_argument("--cosmodc2-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tracts", type=parse_ints, default=None)
    parser.add_argument("--patches", type=parse_patches, default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--image-slen", type=int, default=None)
    parser.add_argument("--n-image-split", type=int, default=None)
    parser.add_argument("--tile-slen", type=int, default=None)
    parser.add_argument("--avg-ellip-kernel-size", type=int, default=None)
    parser.add_argument("--avg-ellip-kernel-sigma", type=int, default=None)
    parser.add_argument(
        "--redshift-quantiles",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument(
        "--footprint-padding-deg",
        type=float,
        default=0.02,
        help="RA/Dec padding used when selecting neighboring truth tracts for each patch.",
    )
    parser.add_argument(
        "--tract-cache-size",
        type=int,
        default=12,
        help="Maximum number of truth/object tract parquet tables to keep in memory.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path for a copy of the generation log.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_config_defaults(args)
    setup_logging(args.log_file)
    run_start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)

    selected_tracts = set(args.tracts) if args.tracts else None
    selected_patches = args.patches
    manifest = []
    for tract, patch in iter_patch_manifest(args.dc2_root):
        if selected_tracts is not None and tract not in selected_tracts:
            continue
        if selected_patches is not None and patch not in selected_patches:
            continue
        manifest.append((tract, patch))
        if args.max_patches is not None and len(manifest) >= args.max_patches:
            break

    total = len(manifest)
    logging.info("Found %d tract/patch groups to process", total)
    logging.info("Generation config: %s", args.config)
    logging.info("Input DC2 root: %s", args.dc2_root)
    logging.info("Input CosmoDC2 root: %s", args.cosmodc2_root)
    logging.info("Output directory: %s", args.output)
    logging.info(
        "Generation options: overwrite=%s skip_errors=%s image_slen=%d "
        "n_image_split=%d tile_slen=%d tract_cache_size=%d redshift_quantiles=%s",
        args.overwrite,
        args.skip_errors,
        args.image_slen,
        args.n_image_split,
        args.tile_slen,
        args.tract_cache_size,
        args.redshift_quantiles,
    )

    setup_start = time.monotonic()
    logging.info("Loading CosmoDC2 catalog")
    cosmo_cat = load_cosmodc2_catalog(args.cosmodc2_root)
    data_module = make_data_module(args)
    logging.info("Loading truth tract sky bounds")
    all_truth_bounds = truth_tract_bounds(args.dc2_root)
    logging.info(
        "Loaded truth bounds for %d tracts with footprint_padding_deg=%s",
        len(all_truth_bounds),
        args.footprint_padding_deg,
    )
    logging.info(
        "Finished setup in %s", format_duration(time.monotonic() - setup_start)
    )

    object_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
    truth_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
    written = 0
    skipped_existing = 0
    empty_catalogs = []
    failures = []
    current_tract = None
    tract_start = None
    tract_written = 0
    tract_patches = 0

    for idx, (tract, patch) in enumerate(manifest, start=1):
        if current_tract != tract:
            if current_tract is not None and tract_start is not None:
                logging.info(
                    "Finished tract=%s patches=%d split_files=%d elapsed=%s",
                    current_tract,
                    tract_patches,
                    tract_written,
                    format_duration(time.monotonic() - tract_start),
                )
            current_tract = tract
            tract_start = time.monotonic()
            tract_written = 0
            tract_patches = 0
            logging.info("Starting tract=%s", tract)

        patch_start = time.monotonic()
        if not args.overwrite and output_exists(args, tract, patch):
            skipped_existing += 1
            elapsed = time.monotonic() - patch_start
            logging.info(
                "[%d/%d] skipped existing tract=%s patch=%s elapsed=%s eta=%s",
                idx,
                total,
                tract,
                patch,
                format_duration(elapsed),
                format_eta(run_start, idx, total),
            )
            continue
        try:
            logging.info(
                "[%d/%d] processing tract=%s patch=%s", idx, total, tract, patch
            )
            image, wcs_header_str = read_patch_image(
                args.dc2_root, tract, patch, args.image_slen
            )
            patch_bounds = patch_sky_bounds(
                wcs_header_str, args.image_slen, args.image_slen
            )
            overlap_tracts = overlapping_truth_tracts(
                patch_bounds, all_truth_bounds, args.footprint_padding_deg
            )
            overlap_tracts = put_nominal_tract_first(overlap_tracts, tract)
            if not overlap_tracts:
                empty_catalogs.append((tract, patch))
                logging.warning(
                    "No overlapping truth tracts for tract=%s patch=%s bounds=%s",
                    tract,
                    patch,
                    patch_bounds,
                )
                continue

            truth_by_tract = {}
            object_by_tract = {}
            for overlap_tract in overlap_tracts:
                truth_df = load_cached_tract(
                    truth_cache,
                    overlap_tract,
                    args.tract_cache_size,
                    "truth",
                    lambda selected_tract: read_truth_tract(
                        args.dc2_root, selected_tract
                    ),
                )
                truth_filtered = filter_truth_to_bounds(
                    truth_df, patch_bounds, args.footprint_padding_deg
                )
                if truth_filtered.empty:
                    continue
                truth_by_tract[overlap_tract] = truth_filtered

                object_df = load_cached_tract(
                    object_cache,
                    overlap_tract,
                    args.tract_cache_size,
                    "object",
                    lambda selected_tract: read_object_tract(
                        args.dc2_root, selected_tract
                    ),
                )
                object_by_tract[overlap_tract] = object_df

            catalog_start = time.monotonic()
            logging.info(
                "Building patch lensing catalog for tract=%s patch=%s overlap_tracts=%s",
                tract,
                patch,
                sorted(truth_by_tract),
            )
            catalog = build_lensing_dataframe(
                args.dc2_root,
                cosmo_cat,
                truth_by_tract=truth_by_tract,
                object_by_tract=object_by_tract,
            )
            catalog = filter_catalog_to_wcs(
                catalog, wcs_header_str, args.image_slen, args.image_slen
            )
            logging.info(
                "Built patch lensing catalog for tract=%s patch=%s rows=%d elapsed=%s",
                tract,
                patch,
                len(catalog),
                format_duration(time.monotonic() - catalog_start),
            )
            if catalog.empty:
                empty_catalogs.append((tract, patch))
                logging.warning(
                    "No usable catalog rows for tract=%s patch=%s", tract, patch
                )
                continue
            patch_written = write_training_examples(
                args, data_module, image, wcs_header_str, catalog, tract, patch
            )
            written += patch_written
            tract_written += patch_written
            tract_patches += 1
            logging.info(
                "[%d/%d] finished tract=%s patch=%s split_files=%d elapsed=%s "
                "total_split_files=%d eta=%s",
                idx,
                total,
                tract,
                patch,
                patch_written,
                format_duration(time.monotonic() - patch_start),
                written,
                format_eta(run_start, idx, total),
            )
        except Exception as exc:
            failures.append((tract, patch, repr(exc)))
            logging.exception("Failed tract=%s patch=%s", tract, patch)
            if not args.skip_errors:
                raise

    if current_tract is not None and tract_start is not None:
        logging.info(
            "Finished tract=%s patches=%d split_files=%d elapsed=%s",
            current_tract,
            tract_patches,
            tract_written,
            format_duration(time.monotonic() - tract_start),
        )

    logging.info(
        "Done. Wrote %d split cache files to %s in %s",
        written,
        args.output,
        format_duration(time.monotonic() - run_start),
    )
    logging.info(
        "Summary: requested_patches=%d skipped_existing=%d empty_catalogs=%d "
        "failures=%d",
        total,
        skipped_existing,
        len(empty_catalogs),
        len(failures),
    )
    if empty_catalogs:
        logging.warning("Empty catalog patches: %s", empty_catalogs)
    if failures:
        logging.error("Failed patches:")
        for tract, patch, error in failures:
            logging.error("  tract=%s patch=%s error=%s", tract, patch, error)


if __name__ == "__main__":
    main()
