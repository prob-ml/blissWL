"""Generate padded DC2 crops for interior-only mass-map training.

This generator writes overlapping ``2048 x 2048`` image crops.  Each crop has
an ``8 x 8`` tile target map, but only the central ``4 x 4`` tiles are meant to
contribute to loss/evaluation by default.  The two-tile image border is context.
With the default ``256 x 256`` target tiles, this gives ``4 x 4`` crops per
nominal DC2 patch.

Images are read from the NERSC DC2 layout (``paths.dc2``).  Target maps are
built from the DC2 truth and object tables joined to CosmoDC2 lensing fields.
The join is done once per tract and held in memory, so each crop only filters
the merged catalogs of the tracts it overlaps.  Image pixels and target maps
are written together to one ``.pt`` file per crop in ``cached_data_path``.

The tracts to generate are read from ``surveys.dc2.tracts`` in the config and
can be overridden with ``--tracts``.  Existing crop files are skipped unless
``--overwrite`` is passed.

With ``--write-catalogs``, each merged tract catalog is also saved as
``tract_<tract>.parquet`` (default directory ``/nfs/turbo/lsa-regier/dc2_wl_catalog``)
for analysis notebooks.  The selected tracts are always written; neighboring
tracts are written when crops need them.

The output datum schema extends the existing DC2 cache schema with:

* ``interior_mask``: bool tensor of shape ``8 x 8``.
* ``image_valid_tile_mask``: bool tensor of shape ``8 x 8``.
* ``source_id`` and ``source_names``: provenance for assembled image pixels.
* crop WCS/patch-position metadata for stitched inference.

Examples:
    # All tracts listed in the config.
    python -u images_to_maps/dc2/generate_dc2_cache.py

    # Also save the merged per-tract catalogs to the default catalog directory.
    python -u images_to_maps/dc2/generate_dc2_cache.py --write-catalogs

    # Small smoke test.
    python -u images_to_maps/dc2/generate_dc2_cache.py \
        --tracts 3828 --max-crops 2 --output /tmp/dc2_padded_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import GCRCatalogs
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from astropy.io import fits
from astropy.wcs import WCS
from GCRCatalogs import GCRQuery
from omegaconf import OmegaConf
from scipy.ndimage import map_coordinates

from images_to_maps.dc2.dc2 import LensingDC2Catalog, LensingDC2DataModule
from images_to_maps.dc2.utils import wcs_from_wcs_header_str
from images_to_maps.utils.weighted_avg_ellip import compute_weighted_avg_ellip

BANDS = ("u", "g", "r", "i", "z", "y")
DEFAULT_CONFIG = Path(__file__).with_name("config_train_npe.yaml")
DEFAULT_DC2_NERSC_ROOT = Path("/nfs/turbo/lsa-regier/dc2_nersc")
DEFAULT_COSMODC2_ROOT = Path(
    "/nfs/turbo/lsa-regier/lsstdesc-public/dc2/cosmoDC2_v1.1.4"
)
DEFAULT_OUTPUT = Path("/nfs/turbo/lsa-regier/dc2_wl")
DEFAULT_CATALOG_OUTPUT = Path("/nfs/turbo/lsa-regier/dc2_wl_catalog")
DEFAULT_REDSHIFT_QUANTILES = [0.00, 0.762988, 1.120420, 1.592735]
COPY_WCS_ATOL_DEG = 1.0e-8

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
    *[
        f"{moment}PSF_pixel_{band}"
        for moment in ("Ixx", "Iyy", "Ixy")
        for band in BANDS
    ],
    *[f"psf_fwhm_{band}" for band in BANDS],
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
    """Fill unset CLI options from the training config."""
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
    if args.tracts is None:
        args.tracts = cfg_select(cfg, "surveys.dc2.tracts", None)
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
                cfg, "surveys.dc2.redshift_quantiles", DEFAULT_REDSHIFT_QUANTILES
            )
        )


# --- DC2 NERSC inputs --------------------------------------------------------


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


def truth_path(dc2_root: Path, tract: int) -> Path:
    return (
        dc2_root
        / "truth_merged_summary_v1-0-0"
        / "match_dr6_v2"
        / f"truth_tract{tract}.parquet"
    )


def read_truth_tract(dc2_root: Path, tract: int) -> pd.DataFrame:
    truth = pd.read_parquet(truth_path(dc2_root, tract), columns=TRUTH_COLUMNS)
    return truth[(truth["truth_type"] == 1) & (truth["cosmodc2_id"] >= 0)].copy()


def read_object_tract(dc2_root: Path, tract: int) -> pd.DataFrame:
    path = dc2_root / "object_dpdd_only" / f"object_dpdd_tract{tract}.parquet"
    return pd.read_parquet(path, columns=OBJECT_COLUMNS)


def truth_tract_bounds(dc2_root: Path) -> dict[int, tuple[float, float, float, float]]:
    """RA/Dec bounds of every truth tract, from parquet row-group statistics."""
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


# --- Merged truth/object/CosmoDC2 catalogs -----------------------------------


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
    # GCR returns quantities in arbitrary order; fix it so written catalogs are stable.
    return pd.DataFrame(cosmo, columns=COSMO_COLUMNS)


def build_lensing_dataframe(dc2_root: Path, cosmo_cat, tract: int) -> pd.DataFrame:
    """Join one tract's truth and object tables to CosmoDC2 lensing fields."""
    truth = read_truth_tract(dc2_root, tract)
    truth["source_truth_tract"] = tract
    truth.drop_duplicates(subset=["cosmodc2_id"], inplace=True)
    if truth.empty:
        return pd.DataFrame()

    object_df = read_object_tract(dc2_root, tract)
    object_df = object_df.drop_duplicates(subset=["objectId"]).copy()
    object_df["source_object_tract"] = tract

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


def filter_catalog_to_bounds(
    catalog: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    padding: float,
) -> pd.DataFrame:
    if catalog.empty:
        return catalog
    ra_min, ra_max, dec_min, dec_max = bounds
    return catalog[
        (catalog["ra"] >= ra_min - padding)
        & (catalog["ra"] <= ra_max + padding)
        & (catalog["dec"] >= dec_min - padding)
        & (catalog["dec"] <= dec_max + padding)
    ].copy()


class TractCatalogCache:
    """In-memory LRU cache of per-tract merged lensing catalogs.

    LSST coadd patches include overlap sky, so a crop can need catalog rows from
    several truth tracts.  Each tract is merged once (about 1-2 minutes and
    2 GB in memory) and reused by every crop that overlaps it.  When
    ``write_dir`` is set, each newly built catalog is also saved as parquet.
    """

    def __init__(
        self,
        dc2_root: Path,
        cosmodc2_root: Path,
        max_tracts: int,
        write_dir: Path | None = None,
        overwrite: bool = False,
    ):
        self.dc2_root = dc2_root
        self.cosmodc2_root = cosmodc2_root
        self.max_tracts = max_tracts
        self.write_dir = write_dir
        self.overwrite = overwrite
        self._cosmo_cat = None
        self._cache: OrderedDict[int, pd.DataFrame] = OrderedDict()

    def get(self, tract: int) -> pd.DataFrame:
        cached = self._cache.get(tract)
        if cached is not None:
            self._cache.move_to_end(tract)
            return cached

        if self._cosmo_cat is None:
            logging.info("Loading CosmoDC2 catalog from %s", self.cosmodc2_root)
            self._cosmo_cat = load_cosmodc2_catalog(self.cosmodc2_root)
        load_start = time.monotonic()
        logging.info("Building merged catalog for tract=%s", tract)
        catalog = build_lensing_dataframe(self.dc2_root, self._cosmo_cat, tract)
        self._cache[tract] = catalog
        logging.info(
            "Built merged catalog for tract=%s rows=%d elapsed=%s",
            tract,
            len(catalog),
            format_duration(time.monotonic() - load_start),
        )
        if self.write_dir is not None and (
            self.overwrite or not self.catalog_path(tract).exists()
        ):
            self.write_dir.mkdir(parents=True, exist_ok=True)
            catalog.to_parquet(self.catalog_path(tract), index=False)
            logging.info("Wrote merged catalog %s", self.catalog_path(tract))
        while len(self._cache) > self.max_tracts:
            evicted_tract, _ = self._cache.popitem(last=False)
            logging.info("Evicted merged catalog for tract=%s", evicted_tract)
        return catalog

    def catalog_path(self, tract: int) -> Path:
        return self.write_dir / f"tract_{tract}.parquet"

    def for_bounds(
        self,
        tracts: list[int],
        bounds: tuple[float, float, float, float],
        padding: float,
    ) -> pd.DataFrame:
        frames = []
        for tract in tracts:
            filtered = filter_catalog_to_bounds(self.get(tract), bounds, padding)
            if not filtered.empty:
                frames.append(filtered)
        if not frames:
            return pd.DataFrame()
        catalog = pd.concat(frames, ignore_index=True)
        catalog.drop_duplicates(subset=["cosmodc2_id"], inplace=True)
        return catalog


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


# --- Padded crops ------------------------------------------------------------


@dataclass(frozen=True)
class PatchFootprint:
    tract: int
    patch: str
    path_by_band: dict[str, Path]
    wcs_header_str: str
    width: int
    height: int
    bounds: tuple[float, float, float, float]

    @property
    def wcs(self) -> WCS:
        return wcs_from_wcs_header_str(self.wcs_header_str)

    @property
    def source_name(self) -> str:
        return f"{self.tract}_{self.patch}"


@dataclass(frozen=True)
class CropSpec:
    nominal_tract: int
    nominal_patch: str
    crop_row: int
    crop_col: int
    origin_x: int
    origin_y: int
    index: int


def parse_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def header_shape(header) -> tuple[int, int]:
    width = int(header.get("ZNAXIS1", header.get("NAXIS1")))
    height = int(header.get("ZNAXIS2", header.get("NAXIS2")))
    return height, width


def read_patch_footprint(dc2_root: Path, tract: int, patch: str) -> PatchFootprint:
    path_by_band = {band: coadd_path(dc2_root, band, tract, patch) for band in BANDS}
    with fits.open(path_by_band["r"], memmap=True) as hdul:
        header = hdul[1].header
        height, width = header_shape(header)
        wcs_header_str = header.tostring()
    bounds = patch_sky_bounds(wcs_header_str, height, width)
    return PatchFootprint(
        tract=tract,
        patch=patch,
        path_by_band=path_by_band,
        wcs_header_str=wcs_header_str,
        width=width,
        height=height,
        bounds=bounds,
    )


def iter_patch_manifest_for_tracts(
    dc2_root: Path, tracts: set[int] | list[int] | tuple[int, ...]
):
    r_root = band_root(dc2_root, "r")
    for tract in sorted(tracts):
        tract_root = r_root / str(tract)
        for path in sorted(tract_root.glob("*/calexp-r-*-*.fits")):
            patch = path.parent.name
            if all(coadd_path(dc2_root, band, tract, patch).exists() for band in BANDS):
                yield tract, patch


def load_patch_footprints(
    dc2_root: Path, tracts: set[int] | list[int] | tuple[int, ...]
) -> dict[tuple[int, str], PatchFootprint]:
    footprints = {}
    for tract, patch in iter_patch_manifest_for_tracts(dc2_root, tracts):
        footprints[(tract, patch)] = read_patch_footprint(dc2_root, tract, patch)
    return footprints


def tract_footprint_bounds(
    footprints: dict[tuple[int, str], PatchFootprint], tract: int
) -> tuple[float, float, float, float]:
    tract_patches = [fp for (t, _), fp in footprints.items() if t == tract]
    if not tract_patches:
        raise ValueError(f"No patch footprints found for tract={tract}")
    return (
        min(fp.bounds[0] for fp in tract_patches),
        max(fp.bounds[1] for fp in tract_patches),
        min(fp.bounds[2] for fp in tract_patches),
        max(fp.bounds[3] for fp in tract_patches),
    )


def crop_wcs(anchor_wcs: WCS, origin_x: int, origin_y: int) -> WCS:
    wcs = anchor_wcs.deepcopy()
    wcs.wcs.crpix = anchor_wcs.wcs.crpix - np.array([origin_x, origin_y], dtype=float)
    return wcs


def crop_bounds(wcs: WCS, crop_slen: int) -> tuple[float, float, float, float]:
    xs = np.array([0, crop_slen, 0, crop_slen, crop_slen / 2])
    ys = np.array([0, 0, crop_slen, crop_slen, crop_slen / 2])
    ra, dec = wcs.pixel_to_world_values(xs, ys)
    return (
        float(np.nanmin(ra)),
        float(np.nanmax(ra)),
        float(np.nanmin(dec)),
        float(np.nanmax(dec)),
    )


def patch_sort_key(
    item: tuple[tuple[int, str], PatchFootprint],
) -> tuple[int, int, int]:
    (tract, patch), _footprint = item
    patch_x, patch_y = (int(value) for value in patch.split(","))
    return tract, patch_y, patch_x


def iter_crop_specs_for_patches(
    selected_footprints: dict[tuple[int, str], PatchFootprint],
    tile_slen: int,
    interior_tiles: int,
    border_tiles: int,
) -> list[CropSpec]:
    specs: list[CropSpec] = []
    crop_index = 0
    for (tract, patch), footprint in sorted(
        selected_footprints.items(), key=patch_sort_key
    ):
        patch_tiles_x = footprint.width // tile_slen
        patch_tiles_y = footprint.height // tile_slen
        n_cols = (patch_tiles_x + interior_tiles - 1) // interior_tiles
        n_rows = (patch_tiles_y + interior_tiles - 1) // interior_tiles
        for row in range(n_rows):
            origin_y = (row * interior_tiles - border_tiles) * tile_slen
            for col in range(n_cols):
                origin_x = (col * interior_tiles - border_tiles) * tile_slen
                specs.append(
                    CropSpec(
                        nominal_tract=tract,
                        nominal_patch=patch,
                        crop_row=row,
                        crop_col=col,
                        origin_x=origin_x,
                        origin_y=origin_y,
                        index=crop_index,
                    )
                )
                crop_index += 1
    return specs


class PatchImageCache:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self._cache: OrderedDict[tuple[int, str, str], np.ndarray] = OrderedDict()

    def get(self, footprint: PatchFootprint, band: str) -> np.ndarray:
        key = (footprint.tract, footprint.patch, band)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        path = footprint.path_by_band[band]
        with fits.open(path, memmap=False) as hdul:
            arr = np.asarray(hdul[1].data, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        self._cache[key] = arr
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return arr


def source_candidates(
    footprints: dict[tuple[int, str], PatchFootprint],
    bounds: tuple[float, float, float, float],
    padding: float,
    nominal_tract: int,
    nominal_patch: str,
) -> list[PatchFootprint]:
    candidates = [
        fp
        for fp in sorted(footprints.values(), key=lambda item: (item.tract, item.patch))
        if bounds_overlap(bounds, fp.bounds, padding=padding)
    ]
    return [
        *[
            fp
            for fp in candidates
            if fp.tract == nominal_tract and fp.patch == nominal_patch
        ],
        *[
            fp
            for fp in candidates
            if fp.tract == nominal_tract and fp.patch != nominal_patch
        ],
        *[fp for fp in candidates if fp.tract != nominal_tract],
    ]


def copy_offset_if_aligned(target_wcs: WCS, source_wcs: WCS) -> tuple[int, int] | None:
    if not np.allclose(
        target_wcs.pixel_scale_matrix,
        source_wcs.pixel_scale_matrix,
        rtol=0.0,
        atol=1.0e-12,
    ):
        return None

    offset = source_wcs.wcs.crpix - target_wcs.wcs.crpix
    rounded = np.rint(offset).astype(int)
    if not np.allclose(offset, rounded, rtol=0.0, atol=1.0e-6):
        return None

    # Verify the integer CRPIX shift really represents the same sky coordinates.
    # This admits neighboring tracts with different CRVAL values when their pixel
    # grids still join cleanly on the sky.
    target_x = np.array([0.0, 0.0, 1024.0, 2047.0])
    target_y = np.array([0.0, 2047.0, 1024.0, 2047.0])
    target_ra, target_dec = target_wcs.pixel_to_world_values(target_x, target_y)
    source_ra, source_dec = source_wcs.pixel_to_world_values(
        target_x + rounded[0], target_y + rounded[1]
    )
    if not (
        np.allclose(target_ra, source_ra, rtol=0.0, atol=COPY_WCS_ATOL_DEG)
        and np.allclose(target_dec, source_dec, rtol=0.0, atol=COPY_WCS_ATOL_DEG)
    ):
        return None
    return int(rounded[0]), int(rounded[1])


def copy_aligned_patch_pixels(
    image: np.ndarray,
    source_id: np.ndarray,
    image_cache: PatchImageCache,
    footprint: PatchFootprint,
    sid: int,
    offset_x: int,
    offset_y: int,
    crop_slen: int,
) -> bool:
    dst_x0 = max(0, -offset_x)
    dst_y0 = max(0, -offset_y)
    dst_x1 = min(crop_slen, footprint.width - offset_x)
    dst_y1 = min(crop_slen, footprint.height - offset_y)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return False

    unfilled = source_id[dst_y0:dst_y1, dst_x0:dst_x1] == 0
    if not unfilled.any():
        return True

    src_x0 = dst_x0 + offset_x
    src_y0 = dst_y0 + offset_y
    src_x1 = dst_x1 + offset_x
    src_y1 = dst_y1 + offset_y
    for band_idx, band in enumerate(BANDS):
        src = image_cache.get(footprint, band)
        dst = image[band_idx, dst_y0:dst_y1, dst_x0:dst_x1]
        src_crop = src[src_y0:src_y1, src_x0:src_x1]
        dst[unfilled] = src_crop[unfilled]
    source_id[dst_y0:dst_y1, dst_x0:dst_x1][unfilled] = sid
    return True


def resample_patch_pixels(
    image: np.ndarray,
    source_id: np.ndarray,
    image_cache: PatchImageCache,
    footprint: PatchFootprint,
    sid: int,
    target_wcs: WCS,
    crop_slen: int,
) -> None:
    yy, xx = np.mgrid[0:crop_slen, 0:crop_slen].astype(np.float64)
    ra, dec = target_wcs.pixel_to_world_values(xx, yy)
    source_wcs = footprint.wcs
    sx, sy = source_wcs.world_to_pixel_values(ra, dec)
    valid = (
        (sx >= 0)
        & (sx <= footprint.width - 1)
        & (sy >= 0)
        & (sy <= footprint.height - 1)
        & (source_id == 0)
    )
    if not valid.any():
        return
    coords = np.vstack([sy[valid], sx[valid]])
    for band_idx, band in enumerate(BANDS):
        src = image_cache.get(footprint, band)
        values = map_coordinates(src, coords, order=1, mode="nearest")
        image[band_idx][valid] = values.astype(np.float32)
    source_id[valid] = sid


def assemble_crop_image(
    footprints: list[PatchFootprint],
    image_cache: PatchImageCache,
    target_wcs: WCS,
    crop_slen: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, str]]:
    image = np.zeros((len(BANDS), crop_slen, crop_slen), dtype=np.float32)
    source_id = np.zeros((crop_slen, crop_slen), dtype=np.int16)
    source_names = {0: "unfilled"}

    for sid, footprint in enumerate(footprints, start=1):
        source_names[sid] = footprint.source_name
        offset = copy_offset_if_aligned(target_wcs, footprint.wcs)
        if offset is not None and copy_aligned_patch_pixels(
            image,
            source_id,
            image_cache,
            footprint,
            sid,
            offset[0],
            offset[1],
            crop_slen,
        ):
            continue
        resample_patch_pixels(
            image, source_id, image_cache, footprint, sid, target_wcs, crop_slen
        )

    return torch.from_numpy(image), torch.from_numpy(source_id), source_names


def tile_valid_mask(source_id: torch.Tensor, tile_slen: int) -> torch.Tensor:
    n_tiles_h = source_id.shape[0] // tile_slen
    n_tiles_w = source_id.shape[1] // tile_slen
    mask = torch.zeros((n_tiles_h, n_tiles_w), dtype=torch.bool)
    for row in range(n_tiles_h):
        for col in range(n_tiles_w):
            tile = source_id[
                row * tile_slen : (row + 1) * tile_slen,
                col * tile_slen : (col + 1) * tile_slen,
            ]
            mask[row, col] = bool((tile != 0).all())
    return mask


def interior_mask(args) -> torch.Tensor:
    mask = torch.zeros((args.crop_tiles, args.crop_tiles), dtype=torch.bool)
    border = args.border_tiles
    mask[
        border : border + args.interior_tiles,
        border : border + args.interior_tiles,
    ] = True
    return mask


def patch_tile_row_col(args, spec: CropSpec) -> torch.Tensor:
    rows = spec.origin_y // args.tile_slen + torch.arange(args.crop_tiles)
    cols = spec.origin_x // args.tile_slen + torch.arange(args.crop_tiles)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([row_grid, col_grid], dim=-1).to(torch.int64)


def patch_output_tile_mask(
    args, spec: CropSpec, footprint: PatchFootprint
) -> torch.Tensor:
    patch_tiles_y = footprint.height // args.tile_slen
    patch_tiles_x = footprint.width // args.tile_slen
    mask = torch.zeros((args.crop_tiles, args.crop_tiles), dtype=torch.bool)
    tile_row_col = patch_tile_row_col(args, spec)
    central = interior_mask(args)
    for row in range(args.crop_tiles):
        for col in range(args.crop_tiles):
            patch_row = int(tile_row_col[row, col, 0].item())
            patch_col = int(tile_row_col[row, col, 1].item())
            mask[row, col] = (
                bool(central[row, col])
                and 0 <= patch_row < patch_tiles_y
                and 0 <= patch_col < patch_tiles_x
            )
    return mask


def make_data_module(args) -> LensingDC2DataModule:
    return LensingDC2DataModule(
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


def psf_params_from_tile_cat(tile_cat: dict, n_bands: int, n_psf_params: int):
    psf_params = (tile_cat["psf_sum"] / tile_cat["psf_count"]).view(
        *tile_cat["psf_sum"].shape[:-1], n_bands, n_psf_params
    )
    del tile_cat["psf_sum"]
    del tile_cat["psf_count"]
    return psf_params


def build_tile_payload(
    args,
    data_module: LensingDC2DataModule,
    target_wcs: WCS,
    catalog: pd.DataFrame,
) -> tuple[dict, torch.Tensor]:
    catalog_dict = LensingDC2Catalog.from_dataframe(
        catalog,
        target_wcs,
        args.crop_slen,
        args.crop_slen,
        bands=BANDS,
        n_bands=len(BANDS),
    )
    full_cat = LensingDC2Catalog.bin_catalog_by_redshift(
        catalog_dict,
        args.crop_slen,
        args.crop_slen,
        redshift_quantiles=args.redshift_quantiles,
    )
    tile_cat = data_module.to_tile_catalog(full_cat, args.crop_slen, args.crop_slen)
    psf_params = psf_params_from_tile_cat(
        tile_cat, len(BANDS), full_cat["psf"].shape[-1]
    )
    tile_dict = data_module.squeeze_tile_dict(tile_cat)
    tile_dict = finalize_tile_dict(
        tile_dict, args.avg_ellip_kernel_size, args.avg_ellip_kernel_sigma
    )
    psf_summary = psf_params.squeeze(0).flatten(0, 1).nanmean(0)
    return tile_dict, psf_summary


def flat_output_file(args, spec: CropSpec) -> Path:
    """Output path for a crop, reusing the existing file name if there is one.

    The ``idx`` field is the crop's position in the run that first wrote it, so it
    depends on which tracts that run selected.  Matching on the rest of the name
    keeps reruns with a different tract selection from writing duplicates.
    """
    patch_label = spec.nominal_patch.replace(",", "_")
    prefix = (
        f"tract_{spec.nominal_tract}_patch_{patch_label}"
        f"_padded_crop_{spec.crop_row}_{spec.crop_col}"
    )
    existing = sorted(args.output.glob(f"{prefix}_idx_*_size_1.pt"))
    if len(existing) > 1:
        raise RuntimeError(f"Multiple cache files for one crop: {existing}")
    if existing:
        return existing[0]
    return args.output / f"{prefix}_idx_{spec.index:06d}_size_1.pt"


def write_example(
    args,
    spec: CropSpec,
    nominal_footprint: PatchFootprint,
    image: torch.Tensor,
    source_id: torch.Tensor,
    source_names: dict[int, str],
    target_wcs: WCS,
    crop_bounds_: tuple[float, float, float, float],
    tile_dict: dict,
    psf_params: torch.Tensor,
    candidate_sources: list[PatchFootprint],
) -> Path:
    mask = interior_mask(args)
    patch_mask = patch_output_tile_mask(args, spec, nominal_footprint)
    image_valid_tile_mask = tile_valid_mask(source_id, args.tile_slen)
    loss_mask = patch_mask & image_valid_tile_mask

    datum = {
        "images": image,
        "tile_catalog": {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in tile_dict.items()
        },
        "psf_params": psf_params,
        "interior_mask": mask,
        "nominal_patch_tile_mask": patch_mask,
        "image_valid_tile_mask": image_valid_tile_mask,
        "loss_mask": loss_mask,
        "source_id": source_id,
        "source_names": source_names,
        "nominal_tract": spec.nominal_tract,
        "nominal_patch": spec.nominal_patch,
        "patch_crop_row_col": torch.tensor(
            [spec.crop_row, spec.crop_col], dtype=torch.int64
        ),
        "patch_crop_origin_xy": torch.tensor(
            [spec.origin_x, spec.origin_y], dtype=torch.int64
        ),
        "patch_tile_row_col": patch_tile_row_col(args, spec),
        "patch_output_shape_tiles": torch.tensor(
            [
                nominal_footprint.height // args.tile_slen,
                nominal_footprint.width // args.tile_slen,
            ],
            dtype=torch.int64,
        ),
        "crop_bounds_ra_dec": torch.tensor(crop_bounds_, dtype=torch.float64),
        "wcs_header_str": target_wcs.to_header().tostring(),
        "source_patches": [
            {"tract": fp.tract, "patch": fp.patch} for fp in candidate_sources
        ],
    }
    out = flat_output_file(args, spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        torch.save([datum], f)
    return out


def crop_summary(
    out: Path,
    spec: CropSpec,
    source_id: torch.Tensor,
    image_valid_tile_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    tile_dict: dict,
) -> dict:
    source_values, source_counts = torch.unique(source_id, return_counts=True)
    loss_indices = torch.nonzero(loss_mask, as_tuple=False)
    return {
        "file": str(out),
        "nominal_tract": spec.nominal_tract,
        "nominal_patch": spec.nominal_patch,
        "crop_index": spec.index,
        "patch_crop_row_col": [spec.crop_row, spec.crop_col],
        "patch_crop_origin_xy": [spec.origin_x, spec.origin_y],
        "loss_tile_indices": loss_indices.tolist(),
        "source_pixel_counts": {
            int(value): int(count) for value, count in zip(source_values, source_counts)
        },
        "valid_image_tiles": int(image_valid_tile_mask.sum().item()),
        "loss_tiles": int(loss_mask.sum().item()),
        "interior_source_count_minmax": [
            float(tile_dict["convergence_count"][loss_mask].sum(-1).min())
            if loss_mask.any()
            else 0.0,
            float(tile_dict["convergence_count"][loss_mask].sum(-1).max())
            if loss_mask.any()
            else 0.0,
        ],
    }


def manifest_file(args, selected_tracts: set[int] | list[int]) -> Path:
    if args.manifest_file is not None:
        return args.manifest_file

    tracts = sorted(selected_tracts)
    if len(tracts) == 1:
        tract_label = f"tract_{tracts[0]}"
    elif len(tracts) <= 8:
        tract_label = "tracts_" + "-".join(str(tract) for tract in tracts)
    else:
        tract_label = f"tracts_{tracts[0]}-{tracts[-1]}_n{len(tracts)}"

    timestamp = time.strftime("%Y%m%dT%H%M%S")
    return (
        args.output
        / f"padded_cache_manifest_{tract_label}_{timestamp}_pid{os.getpid()}.json"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Training config used for paths, tracts, and target-map settings. "
        "CLI options override it.",
    )
    parser.add_argument("--dc2-root", type=Path, default=None)
    parser.add_argument("--cosmodc2-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--tracts",
        type=parse_ints,
        default=None,
        help="Comma-separated tracts. Defaults to surveys.dc2.tracts in the config.",
    )
    parser.add_argument("--max-crops", type=int, default=None)
    parser.add_argument("--tile-slen", type=int, default=None)
    parser.add_argument("--avg-ellip-kernel-size", type=int, default=None)
    parser.add_argument("--avg-ellip-kernel-sigma", type=int, default=None)
    parser.add_argument("--redshift-quantiles", type=float, nargs="+", default=None)
    parser.add_argument("--crop-tiles", type=int, default=8)
    parser.add_argument("--interior-tiles", type=int, default=4)
    parser.add_argument("--border-tiles", type=int, default=2)
    parser.add_argument("--footprint-padding-deg", type=float, default=0.02)
    parser.add_argument("--patch-padding-deg", type=float, default=0.002)
    parser.add_argument("--image-cache-items", type=int, default=48)
    parser.add_argument(
        "--catalog-cache-size",
        type=int,
        default=12,
        help="Maximum number of merged tract catalogs (about 2 GB each) kept in memory.",
    )
    parser.add_argument(
        "--write-catalogs",
        type=Path,
        nargs="?",
        const=DEFAULT_CATALOG_OUTPUT,
        default=None,
        metavar="DIR",
        help=(
            "Also save merged per-tract catalogs as tract_<tract>.parquet, to DIR "
            f"if given, otherwise {DEFAULT_CATALOG_OUTPUT}. Existing files are kept "
            "unless --overwrite is passed."
        ),
    )
    parser.add_argument("--complete-context-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help=(
            "Optional manifest output path. By default, write a unique per-run "
            "manifest filename so parallel jobs in one output directory do not collide."
        ),
    )
    return parser


def normalize_args(args) -> None:
    apply_config_defaults(args)
    if not args.tracts:
        raise ValueError(
            f"No tracts selected: set surveys.dc2.tracts in {args.config} or pass --tracts"
        )
    args.tracts = sorted({int(tract) for tract in args.tracts})
    args.crop_slen = args.crop_tiles * args.tile_slen
    args.stride = args.interior_tiles * args.tile_slen
    if args.crop_tiles != args.interior_tiles + 2 * args.border_tiles:
        raise ValueError(
            "Expected crop_tiles == interior_tiles + 2 * border_tiles; "
            f"got {args.crop_tiles}, {args.interior_tiles}, {args.border_tiles}"
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    normalize_args(args)
    setup_logging(args.log_file)
    run_start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    selected_tracts = args.tracts

    logging.info("Generation config: %s", args.config)
    logging.info("Input DC2 root: %s", args.dc2_root)
    logging.info("Input CosmoDC2 root: %s", args.cosmodc2_root)
    logging.info("Output directory: %s", args.output)
    logging.info("Selected tracts: %s", selected_tracts)
    logging.info(
        "Crop options: crop_tiles=%d interior_tiles=%d tile_slen=%d stride=%d "
        "complete_context_only=%s",
        args.crop_tiles,
        args.interior_tiles,
        args.tile_slen,
        args.stride,
        args.complete_context_only,
    )

    setup_start = time.monotonic()
    logging.info("Loading selected-tract patch footprints")
    selected_footprints = load_patch_footprints(args.dc2_root, selected_tracts)
    logging.info("Loaded %d selected-tract patch footprints", len(selected_footprints))
    logging.info("Loading truth tract sky bounds")
    all_truth_bounds = truth_tract_bounds(args.dc2_root)
    footprint_tracts = set(selected_tracts)
    for tract in selected_tracts:
        tract_bounds = tract_footprint_bounds(selected_footprints, tract)
        footprint_tracts.update(
            overlapping_truth_tracts(
                tract_bounds, all_truth_bounds, args.footprint_padding_deg
            )
        )
    logging.info(
        "Loading patch footprints for selected + neighboring tracts: %s",
        sorted(footprint_tracts),
    )
    footprints = load_patch_footprints(args.dc2_root, footprint_tracts)
    logging.info("Loaded %d patch footprints", len(footprints))
    catalogs = TractCatalogCache(
        args.dc2_root,
        args.cosmodc2_root,
        args.catalog_cache_size,
        write_dir=args.write_catalogs,
        overwrite=args.overwrite,
    )
    if args.write_catalogs is not None:
        logging.info("Writing merged catalogs to %s", args.write_catalogs)
        for tract in selected_tracts:
            if args.overwrite or not catalogs.catalog_path(tract).exists():
                catalogs.get(tract)
    data_module = make_data_module(args)
    logging.info(
        "Finished setup in %s", format_duration(time.monotonic() - setup_start)
    )

    image_cache = PatchImageCache(args.image_cache_items)

    written = 0
    skipped_existing = 0
    skipped_incomplete_context = 0
    empty_catalogs = []
    failures = []
    summaries = []

    all_specs = iter_crop_specs_for_patches(
        selected_footprints,
        args.tile_slen,
        args.interior_tiles,
        args.border_tiles,
    )
    if args.max_crops is not None:
        all_specs = all_specs[: args.max_crops]
    total = len(all_specs)
    logging.info(
        "Found %d patch-local padded crop specs (%d selected patches)",
        total,
        len(selected_footprints),
    )

    for done, spec in enumerate(all_specs, start=1):
        crop_start = time.monotonic()
        out = flat_output_file(args, spec)
        if out.exists() and not args.overwrite:
            skipped_existing += 1
            logging.info(
                "[%d/%d] skipped existing %s eta=%s",
                done,
                total,
                out.name,
                format_eta(run_start, done, total),
            )
            continue

        try:
            nominal_footprint = footprints[(spec.nominal_tract, spec.nominal_patch)]
            target_wcs = crop_wcs(nominal_footprint.wcs, spec.origin_x, spec.origin_y)
            bounds = crop_bounds(target_wcs, args.crop_slen)

            candidates = source_candidates(
                footprints,
                bounds,
                args.patch_padding_deg,
                spec.nominal_tract,
                spec.nominal_patch,
            )
            image, source_id, source_names = assemble_crop_image(
                candidates, image_cache, target_wcs, args.crop_slen
            )
            image_valid_tile_mask = tile_valid_mask(source_id, args.tile_slen)
            if args.complete_context_only and not image_valid_tile_mask.all():
                skipped_incomplete_context += 1
                logging.info(
                    "[%d/%d] skipped incomplete-context tract=%s patch=%s crop=%d "
                    "valid_tiles=%d/%d eta=%s",
                    done,
                    total,
                    spec.nominal_tract,
                    spec.nominal_patch,
                    spec.index,
                    int(image_valid_tile_mask.sum().item()),
                    image_valid_tile_mask.numel(),
                    format_eta(run_start, done, total),
                )
                continue

            overlap_tracts = overlapping_truth_tracts(
                bounds, all_truth_bounds, args.footprint_padding_deg
            )
            overlap_tracts = put_nominal_tract_first(overlap_tracts, spec.nominal_tract)
            catalog = catalogs.for_bounds(
                overlap_tracts, bounds, args.footprint_padding_deg
            )
            if catalog.empty:
                empty_catalogs.append(
                    (spec.nominal_tract, spec.nominal_patch, spec.index)
                )
                logging.warning(
                    "No usable catalog rows for tract=%s patch=%s crop=%d",
                    spec.nominal_tract,
                    spec.nominal_patch,
                    spec.index,
                )
                continue

            tile_dict, psf_params = build_tile_payload(
                args, data_module, target_wcs, catalog
            )
            written_path = write_example(
                args,
                spec,
                nominal_footprint,
                image,
                source_id,
                source_names,
                target_wcs,
                bounds,
                tile_dict,
                psf_params,
                candidates,
            )
            loss_mask = (
                patch_output_tile_mask(args, spec, nominal_footprint)
                & image_valid_tile_mask
            )
            summaries.append(
                crop_summary(
                    written_path,
                    spec,
                    source_id,
                    image_valid_tile_mask,
                    loss_mask,
                    tile_dict,
                )
            )
            written += 1
            logging.info(
                "[%d/%d] wrote tract=%s patch=%s crop=%d row_col=%s file=%s "
                "elapsed=%s written=%d eta=%s",
                done,
                total,
                spec.nominal_tract,
                spec.nominal_patch,
                spec.index,
                (spec.crop_row, spec.crop_col),
                written_path.name,
                format_duration(time.monotonic() - crop_start),
                written,
                format_eta(run_start, done, total),
            )
        except Exception as exc:
            failures.append(
                (spec.nominal_tract, spec.nominal_patch, spec.index, repr(exc))
            )
            logging.exception(
                "Failed tract=%s patch=%s crop=%d",
                spec.nominal_tract,
                spec.nominal_patch,
                spec.index,
            )
            if not args.skip_errors:
                raise

    manifest = {
        "config": str(args.config),
        "dc2_root": str(args.dc2_root),
        "cosmodc2_root": str(args.cosmodc2_root),
        "output": str(args.output),
        "selected_tracts": selected_tracts,
        "crop_tiles": args.crop_tiles,
        "interior_tiles": args.interior_tiles,
        "border_tiles": args.border_tiles,
        "tile_slen": args.tile_slen,
        "crop_slen": args.crop_slen,
        "stride": args.stride,
        "complete_context_only": args.complete_context_only,
        "catalog_output": str(args.write_catalogs) if args.write_catalogs else None,
        "requested_crops": total,
        "written": written,
        "skipped_existing": skipped_existing,
        "skipped_incomplete_context": skipped_incomplete_context,
        "empty_catalogs": empty_catalogs,
        "failures": failures,
        "examples": summaries,
    }
    manifest_path = manifest_file(args, selected_tracts)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logging.info("Wrote manifest to %s", manifest_path)
    logging.info(
        "Done. written=%d skipped_existing=%d skipped_incomplete_context=%d "
        "empty_catalogs=%d failures=%d elapsed=%s",
        written,
        skipped_existing,
        skipped_incomplete_context,
        len(empty_catalogs),
        len(failures),
        format_duration(time.monotonic() - run_start),
    )


if __name__ == "__main__":
    main()
