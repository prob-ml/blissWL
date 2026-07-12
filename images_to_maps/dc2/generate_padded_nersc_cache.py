"""Generate padded DC2 cache crops for interior-only mass-map training.

This generator writes overlapping ``2048 x 2048`` image crops.  Each crop has
an ``8 x 8`` tile target map, but only the central ``4 x 4`` tiles are meant to
contribute to loss/evaluation by default.  The two-tile image border is context.
With the default ``256 x 256`` target tiles, this gives ``4 x 4`` crops per
nominal DC2 patch.

The output datum schema extends the existing DC2 cache schema with:

* ``interior_mask``: bool tensor of shape ``8 x 8``.
* ``image_valid_tile_mask``: bool tensor of shape ``8 x 8``.
* ``source_id`` and ``source_names``: provenance for assembled image pixels.
* crop WCS/patch-position metadata for stitched inference.

Examples:
    # Small smoke test.
    python -u images_to_maps/dc2/generate_padded_nersc_cache.py \
        --tracts 3828 --max-crops 2 --output /tmp/dc2_padded_smoke

    # One full tract, skipping true survey-edge crops with incomplete context.
    python -u images_to_maps/dc2/generate_padded_nersc_cache.py \
        --tracts 3828 \
        --output /nfs/turbo/lsa-regier/dc2_wl_padded_tract3828_test \
        --complete-context-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

from images_to_maps.dc2.dc2 import LensingDC2Catalog, LensingDC2DataModule
from images_to_maps.dc2.generate_merged_catalog_cache import (
    DEFAULT_MERGED_CATALOG_CACHE,
    build_catalog_from_merged_cache,
    merged_catalog_path,
    write_merged_catalog,
)
from images_to_maps.dc2.generate_nersc_cache import (
    BANDS,
    DEFAULT_CONFIG,
    apply_config_defaults,
    band_root,
    bounds_overlap,
    build_lensing_dataframe,
    coadd_path,
    filter_truth_to_bounds,
    finalize_tile_dict,
    format_duration,
    format_eta,
    iter_patch_manifest,
    load_cached_tract,
    load_cosmodc2_catalog,
    overlapping_truth_tracts,
    patch_sky_bounds,
    put_nominal_tract_first,
    read_object_tract,
    read_truth_tract,
    setup_logging,
    truth_tract_bounds,
)
from images_to_maps.dc2.utils import wcs_from_wcs_header_str

DEFAULT_OUTPUT = Path("/nfs/turbo/lsa-regier/dc2_wl_padded")
COPY_WCS_ATOL_DEG = 1.0e-8


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
    dc2_root: Path, tracts: set[int] | list[int] | tuple[int, ...] | None = None
) -> dict[tuple[int, str], PatchFootprint]:
    footprints = {}
    manifest_iter = (
        iter_patch_manifest(dc2_root)
        if tracts is None
        else iter_patch_manifest_for_tracts(dc2_root, tracts)
    )
    for tract, patch in manifest_iter:
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
        dc2_image_dir="",
        dc2_cat_path="",
        image_slen=args.crop_slen,
        n_image_split=1,
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
    patch_label = spec.nominal_patch.replace(",", "_")
    return (
        args.output / f"tract_{spec.nominal_tract}_patch_{patch_label}"
        f"_padded_crop_{spec.crop_row}_{spec.crop_col}"
        f"_idx_{spec.index:06d}_size_1.pt"
    )


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


def ensure_merged_catalog_cache(args, required_tracts: set[int]) -> None:
    if args.merged_catalog_cache is None:
        raise ValueError("merged_catalog_cache must be set before preparation")

    required = sorted(required_tracts)
    to_build = [
        tract
        for tract in required
        if args.overwrite_merged_catalog_cache
        or not merged_catalog_path(args.merged_catalog_cache, tract).exists()
    ]
    if not to_build:
        logging.info(
            "Merged catalog cache already contains all %d required tracts at %s",
            len(required),
            args.merged_catalog_cache,
        )
        return

    logging.info(
        "Preparing merged catalog cache at %s for %d/%d required tracts: %s",
        args.merged_catalog_cache,
        len(to_build),
        len(required),
        to_build,
    )
    setup_start = time.monotonic()
    logging.info("Loading CosmoDC2 catalog for merged catalog preparation")
    cosmo_cat = load_cosmodc2_catalog(args.cosmodc2_root)
    start = time.monotonic()
    for done, tract in enumerate(to_build, start=1):
        tract_start = time.monotonic()
        out, rows, truth_rows = write_merged_catalog(
            args.dc2_root,
            cosmo_cat,
            args.merged_catalog_cache,
            tract,
            overwrite=args.overwrite_merged_catalog_cache,
        )
        logging.info(
            "[%d/%d] prepared merged catalog tract=%s rows=%d truth_rows=%d "
            "file=%s elapsed=%s eta=%s",
            done,
            len(to_build),
            tract,
            rows,
            truth_rows,
            out,
            format_duration(time.monotonic() - tract_start),
            format_eta(start, done, len(to_build)),
        )
    logging.info(
        "Finished merged catalog preparation in %s",
        format_duration(time.monotonic() - setup_start),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dc2-root", type=Path, default=None)
    parser.add_argument("--cosmodc2-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tracts", type=parse_ints, default=None)
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
    parser.add_argument("--tract-cache-size", type=int, default=12)
    parser.add_argument("--image-cache-items", type=int, default=48)
    parser.add_argument(
        "--merged-catalog-cache",
        type=Path,
        default=None,
        help=(
            "Optional directory of per-tract merged catalog parquet files from "
            "generate_merged_catalog_cache.py. When set, skip per-crop "
            "truth/object/CosmoDC2 merging."
        ),
    )
    parser.add_argument(
        "--prepare-merged-catalog-cache",
        action="store_true",
        help=(
            "Before generating padded crops, build any missing merged catalog "
            "tract parquet files required by the selected tracts and their "
            "neighbors. If --merged-catalog-cache is omitted, use the default "
            f"{DEFAULT_MERGED_CATALOG_CACHE}."
        ),
    )
    parser.add_argument(
        "--overwrite-merged-catalog-cache",
        action="store_true",
        help="Rebuild required merged catalog parquet files even if they already exist.",
    )
    parser.add_argument(
        "--merged-catalog-cache-size",
        type=int,
        default=12,
        help="Maximum number of merged catalog tract tables to keep in memory.",
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
    # Reuse the existing config-default helper by setting attributes it expects.
    output_was_provided = args.output is not None
    args.image_slen = None
    args.n_image_split = None
    apply_config_defaults(args)
    if not output_was_provided and args.output == Path("/nfs/turbo/lsa-regier/dc2_wl"):
        args.output = DEFAULT_OUTPUT
    if args.prepare_merged_catalog_cache and args.merged_catalog_cache is None:
        args.merged_catalog_cache = DEFAULT_MERGED_CATALOG_CACHE
    if args.overwrite_merged_catalog_cache and not args.prepare_merged_catalog_cache:
        raise ValueError(
            "--overwrite-merged-catalog-cache requires --prepare-merged-catalog-cache"
        )
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

    selected_tracts = set(args.tracts) if args.tracts else None
    if not selected_tracts:
        selected_tracts = sorted(
            {tract for tract, _patch in iter_patch_manifest(args.dc2_root)}
        )

    logging.info("Output directory: %s", args.output)
    logging.info("Selected tracts: %s", sorted(selected_tracts))
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
    if args.prepare_merged_catalog_cache:
        ensure_merged_catalog_cache(args, footprint_tracts)
    footprints = load_patch_footprints(args.dc2_root, footprint_tracts)
    logging.info("Loaded %d patch footprints", len(footprints))
    if args.merged_catalog_cache is None:
        logging.info("Loading CosmoDC2 catalog")
        cosmo_cat = load_cosmodc2_catalog(args.cosmodc2_root)
    else:
        logging.info("Using merged catalog cache: %s", args.merged_catalog_cache)
        cosmo_cat = None
    data_module = make_data_module(args)
    logging.info(
        "Finished setup in %s", format_duration(time.monotonic() - setup_start)
    )

    truth_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
    object_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
    merged_catalog_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
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
            if args.merged_catalog_cache is not None:
                catalog = build_catalog_from_merged_cache(
                    merged_catalog_cache,
                    args.merged_catalog_cache,
                    overlap_tracts,
                    bounds,
                    args.footprint_padding_deg,
                    args.merged_catalog_cache_size,
                )
            else:
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
                        truth_df, bounds, args.footprint_padding_deg
                    )
                    if truth_filtered.empty:
                        continue
                    truth_by_tract[overlap_tract] = truth_filtered
                    object_by_tract[overlap_tract] = load_cached_tract(
                        object_cache,
                        overlap_tract,
                        args.tract_cache_size,
                        "object",
                        lambda selected_tract: read_object_tract(
                            args.dc2_root, selected_tract
                        ),
                    )
                catalog = build_lensing_dataframe(
                    args.dc2_root,
                    cosmo_cat,
                    truth_by_tract=truth_by_tract,
                    object_by_tract=object_by_tract,
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
        "output": str(args.output),
        "selected_tracts": sorted(selected_tracts),
        "crop_tiles": args.crop_tiles,
        "interior_tiles": args.interior_tiles,
        "border_tiles": args.border_tiles,
        "tile_slen": args.tile_slen,
        "crop_slen": args.crop_slen,
        "stride": args.stride,
        "complete_context_only": args.complete_context_only,
        "merged_catalog_cache": str(args.merged_catalog_cache)
        if args.merged_catalog_cache is not None
        else None,
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
