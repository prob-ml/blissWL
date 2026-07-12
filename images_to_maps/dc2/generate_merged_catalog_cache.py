"""Precompute merged per-tract DC2 lensing catalogs.

The padded crop generator can reuse these files instead of repeating the
truth/object/CosmoDC2 merge for every crop.

Example:
    python -u images_to_maps/dc2/generate_merged_catalog_cache.py \
        --tracts 3828 \
        --output /nfs/turbo/lsa-regier/dc2_wl_catalog_cache
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from images_to_maps.dc2.generate_nersc_cache import (
    DEFAULT_CONFIG,
    apply_config_defaults,
    build_lensing_dataframe,
    format_duration,
    format_eta,
    iter_patch_manifest,
    load_cached_tract,
    load_cosmodc2_catalog,
    read_object_tract,
    read_truth_tract,
    setup_logging,
)

DEFAULT_MERGED_CATALOG_CACHE = Path("/nfs/turbo/lsa-regier/dc2_wl_catalog_cache")


def parse_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def merged_catalog_path(cache_root: Path, tract: int) -> Path:
    return cache_root / f"tract_{tract}.parquet"


def read_merged_catalog_tract(cache_root: Path, tract: int) -> pd.DataFrame:
    return pd.read_parquet(merged_catalog_path(cache_root, tract))


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


def build_catalog_from_merged_cache(
    cache: OrderedDict[int, pd.DataFrame],
    cache_root: Path,
    tracts: list[int],
    bounds: tuple[float, float, float, float],
    padding: float,
    max_tracts: int,
) -> pd.DataFrame:
    frames = []
    for tract in tracts:
        catalog = load_cached_tract(
            cache,
            tract,
            max_tracts,
            "merged catalog",
            lambda selected_tract: read_merged_catalog_tract(
                cache_root, selected_tract
            ),
        )
        filtered = filter_catalog_to_bounds(catalog, bounds, padding)
        if not filtered.empty:
            frames.append(filtered)
    if not frames:
        return pd.DataFrame()
    catalog = pd.concat(frames, ignore_index=True)
    if "cosmodc2_id" in catalog.columns:
        catalog.drop_duplicates(subset=["cosmodc2_id"], inplace=True)
    elif "galaxy_id" in catalog.columns:
        catalog.drop_duplicates(subset=["galaxy_id"], inplace=True)
    return catalog


def write_merged_catalog(
    dc2_root: Path,
    cosmo_cat,
    output: Path,
    tract: int,
    overwrite: bool,
) -> tuple[Path, int, int]:
    out = merged_catalog_path(output, tract)
    if out.exists() and not overwrite:
        existing = pd.read_parquet(out, columns=["ra"])
        return out, len(existing), 0

    truth = read_truth_tract(dc2_root, tract)
    object_df = read_object_tract(dc2_root, tract)
    catalog = build_lensing_dataframe(
        dc2_root,
        cosmo_cat,
        truth_by_tract={tract: truth},
        object_by_tract={tract: object_df},
    )
    output.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out, index=False)
    return out, len(catalog), len(truth)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dc2-root", type=Path, default=None)
    parser.add_argument("--cosmodc2-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_MERGED_CATALOG_CACHE)
    parser.add_argument("--tracts", type=parse_ints, default=None)
    parser.add_argument("--max-tracts", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser


def normalize_args(args) -> None:
    args.image_slen = None
    args.n_image_split = None
    args.tile_slen = None
    args.avg_ellip_kernel_size = None
    args.avg_ellip_kernel_sigma = None
    args.redshift_quantiles = None
    apply_config_defaults(args)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    normalize_args(args)
    setup_logging(args.log_file)
    run_start = time.monotonic()

    if args.tracts is None:
        tracts = sorted({tract for tract, _patch in iter_patch_manifest(args.dc2_root)})
    else:
        tracts = sorted(set(args.tracts))
    if args.max_tracts is not None:
        tracts = tracts[: args.max_tracts]

    logging.info("Output merged catalog cache: %s", args.output)
    logging.info("Selected tracts: %s", tracts)
    logging.info("Loading CosmoDC2 catalog")
    cosmo_cat = load_cosmodc2_catalog(args.cosmodc2_root)

    written = 0
    skipped = 0
    failures = []
    summaries = []
    total = len(tracts)
    for done, tract in enumerate(tracts, start=1):
        start = time.monotonic()
        out = merged_catalog_path(args.output, tract)
        if out.exists() and not args.overwrite:
            skipped += 1
            logging.info(
                "[%d/%d] skipped existing tract=%s file=%s eta=%s",
                done,
                total,
                tract,
                out,
                format_eta(run_start, done, total),
            )
            continue
        try:
            out, rows, truth_rows = write_merged_catalog(
                args.dc2_root, cosmo_cat, args.output, tract, args.overwrite
            )
            written += 1
            summaries.append(
                {
                    "tract": tract,
                    "file": str(out),
                    "rows": rows,
                    "truth_rows": truth_rows,
                }
            )
            logging.info(
                "[%d/%d] wrote tract=%s rows=%d file=%s elapsed=%s eta=%s",
                done,
                total,
                tract,
                rows,
                out,
                format_duration(time.monotonic() - start),
                format_eta(run_start, done, total),
            )
        except Exception as exc:
            failures.append((tract, repr(exc)))
            logging.exception("Failed tract=%s", tract)
            if not args.skip_errors:
                raise

    manifest = {
        "output": str(args.output),
        "selected_tracts": tracts,
        "written": written,
        "skipped_existing": skipped,
        "failures": failures,
        "examples": summaries[:20],
    }
    manifest_path = (
        args.output / f"merged_catalog_manifest_{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logging.info("Wrote manifest to %s", manifest_path)
    logging.info(
        "Done. written=%d skipped=%d failures=%d elapsed=%s",
        written,
        skipped,
        len(failures),
        format_duration(time.monotonic() - run_start),
    )


if __name__ == "__main__":
    main()
