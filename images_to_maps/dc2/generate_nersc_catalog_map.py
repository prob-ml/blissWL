"""Generate a real DC2 convergence map directly from galaxy catalogs.

This script does not read coadd FITS images. It:

1. Selects truth tracts overlapping a target sky rectangle.
2. Reads galaxy RA, Dec, redshift, and CosmoDC2 IDs.
3. Queries CosmoDC2 convergence values.
4. Projects galaxies onto a fixed angular grid.
5. Accumulates convergence into tomographic bins.
6. Saves a [height, width, redshift_bin] map.

Example
-------
python -u images_to_maps/dc2/generate_nersc_catalog_map.py \
    --center-ra 56.0 \
    --center-dec -36.8 \
    --height 64 \
    --width 40 \
    --pixel-size-arcmin 6.871 \
    --redshift-edges 0.0 0.358 0.631 0.872 2.0 \
    --output /data/scratch/convergence_maps/dc2_real_64x40x4.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import astropy.units as u
import GCRCatalogs
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from GCRCatalogs import GCRQuery

from images_to_maps.dc2.generate_nersc_cache import truth_tract_bounds


DEFAULT_DC2_ROOT = Path(
    "/nfs/turbo/lsa-regier/dc2_nersc"
)

DEFAULT_COSMODC2_ROOT = Path(
    "/nfs/turbo/lsa-regier/lsstdesc-public/"
    "dc2/cosmoDC2_v1.1.4"
)

TRUTH_COLUMNS = [
    "truth_type",
    "ra",
    "dec",
    "redshift",
    "cosmodc2_hp",
    "cosmodc2_id",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--dc2-root",
        type=Path,
        default=DEFAULT_DC2_ROOT,
    )
    parser.add_argument(
        "--cosmodc2-root",
        type=Path,
        default=DEFAULT_COSMODC2_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--center-ra",
        type=float,
        required=True,
        help="Map-center right ascension in degrees.",
    )
    parser.add_argument(
        "--center-dec",
        type=float,
        required=True,
        help="Map-center declination in degrees.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--pixel-size-arcmin",
        type=float,
        default=6.871,
    )

    parser.add_argument(
        "--redshift-edges",
        type=float,
        nargs="+",
        default=[
            0.0,
            0.358,
            0.631,
            0.872,
            2.0,
        ],
        help=(
            "Tomographic bin edges. Four bins require "
            "five edges."
        ),
    )

    parser.add_argument(
        "--tracts",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional explicit truth tract list. "
            "Otherwise select tracts using parquet "
            "RA/Dec bounds."
        ),
    )

    parser.add_argument(
        "--tract-padding-deg",
        type=float,
        default=0.3,
        help=(
            "Padding used during approximate truth-tract "
            "preselection."
        ),
    )

    parser.add_argument(
        "--skip-errors",
        action="store_true",
    )

    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def truth_path(
    dc2_root: Path,
    tract: int,
) -> Path:
    return (
        dc2_root
        / "truth_merged_summary_v1-0-0"
        / "match_dr6_v2"
        / f"truth_tract{tract}.parquet"
    )


def read_truth_tract(
    dc2_root: Path,
    tract: int,
) -> pd.DataFrame:
    """Read only catalog columns needed for the map."""

    truth = pd.read_parquet(
        truth_path(dc2_root, tract),
        columns=TRUTH_COLUMNS,
    )

    truth = truth[
        (truth["truth_type"] == 1)
        & (truth["cosmodc2_id"] >= 0)
        & (truth["cosmodc2_hp"] >= 0)
        & truth["ra"].notna()
        & truth["dec"].notna()
        & truth["redshift"].notna()
    ].copy()

    truth["cosmodc2_id"] = (
        truth["cosmodc2_id"].astype(np.int64)
    )
    truth["cosmodc2_hp"] = (
        truth["cosmodc2_hp"].astype(np.int64)
    )

    truth.drop_duplicates(
        subset=["cosmodc2_id"],
        inplace=True,
    )

    return truth


def load_cosmodc2_catalog(
    cosmodc2_root: Path,
):
    """Load the relocated CosmoDC2 catalog."""

    GCRCatalogs.set_root_dir(
        str(cosmodc2_root.parent)
    )

    return GCRCatalogs.load_catalog(
        "desc_cosmodc2",
        {
            "catalog_root_dir": str(
                cosmodc2_root
            ),
        },
    )


def split_ra_interval(
    lower: float,
    upper: float,
) -> list[tuple[float, float]]:
    """Split a possibly wrapped RA interval at 0/360."""

    if upper - lower >= 360:
        return [(0.0, 360.0)]

    lower = lower % 360.0
    upper = upper % 360.0

    if lower <= upper:
        return [(lower, upper)]

    return [
        (lower, 360.0),
        (0.0, upper),
    ]


def intervals_overlap(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return (
        first[0] <= second[1]
        and first[1] >= second[0]
    )


def select_truth_tracts(
    dc2_root: Path,
    center_ra: float,
    center_dec: float,
    height: int,
    width: int,
    pixel_size_arcmin: float,
    padding_deg: float,
) -> list[int]:
    """Select tracts using parquet RA/Dec metadata.

    This is only a coarse preselection. Exact selection is
    performed later using the tangent-plane map projection.
    """

    bounds = truth_tract_bounds(dc2_root)

    pixel_size_deg = pixel_size_arcmin / 60.0

    half_height = (
        0.5 * height * pixel_size_deg
        + padding_deg
    )
    half_width = (
        0.5 * width * pixel_size_deg
        + padding_deg
    )

    cos_dec = np.cos(
        np.deg2rad(center_dec)
    )
    cos_dec = max(abs(cos_dec), 1.0e-6)

    ra_half_width = half_width / cos_dec

    target_ra_intervals = split_ra_interval(
        center_ra - ra_half_width,
        center_ra + ra_half_width,
    )

    target_dec_min = center_dec - half_height
    target_dec_max = center_dec + half_height

    selected = []

    for tract, tract_bounds in bounds.items():
        (
            tract_ra_min,
            tract_ra_max,
            tract_dec_min,
            tract_dec_max,
        ) = tract_bounds

        dec_overlap = (
            tract_dec_min <= target_dec_max
            and tract_dec_max >= target_dec_min
        )
        if not dec_overlap:
            continue

        tract_ra_intervals = split_ra_interval(
            tract_ra_min,
            tract_ra_max,
        )

        ra_overlap = any(
            intervals_overlap(target_interval, tract_interval)
            for target_interval in target_ra_intervals
            for tract_interval in tract_ra_intervals
        )

        if ra_overlap:
            selected.append(tract)

    return sorted(selected)


class SkyMapProjector:
    """Project ICRS RA/Dec onto a fixed tangent-plane map."""

    def __init__(
        self,
        center_ra: float,
        center_dec: float,
        height: int,
        width: int,
        pixel_size_arcmin: float,
    ):
        self.center_ra = center_ra
        self.center_dec = center_dec
        self.height = height
        self.width = width
        self.pixel_size_arcmin = pixel_size_arcmin
        self.pixel_size_deg = (
            pixel_size_arcmin / 60.0
        )

        center = SkyCoord(
            center_ra * u.deg,
            center_dec * u.deg,
            frame="icrs",
        )
        self.offset_frame = SkyOffsetFrame(
            origin=center
        )

    def project(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
    ):
        sky = SkyCoord(
            np.asarray(ra) * u.deg,
            np.asarray(dec) * u.deg,
            frame="icrs",
        )

        offset = sky.transform_to(
            self.offset_frame
        )

        x_deg = (
            offset.lon
            .wrap_at(180 * u.deg)
            .to_value(u.deg)
        )
        y_deg = offset.lat.to_value(u.deg)

        col = np.floor(
            x_deg / self.pixel_size_deg
            + self.width / 2
        ).astype(np.int64)

        row = np.floor(
            y_deg / self.pixel_size_deg
            + self.height / 2
        ).astype(np.int64)

        inside = (
            np.isfinite(x_deg)
            & np.isfinite(y_deg)
            & (row >= 0)
            & (row < self.height)
            & (col >= 0)
            & (col < self.width)
        )

        return row, col, inside


def read_cosmo_convergence(
    cosmo_catalog,
    truth_chunk: pd.DataFrame,
) -> pd.DataFrame:
    """Query convergence for selected CosmoDC2 IDs."""

    ids = (
        truth_chunk["cosmodc2_id"]
        .to_numpy(dtype=np.int64)
    )
    ids = np.unique(ids)

    hpix = (
        truth_chunk["cosmodc2_hp"]
        .to_numpy(dtype=np.int64)
    )
    hpix = np.unique(hpix)
    hpix = hpix[hpix >= 0]

    if len(ids) == 0:
        return pd.DataFrame(
            columns=[
                "galaxy_id",
                "convergence",
            ]
        )

    filters = [
        GCRQuery(
            (
                lambda values, selected=ids: np.isin(
                    values,
                    selected,
                ),
                "galaxy_id",
            )
        )
    ]

    native_filters = []

    if len(hpix) > 0:
        native_filters = [
            GCRQuery(
                (
                    lambda values, selected=hpix: np.isin(
                        values,
                        selected,
                    ),
                    "healpix_pixel",
                )
            )
        ]

    quantities = cosmo_catalog.get_quantities(
        quantities=[
            "galaxy_id",
            "convergence",
        ],
        filters=filters,
        native_filters=native_filters,
    )

    result = pd.DataFrame(quantities)

    if not result.empty:
        result["galaxy_id"] = (
            result["galaxy_id"].astype(np.int64)
        )
        result.drop_duplicates(
            subset=["galaxy_id"],
            inplace=True,
        )

    return result


def tomographic_weights(
    redshift: np.ndarray,
    redshift_edges: np.ndarray,
) -> np.ndarray:
    """Return hard-bin tomographic weights.

    Output shape:
        [number_of_galaxies, number_of_bins]

    This function can later be replaced by weights derived
    from the CosmoGrid source n(z) distributions.
    """

    redshift = np.asarray(
        redshift,
        dtype=np.float64,
    )

    num_bins = len(redshift_edges) - 1

    bin_index = np.searchsorted(
        redshift_edges,
        redshift,
        side="right",
    ) - 1

    # Include a value exactly equal to the final edge
    # in the final bin.
    bin_index[
        redshift == redshift_edges[-1]
    ] = num_bins - 1

    valid = (
        np.isfinite(redshift)
        & (bin_index >= 0)
        & (bin_index < num_bins)
    )

    weights = np.zeros(
        (len(redshift), num_bins),
        dtype=np.float64,
    )

    rows = np.arange(len(redshift))[valid]

    weights[
        rows,
        bin_index[valid],
    ] = 1.0

    return weights


def accumulate_galaxies(
    galaxies: pd.DataFrame,
    convergence_sum: np.ndarray,
    convergence_weight: np.ndarray,
    redshift_edges: np.ndarray,
):
    """Accumulate one galaxy chunk into the map."""

    if galaxies.empty:
        return 0

    row = galaxies["map_row"].to_numpy(
        dtype=np.int64
    )
    col = galaxies["map_col"].to_numpy(
        dtype=np.int64
    )
    redshift = galaxies["redshift"].to_numpy(
        dtype=np.float64
    )
    convergence = galaxies[
        "convergence"
    ].to_numpy(dtype=np.float64)

    weights = tomographic_weights(
        redshift,
        redshift_edges,
    )

    finite_convergence = np.isfinite(
        convergence
    )

    accumulated = 0

    for bin_index in range(
        convergence_sum.shape[-1]
    ):
        weight = weights[:, bin_index]

        valid = (
            finite_convergence
            & np.isfinite(weight)
            & (weight > 0)
        )

        if not valid.any():
            continue

        np.add.at(
            convergence_sum[..., bin_index],
            (row[valid], col[valid]),
            weight[valid] * convergence[valid],
        )

        np.add.at(
            convergence_weight[..., bin_index],
            (row[valid], col[valid]),
            weight[valid],
        )

        accumulated += int(valid.sum())

    return accumulated


def main():
    args = parse_args()
    setup_logging()

    redshift_edges = np.asarray(
        args.redshift_edges,
        dtype=np.float64,
    )

    if len(redshift_edges) < 2:
        raise ValueError(
            "At least two redshift edges are required."
        )

    if not np.all(
        np.diff(redshift_edges) > 0
    ):
        raise ValueError(
            "redshift_edges must be strictly increasing."
        )

    num_bins = len(redshift_edges) - 1

    logging.info(
        "Target map: shape=(%d, %d, %d), "
        "pixel_size=%.6f arcmin",
        args.height,
        args.width,
        num_bins,
        args.pixel_size_arcmin,
    )

    logging.info(
        "Field size: %.6f x %.6f degrees",
        args.height
        * args.pixel_size_arcmin
        / 60.0,
        args.width
        * args.pixel_size_arcmin
        / 60.0,
    )

    projector = SkyMapProjector(
        center_ra=args.center_ra,
        center_dec=args.center_dec,
        height=args.height,
        width=args.width,
        pixel_size_arcmin=args.pixel_size_arcmin,
    )

    if args.tracts:
        selected_tracts = sorted(
            set(args.tracts)
        )
    else:
        logging.info(
            "Selecting overlapping truth tracts"
        )
        selected_tracts = select_truth_tracts(
            dc2_root=args.dc2_root,
            center_ra=args.center_ra,
            center_dec=args.center_dec,
            height=args.height,
            width=args.width,
            pixel_size_arcmin=(
                args.pixel_size_arcmin
            ),
            padding_deg=(
                args.tract_padding_deg
            ),
        )

    if not selected_tracts:
        raise RuntimeError(
            "No truth tracts overlap the target footprint."
        )

    logging.info(
        "Selected %d truth tracts: %s",
        len(selected_tracts),
        selected_tracts,
    )

    logging.info(
        "Loading CosmoDC2 catalog"
    )
    cosmo_catalog = load_cosmodc2_catalog(
        args.cosmodc2_root
    )

    convergence_sum = np.zeros(
        (
            args.height,
            args.width,
            num_bins,
        ),
        dtype=np.float64,
    )
    convergence_weight = np.zeros_like(
        convergence_sum
    )

    # Prevent duplicate galaxies from neighboring truth
    # tracts from being accumulated more than once.
    seen_ids: set[int] = set()

    total_truth_rows = 0
    total_inside = 0
    total_matched = 0
    total_accumulated = 0

    for tract_index, tract in enumerate(
        selected_tracts,
        start=1,
    ):
        logging.info(
            "[%d/%d] Reading truth tract %d",
            tract_index,
            len(selected_tracts),
            tract,
        )

        try:
            truth = read_truth_tract(
                args.dc2_root,
                tract,
            )

            total_truth_rows += len(truth)

            row, col, inside = projector.project(
                truth["ra"].to_numpy(),
                truth["dec"].to_numpy(),
            )

            truth = truth.loc[inside].copy()
            truth["map_row"] = row[inside]
            truth["map_col"] = col[inside]

            total_inside += len(truth)

            if truth.empty:
                continue

            if seen_ids:
                duplicate = truth[
                    "cosmodc2_id"
                ].isin(seen_ids)

                truth = truth.loc[
                    ~duplicate
                ].copy()

            if truth.empty:
                continue

            # Process one native HEALPix partition at a
            # time, limiting peak memory use.
            for _, truth_chunk in truth.groupby(
                "cosmodc2_hp",
                sort=False,
            ):
                cosmo = read_cosmo_convergence(
                    cosmo_catalog,
                    truth_chunk,
                )

                if cosmo.empty:
                    continue

                galaxies = truth_chunk.merge(
                    cosmo,
                    left_on="cosmodc2_id",
                    right_on="galaxy_id",
                    how="inner",
                    validate="one_to_one",
                )

                total_matched += len(galaxies)

                total_accumulated += (
                    accumulate_galaxies(
                        galaxies=galaxies,
                        convergence_sum=(
                            convergence_sum
                        ),
                        convergence_weight=(
                            convergence_weight
                        ),
                        redshift_edges=(
                            redshift_edges
                        ),
                    )
                )

                seen_ids.update(
                    galaxies[
                        "cosmodc2_id"
                    ].astype(int)
                )

            logging.info(
                "Tract %d: footprint galaxies=%d, "
                "total matched=%d",
                tract,
                len(truth),
                total_matched,
            )

        except Exception:
            logging.exception(
                "Failed to process tract %d",
                tract,
            )

            if not args.skip_errors:
                raise

    dc2_map = np.divide(
        convergence_sum,
        convergence_weight,
        out=np.full_like(
            convergence_sum,
            np.nan,
        ),
        where=convergence_weight > 0,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        args.output,
        map=dc2_map.astype(np.float32),
        convergence_sum=(
            convergence_sum.astype(np.float32)
        ),
        weight=(
            convergence_weight.astype(np.float32)
        ),
        center_ra=np.float64(args.center_ra),
        center_dec=np.float64(args.center_dec),
        pixel_size_arcmin=np.float64(
            args.pixel_size_arcmin
        ),
        redshift_edges=(
            redshift_edges.astype(np.float64)
        ),
        selected_tracts=np.asarray(
            selected_tracts,
            dtype=np.int64,
        ),
    )

    map_npy_path = (
        args.output.parent
        / f"{args.output.stem}_map.npy"
    )
    weight_npy_path = (
        args.output.parent
        / f"{args.output.stem}_weight.npy"
    )

    np.save(
        map_npy_path,
        dc2_map.astype(np.float32),
    )
    np.save(
        weight_npy_path,
        convergence_weight.astype(np.float32),
    )

    spatial_coverage = (
        convergence_weight.sum(axis=-1) > 0
    )

    logging.info(
        "Truth rows read: %d",
        total_truth_rows,
    )
    logging.info(
        "Galaxies inside footprint: %d",
        total_inside,
    )
    logging.info(
        "Matched CosmoDC2 galaxies: %d",
        total_matched,
    )
    logging.info(
        "Accumulated galaxy-bin entries: %d",
        total_accumulated,
    )
    logging.info(
        "Spatial coverage: %.4f",
        spatial_coverage.mean(),
    )

    for bin_index in range(num_bins):
        valid = (
            convergence_weight[..., bin_index]
            > 0
        )

        logging.info(
            "Bin %d [%s, %s]: coverage=%.4f",
            bin_index,
            redshift_edges[bin_index],
            redshift_edges[bin_index + 1],
            valid.mean(),
        )

    logging.info(
        "Saved archive: %s",
        args.output,
    )
    logging.info(
        "Saved map: %s shape=%s",
        map_npy_path,
        dc2_map.shape,
    )
    logging.info(
        "Saved weight map: %s",
        weight_npy_path,
    )


if __name__ == "__main__":
    main()