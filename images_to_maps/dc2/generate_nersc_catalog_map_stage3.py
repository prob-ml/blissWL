"""Generate a Stage-III weighted DC2 convergence map from catalogs.

This script never reads coadd images. It selects DC2 truth galaxies,
queries their CosmoDC2 source-shell redshift and convergence, accumulates
fine source-redshift slices, and combines those slices using four
overlapping Stage-III source n(z) distributions.

The output is a single compressed NPZ archive containing a
``[height, width, 4]`` convergence map plus the source-distribution and
importance-weight diagnostics needed to reproduce it.

For exact release provenance, pass the four official Stage-III n(z)
tables with ``--source-nz-files``. An explicitly selected fallback uses
high-precision Smail parameters inferred from those tables; these
closely reproduce the public tabulations but are not separately
published parameters.

Example
-------
nohup /home/pdailin/blissWL/.venv/bin/python -u \
  images_to_maps/dc2/generate_nersc_catalog_map_stage3.py \
  --center-ra 56.0 \
  --center-dec -36.8 \
  --height 64 \
  --width 40 \
  --pixel-size-arcmin 6.871 \
  --use-inferred-smail-fit \
  --max-missing-target-mass 2e-5 \
  --output /home/pdailin/blissWL/maps_to_cosmology/notebooks/dc2_stage3_nz.npz \
  > /home/pdailin/blissWL/maps_to_cosmology/notebooks/dc2_stage3_nz.log \
  2>&1 &
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from GCRCatalogs import GCRQuery

from images_to_maps.dc2.generate_nersc_catalog_map import (
    DEFAULT_COSMODC2_ROOT,
    DEFAULT_DC2_ROOT,
    SkyMapProjector,
    load_cosmodc2_catalog,
    read_truth_tract,
    select_truth_tracts,
    setup_logging,
)


NUM_STAGE3_BINS = 4

# Full-precision fits inferred from the official nz_stage3_1.txt through
# nz_stage3_4.txt tables. The rounded published values are respectively
# (1.99, 1.44, 0.20), (3.46, 2.34, 0.39),
# (6.03, 3.60, 0.66), and (3.53, 4.49, 1.03).
STAGE3_SMAIL_PARAMETERS = np.asarray(
    [
        [1.98896365, 1.44494416, 0.19581900],
        [3.45988090, 2.33902763, 0.39203522],
        [6.02912468, 3.60057176, 0.66057689],
        [3.53180261, 4.49497939, 1.02932145],
    ],
    dtype=np.float64,
)

DEFAULT_SOURCE_Z_MIN = 0.0
DEFAULT_SOURCE_Z_MAX = 5.0
DEFAULT_SOURCE_Z_POINTS = 1000


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
        help="Output NPZ archive. Existing files require --overwrite.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
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

    source_nz_group = parser.add_mutually_exclusive_group(
        required=True
    )
    source_nz_group.add_argument(
        "--source-nz-files",
        type=Path,
        nargs=NUM_STAGE3_BINS,
        metavar=("NZ1", "NZ2", "NZ3", "NZ4"),
        help=(
            "Four authoritative Stage-III tables containing z and "
            "n(z)."
        ),
    )
    source_nz_group.add_argument(
        "--use-inferred-smail-fit",
        action="store_true",
        help=(
            "Explicitly use embedded high-precision Smail fits "
            "inferred from the official tables instead of the tables."
        ),
    )
    parser.add_argument(
        "--max-missing-target-mass",
        type=float,
        default=1.0e-6,
        help=(
            "Maximum target probability allowed in source-redshift "
            "intervals containing no usable DC2 galaxies."
        ),
    )
    parser.add_argument(
        "--warn-proposal-count",
        type=int,
        default=20,
        help=(
            "Warn when a target-relevant source-redshift interval has "
            "fewer than this many DC2 galaxies."
        ),
    )

    parser.add_argument(
        "--tracts",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional explicit truth tract list. Otherwise select "
            "tracts using parquet RA/Dec bounds."
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


def normalize_nz(
    redshift: np.ndarray,
    nz: np.ndarray,
) -> np.ndarray:
    """Validate and normalize one or more n(z) columns."""

    redshift = np.asarray(
        redshift,
        dtype=np.float64,
    )
    nz = np.asarray(
        nz,
        dtype=np.float64,
    )

    if redshift.ndim != 1 or len(redshift) < 2:
        raise ValueError(
            "The source-redshift grid must be one-dimensional "
            "with at least two points."
        )
    if not np.all(np.isfinite(redshift)):
        raise ValueError(
            "The source-redshift grid contains non-finite values."
        )
    if not np.all(np.diff(redshift) > 0):
        raise ValueError(
            "The source-redshift grid must be strictly increasing."
        )

    if nz.ndim == 1:
        nz = nz[:, None]
    if nz.ndim != 2 or nz.shape[0] != len(redshift):
        raise ValueError(
            "n(z) must have shape [number_of_redshifts, number_of_bins]."
        )
    if not np.all(np.isfinite(nz)):
        raise ValueError(
            "n(z) contains non-finite values."
        )
    if np.any(nz < 0):
        raise ValueError(
            "n(z) contains negative values."
        )

    normalization = np.trapezoid(
        nz,
        redshift,
        axis=0,
    )
    if np.any(~np.isfinite(normalization)) or np.any(
        normalization <= 0
    ):
        raise ValueError(
            "Every n(z) channel must have a positive finite integral."
        )

    return nz / normalization[None, :]


def embedded_stage3_nz() -> tuple[np.ndarray, np.ndarray]:
    """Return the table-equivalent full-precision Smail fits."""

    redshift = np.linspace(
        DEFAULT_SOURCE_Z_MIN,
        DEFAULT_SOURCE_Z_MAX,
        DEFAULT_SOURCE_Z_POINTS,
        dtype=np.float64,
    )

    alpha = STAGE3_SMAIL_PARAMETERS[:, 0]
    beta = STAGE3_SMAIL_PARAMETERS[:, 1]
    z0 = STAGE3_SMAIL_PARAMETERS[:, 2]

    scaled_redshift = (
        redshift[:, None] / z0[None, :]
    )
    nz = (
        redshift[:, None] ** alpha[None, :]
        * np.exp(
            -(scaled_redshift ** beta[None, :])
        )
    )

    return redshift, normalize_nz(redshift, nz)


def load_stage3_nz_files(
    paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Load four Stage-III n(z) tables with a common z grid."""

    redshift_reference = None
    curves = []

    for path in paths:
        table = np.loadtxt(
            path,
            dtype=np.float64,
        )
        if table.ndim != 2 or table.shape[1] < 2:
            raise ValueError(
                f"{path} must contain columns z and n(z)."
            )

        redshift = table[:, 0]
        nz = table[:, 1]

        if redshift_reference is None:
            redshift_reference = redshift
        elif not np.allclose(
            redshift,
            redshift_reference,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "All Stage-III n(z) files must use the same z grid."
            )

        curves.append(nz)

    if redshift_reference is None:
        raise ValueError(
            "No Stage-III n(z) files were supplied."
        )

    curves_array = np.column_stack(curves)
    raw_normalization = np.trapezoid(
        curves_array,
        redshift_reference,
        axis=0,
    )
    if not np.allclose(
        raw_normalization,
        np.ones(NUM_STAGE3_BINS),
        rtol=0.0,
        atol=1.0e-3,
    ):
        raise ValueError(
            "Stage-III source n(z) tables must already integrate "
            f"to unity; measured integrals={raw_normalization}."
        )

    return (
        np.asarray(redshift_reference, dtype=np.float64),
        normalize_nz(
            redshift_reference,
            curves_array,
        ),
    )


def load_target_nz(
    source_nz_files: list[Path] | None,
    use_inferred_smail_fit: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    if source_nz_files is not None:
        redshift, nz = load_stage3_nz_files(
            source_nz_files
        )
        origin = "authoritative_source_nz_tables"
    elif use_inferred_smail_fit:
        redshift, nz = embedded_stage3_nz()
        origin = "embedded_inferred_high_precision_smail_fit"
        logging.warning(
            "Using explicitly requested non-authoritative fallback: %s",
            origin,
        )
    else:
        raise ValueError(
            "Supply --source-nz-files or explicitly request "
            "--use-inferred-smail-fit."
        )

    if nz.shape[1] != NUM_STAGE3_BINS:
        raise ValueError(
            "Stage-III tomography requires exactly four n(z) curves."
        )

    return redshift, nz, origin


def interval_probability_mass(
    redshift: np.ndarray,
    nz: np.ndarray,
) -> np.ndarray:
    """Integrate linearly interpolated n(z) over each z interval."""

    delta_redshift = np.diff(redshift)
    probability_mass = (
        0.5
        * (nz[:-1, :] + nz[1:, :])
        * delta_redshift[:, None]
    )

    normalization = probability_mass.sum(
        axis=0,
        keepdims=True,
    )
    if np.any(normalization <= 0):
        raise ValueError(
            "Every target n(z) must have positive interval mass."
        )

    return probability_mass / normalization


def read_cosmo_lensing(
    cosmo_catalog,
    truth_chunk: pd.DataFrame,
) -> pd.DataFrame:
    """Query source-shell redshift and convergence for DC2 IDs."""

    galaxy_ids = np.unique(
        truth_chunk["cosmodc2_id"].to_numpy(
            dtype=np.int64
        )
    )
    healpix_pixels = np.unique(
        truth_chunk["cosmodc2_hp"].to_numpy(
            dtype=np.int64
        )
    )
    healpix_pixels = healpix_pixels[
        healpix_pixels >= 0
    ]

    columns = [
        "galaxy_id",
        "redshift_true",
        "convergence",
    ]
    if len(galaxy_ids) == 0:
        return pd.DataFrame(columns=columns)

    filters = [
        GCRQuery(
            (
                lambda values, selected=galaxy_ids: np.isin(
                    values,
                    selected,
                ),
                "galaxy_id",
            )
        )
    ]

    native_filters = []
    if len(healpix_pixels) > 0:
        native_filters = [
            GCRQuery(
                (
                    lambda values, selected=healpix_pixels: np.isin(
                        values,
                        selected,
                    ),
                    "healpix_pixel",
                )
            )
        ]

    quantities = cosmo_catalog.get_quantities(
        quantities=columns,
        filters=filters,
        native_filters=native_filters,
    )
    result = pd.DataFrame(quantities)

    if not result.empty:
        result["galaxy_id"] = result[
            "galaxy_id"
        ].astype(np.int64)
        result.drop_duplicates(
            subset=["galaxy_id"],
            inplace=True,
        )

    return result


def accumulate_redshift_slices(
    galaxies: pd.DataFrame,
    convergence_sum_z: np.ndarray,
    source_count_z: np.ndarray,
    source_z_grid: np.ndarray,
) -> tuple[int, int, int, int]:
    """Accumulate unweighted truth kappa into fine source-z slices.

    Returns counts for accumulated, below-grid, above-grid, and
    non-finite sources. All arrays use the same unity base weight.
    """

    if galaxies.empty:
        return 0, 0, 0, 0

    row = galaxies["map_row"].to_numpy(
        dtype=np.int64
    )
    col = galaxies["map_col"].to_numpy(
        dtype=np.int64
    )
    redshift = galaxies[
        "redshift_true"
    ].to_numpy(dtype=np.float64)
    convergence = galaxies[
        "convergence"
    ].to_numpy(dtype=np.float64)

    finite = (
        np.isfinite(redshift)
        & np.isfinite(convergence)
    )
    below_grid = finite & (
        redshift < source_z_grid[0]
    )
    above_grid = finite & (
        redshift > source_z_grid[-1]
    )

    redshift_index = np.searchsorted(
        source_z_grid,
        redshift,
        side="right",
    ) - 1
    redshift_index[
        redshift == source_z_grid[-1]
    ] = len(source_z_grid) - 2

    valid = (
        finite
        & (redshift_index >= 0)
        & (redshift_index < len(source_z_grid) - 1)
    )

    if valid.any():
        indices = (
            row[valid],
            col[valid],
            redshift_index[valid],
        )
        np.add.at(
            convergence_sum_z,
            indices,
            convergence[valid],
        )
        np.add.at(
            source_count_z,
            indices,
            1.0,
        )

    return (
        int(valid.sum()),
        int(below_grid.sum()),
        int(above_grid.sum()),
        int((~finite).sum()),
    )


def combine_stage3_channels(
    convergence_sum_z: np.ndarray,
    source_count_z: np.ndarray,
    target_mass: np.ndarray,
    max_missing_target_mass: float,
    warn_proposal_count: int,
) -> dict[str, np.ndarray]:
    """Combine fine source-z slices into four Stage-III channels."""

    if convergence_sum_z.shape != source_count_z.shape:
        raise ValueError(
            "The source-z convergence and count arrays must match."
        )
    if target_mass.shape != (
        convergence_sum_z.shape[-1],
        NUM_STAGE3_BINS,
    ):
        raise ValueError(
            "Target interval masses do not match the source-z slices."
        )
    if max_missing_target_mass < 0:
        raise ValueError(
            "max_missing_target_mass must be non-negative."
        )

    proposal_counts = source_count_z.sum(
        axis=(0, 1)
    )
    proposal_total = proposal_counts.sum()
    if proposal_total <= 0:
        raise RuntimeError(
            "No usable DC2 sources fall inside the target z grid."
        )

    proposal_mass = proposal_counts / proposal_total
    empty_proposal = proposal_counts == 0
    missing_target_mass = target_mass[
        empty_proposal,
        :,
    ].sum(axis=0)

    if np.any(
        missing_target_mass
        > max_missing_target_mass
    ):
        raise RuntimeError(
            "DC2 lacks source-redshift support: missing target "
            f"probability per channel={missing_target_mass}."
        )
    if np.any(missing_target_mass > 0):
        logging.warning(
            "Truncating target mass below tolerance: %s",
            missing_target_mass,
        )

    target_relevant = np.any(
        target_mass > max_missing_target_mass,
        axis=1,
    )
    low_count = (
        target_relevant
        & (proposal_counts < warn_proposal_count)
    )
    if low_count.any():
        logging.warning(
            "%d target-relevant source-z intervals contain fewer "
            "than %d DC2 galaxies; inspect importance ratios and ESS.",
            int(low_count.sum()),
            warn_proposal_count,
        )

    importance_ratio = np.divide(
        target_mass,
        proposal_mass[:, None],
        out=np.zeros_like(target_mass),
        where=proposal_mass[:, None] > 0,
    )

    convergence_sum = np.einsum(
        "hwk,ki->hwi",
        convergence_sum_z,
        importance_ratio,
        optimize=True,
    )
    convergence_weight = np.einsum(
        "hwk,ki->hwi",
        source_count_z,
        importance_ratio,
        optimize=True,
    )

    # Unity base weights make source_count_z both sum(w_base) and
    # sum(w_base**2). Hence this is the exact sum of final weights^2.
    weight_squared_sum = np.einsum(
        "hwk,ki->hwi",
        source_count_z,
        importance_ratio**2,
        optimize=True,
    )

    convergence_map = np.divide(
        convergence_sum,
        convergence_weight,
        out=np.full_like(
            convergence_sum,
            np.nan,
        ),
        where=convergence_weight > 0,
    )
    effective_sample_size = np.divide(
        convergence_weight**2,
        weight_squared_sum,
        out=np.full_like(
            convergence_weight,
            np.nan,
        ),
        where=weight_squared_sum > 0,
    )

    global_weight = convergence_weight.sum(
        axis=(0, 1)
    )
    global_weight_squared = weight_squared_sum.sum(
        axis=(0, 1)
    )
    global_effective_sample_size = np.divide(
        global_weight**2,
        global_weight_squared,
        out=np.zeros_like(global_weight),
        where=global_weight_squared > 0,
    )

    weighted_mass = (
        proposal_counts[:, None]
        * importance_ratio
    )
    weighted_mass = np.divide(
        weighted_mass,
        weighted_mass.sum(axis=0, keepdims=True),
        out=np.zeros_like(weighted_mass),
        where=(
            weighted_mass.sum(axis=0, keepdims=True)
            > 0
        ),
    )

    supported_target_mass = target_mass.copy()
    supported_target_mass[empty_proposal, :] = 0.0
    supported_target_mass = np.divide(
        supported_target_mass,
        supported_target_mass.sum(
            axis=0,
            keepdims=True,
        ),
        out=np.zeros_like(supported_target_mass),
        where=(
            supported_target_mass.sum(
                axis=0,
                keepdims=True,
            )
            > 0
        ),
    )
    nz_max_error = np.max(
        np.abs(
            weighted_mass - supported_target_mass
        ),
        axis=0,
    )

    return {
        "map": convergence_map,
        "convergence_sum": convergence_sum,
        "weight": convergence_weight,
        "weight_squared_sum": weight_squared_sum,
        "effective_sample_size": effective_sample_size,
        "global_effective_sample_size": (
            global_effective_sample_size
        ),
        "proposal_counts": proposal_counts,
        "proposal_mass": proposal_mass,
        "importance_ratio": importance_ratio,
        "missing_target_mass": missing_target_mass,
        "weighted_interval_mass": weighted_mass,
        "nz_max_error": nz_max_error,
    }


def main():
    args = parse_args()
    setup_logging()

    if args.output.suffix.lower() != ".npz":
        raise ValueError(
            "--output must end in .npz"
        )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. "
            "Use --overwrite to replace it."
        )
    if args.height <= 0 or args.width <= 0:
        raise ValueError(
            "Map height and width must be positive."
        )
    if args.pixel_size_arcmin <= 0:
        raise ValueError(
            "pixel-size-arcmin must be positive."
        )
    if args.warn_proposal_count < 1:
        raise ValueError(
            "warn-proposal-count must be at least one."
        )

    source_z_grid, target_nz, target_nz_origin = load_target_nz(
        args.source_nz_files,
        args.use_inferred_smail_fit,
    )
    target_mass = interval_probability_mass(
        source_z_grid,
        target_nz,
    )
    target_mean_redshift = np.trapezoid(
        source_z_grid[:, None] * target_nz,
        source_z_grid,
        axis=0,
    )

    num_source_slices = len(source_z_grid) - 1
    estimated_slice_bytes = (
        2
        * args.height
        * args.width
        * num_source_slices
        * np.dtype(np.float64).itemsize
    )

    logging.info(
        "Target map: shape=(%d, %d, %d), pixel_size=%.6f arcmin",
        args.height,
        args.width,
        NUM_STAGE3_BINS,
        args.pixel_size_arcmin,
    )
    logging.info(
        "Target n(z) origin: %s",
        target_nz_origin,
    )
    logging.info(
        "Target mean redshifts: %s",
        target_mean_redshift,
    )
    logging.info(
        "Fine source-z grid: %d intervals over [%g, %g]; "
        "slice accumulators use %.1f MiB",
        num_source_slices,
        source_z_grid[0],
        source_z_grid[-1],
        estimated_slice_bytes / (1024**2),
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
            padding_deg=args.tract_padding_deg,
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

    convergence_sum_z = np.zeros(
        (
            args.height,
            args.width,
            num_source_slices,
        ),
        dtype=np.float64,
    )
    source_count_z = np.zeros_like(
        convergence_sum_z
    )

    # Prevent duplicate galaxies from neighboring truth tracts from
    # being accumulated more than once.
    seen_ids: set[int] = set()

    total_truth_rows = 0
    total_inside = 0
    total_matched = 0
    total_accumulated = 0
    total_below_grid = 0
    total_above_grid = 0
    total_nonfinite = 0

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

            # Query one native HEALPix partition at a time to limit
            # peak catalog-query memory.
            for _, truth_chunk in truth.groupby(
                "cosmodc2_hp",
                sort=False,
            ):
                cosmo = read_cosmo_lensing(
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

                (
                    accumulated,
                    below_grid,
                    above_grid,
                    nonfinite,
                ) = accumulate_redshift_slices(
                    galaxies=galaxies,
                    convergence_sum_z=(
                        convergence_sum_z
                    ),
                    source_count_z=source_count_z,
                    source_z_grid=source_z_grid,
                )
                total_accumulated += accumulated
                total_below_grid += below_grid
                total_above_grid += above_grid
                total_nonfinite += nonfinite

                seen_ids.update(
                    galaxies[
                        "cosmodc2_id"
                    ].astype(int)
                )

            logging.info(
                "Tract %d: footprint galaxies=%d, total matched=%d",
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

    combined = combine_stage3_channels(
        convergence_sum_z=convergence_sum_z,
        source_count_z=source_count_z,
        target_mass=target_mass,
        max_missing_target_mass=(
            args.max_missing_target_mass
        ),
        warn_proposal_count=(
            args.warn_proposal_count
        ),
    )

    # The fine-slice arrays are intentionally not written to disk.
    # The saved proposal and importance-ratio arrays are sufficient to
    # reproduce the four-channel combination from a repeated catalog run.
    del convergence_sum_z
    del source_count_z

    source_nz_files = np.asarray(
        []
        if args.source_nz_files is None
        else [
            str(path)
            for path in args.source_nz_files
        ],
        dtype=str,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.savez_compressed(
        args.output,
        map=combined["map"].astype(np.float32),
        convergence_sum=combined[
            "convergence_sum"
        ].astype(np.float32),
        weight=combined["weight"].astype(np.float32),
        weight_squared_sum=combined[
            "weight_squared_sum"
        ].astype(np.float32),
        effective_sample_size=combined[
            "effective_sample_size"
        ].astype(np.float32),
        global_effective_sample_size=combined[
            "global_effective_sample_size"
        ].astype(np.float64),
        source_nz_z=source_z_grid.astype(np.float64),
        source_nz=target_nz.astype(np.float64),
        source_nz_origin=np.asarray(
            target_nz_origin
        ),
        source_nz_files=source_nz_files,
        embedded_smail_parameters=(
            STAGE3_SMAIL_PARAMETERS.copy()
        ),
        target_interval_mass=(
            target_mass.astype(np.float64)
        ),
        target_mean_redshift=(
            target_mean_redshift.astype(np.float64)
        ),
        proposal_counts=combined[
            "proposal_counts"
        ].astype(np.float64),
        proposal_mass=combined[
            "proposal_mass"
        ].astype(np.float64),
        importance_ratio=combined[
            "importance_ratio"
        ].astype(np.float64),
        weighted_interval_mass=combined[
            "weighted_interval_mass"
        ].astype(np.float64),
        missing_target_mass=combined[
            "missing_target_mass"
        ].astype(np.float64),
        nz_max_error=combined[
            "nz_max_error"
        ].astype(np.float64),
        center_ra=np.float64(args.center_ra),
        center_dec=np.float64(args.center_dec),
        pixel_size_arcmin=np.float64(
            args.pixel_size_arcmin
        ),
        selected_tracts=np.asarray(
            selected_tracts,
            dtype=np.int64,
        ),
        tomography_mode=np.asarray(
            "stage3_soft_source_nz"
        ),
        redshift_quantity=np.asarray(
            "redshift_true"
        ),
        base_weight=np.asarray("unity"),
        monopole_subtracted=np.bool_(False),
        total_truth_rows=np.int64(
            total_truth_rows
        ),
        total_inside=np.int64(total_inside),
        total_matched=np.int64(total_matched),
        total_accumulated=np.int64(
            total_accumulated
        ),
        total_below_grid=np.int64(
            total_below_grid
        ),
        total_above_grid=np.int64(
            total_above_grid
        ),
        total_nonfinite=np.int64(
            total_nonfinite
        ),
    )

    spatial_coverage = np.any(
        combined["weight"] > 0,
        axis=-1,
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
        "Accumulated usable galaxies: %d",
        total_accumulated,
    )
    logging.info(
        "Outside z grid: below=%d above=%d; non-finite=%d",
        total_below_grid,
        total_above_grid,
        total_nonfinite,
    )
    logging.info(
        "Spatial coverage: %.4f",
        spatial_coverage.mean(),
    )
    logging.info(
        "Missing target mass: %s",
        combined["missing_target_mass"],
    )
    logging.info(
        "Maximum discrete n(z) errors: %s",
        combined["nz_max_error"],
    )
    logging.info(
        "Global ESS: %s",
        combined[
            "global_effective_sample_size"
        ],
    )

    for bin_index in range(NUM_STAGE3_BINS):
        valid = combined["weight"][
            ...,
            bin_index,
        ] > 0
        logging.info(
            "Stage-III bin %d: mean_z=%.6f coverage=%.4f",
            bin_index + 1,
            target_mean_redshift[bin_index],
            valid.mean(),
        )

    logging.info(
        "Saved archive: %s shape=%s",
        args.output,
        combined["map"].shape,
    )
    logging.info(
        "No monopole subtraction was applied."
    )


if __name__ == "__main__":
    main()
