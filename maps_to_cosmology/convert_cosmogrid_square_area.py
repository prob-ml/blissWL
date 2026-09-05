"""Convert CosmoGrid maps using DC2-shaped square spatial apertures.

The original ``convert_cosmogrid.py`` evaluates each HEALPix map at one
output-pixel center with bilinear interpolation.  This standalone variant
instead estimates the solid-angle average over each square pixel used by the
DC2 ``SkyOffsetFrame`` projector.

Each square is integrated by midpoint quadrature: samples are uniform in
local longitude and in ``sin(local latitude)``, so every sample represents an
equal solid angle.  Values are taken directly from the native RING HEALPix
cells; no extra bilinear interpolation kernel is applied.

As in the original converter, the arithmetic mean of each full-sky HEALPix
channel is subtracted from every extracted patch value.

Example
-------
nohup /home/pdailin/blissWL/.venv/bin/python -u \
  /home/pdailin/blissWL/maps_to_cosmology/convert_cosmogrid_square_area.py \
  --nsub 64 \
  --output-dir /data/scratch/convergence_maps/cosmogrid_square_area_nsub64 \
  > /data/scratch/convergence_maps/cosmogrid_square_area_nsub64.log 2>&1 &
"""

import argparse
import json
from pathlib import Path
import re

import h5py
import healpy as hp
import numpy as np
import torch
from tqdm import tqdm


PARAM_NAMES = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]
STAGE3_DATASETS = tuple(f"kg/stage3_lensing{i}" for i in range(1, 5))

DEFAULT_INPUT_DIR = Path("/data/cosmogrid_v1")
DEFAULT_OUTPUT_DIR = Path(
    "/data/scratch/convergence_maps/"
    "cosmogrid_v1_64x40_4bin_skyoffset_square_area_nsub32_"
    "monopole_subtracted"
)


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    """Parse a strictly positive float for argparse."""
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def load_metadata(metainfo_path: Path):
    """Load CosmoGrid cosmological parameters, keyed by Sobol index."""
    with h5py.File(metainfo_path, "r") as handle:
        rows = handle["parameters/all"][:]

    return {
        int(row["sobol_index"]): row
        for row in rows
        if int(row["sobol_index"]) >= 0
    }


def params_from_row(row):
    """Convert CosmoGrid parameters to this repository's six-value order."""
    return np.array(
        [
            row["O_cdm"],
            row["Ob"],
            row["s8"],
            row["H0"] / 100.0,
            row["ns"],
            row["w0"],
        ],
        dtype=np.float32,
    )


def _validate_geometry(
    height: int,
    width: int,
    pixel_size_deg: float,
    nsub: int,
) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not np.isfinite(pixel_size_deg) or pixel_size_deg <= 0:
        raise ValueError("pixel_size_deg must be positive and finite")
    if nsub <= 0:
        raise ValueError("nsub must be positive")

    pixel_size_rad = np.deg2rad(pixel_size_deg)
    y_edges = (
        np.arange(height + 1, dtype=np.float64) - height / 2
    ) * pixel_size_rad
    if y_edges[0] < -np.pi / 2 or y_edges[-1] > np.pi / 2:
        raise ValueError(
            "the local-latitude aperture extends beyond [-90, 90] degrees"
        )


def _offset_frame_basis(lon_center: float, lat_center: float):
    """Return ICRS basis vectors matching Astropy's SkyOffsetFrame."""
    if not np.isfinite(lon_center) or not np.isfinite(lat_center):
        raise ValueError("patch center must be finite")
    if lat_center < -90.0 or lat_center > 90.0:
        raise ValueError("lat_center must lie within [-90, 90] degrees")

    lon = np.deg2rad(lon_center)
    lat = np.deg2rad(lat_center)

    origin = np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=np.float64,
    )
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float64)
    north = np.array(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        dtype=np.float64,
    )
    return origin, east, north


def _iter_square_sample_hpix(
    nside: int,
    lon_center: float,
    lat_center: float,
    height: int,
    width: int,
    pixel_size_deg: float,
    nsub: int,
):
    """Yield ``[width, nsub, nsub]`` RING indices, one output row at a time."""
    _validate_geometry(height, width, pixel_size_deg, nsub)

    pixel_size_rad = np.deg2rad(pixel_size_deg)
    fractions = (np.arange(nsub, dtype=np.float64) + 0.5) / nsub

    # These are exactly the edges implied by DC2's
    # floor(x / pixel_size + size / 2) assignment.
    x_edges = (
        np.arange(width + 1, dtype=np.float64) - width / 2
    ) * pixel_size_rad
    y_edges = (
        np.arange(height + 1, dtype=np.float64) - height / 2
    ) * pixel_size_rad

    x_samples = (
        x_edges[:-1, None] + fractions[None, :] * pixel_size_rad
    )
    origin, east, north = _offset_frame_basis(lon_center, lat_center)

    for row in range(height):
        # dOmega = d(local longitude) * d(sin(local latitude)).
        # Uniform midpoint samples in sin(y) therefore carry equal weights.
        sin_y_lower = np.sin(y_edges[row])
        sin_y_upper = np.sin(y_edges[row + 1])
        sin_y_samples = (
            sin_y_lower + fractions * (sin_y_upper - sin_y_lower)
        )
        y_samples = np.arcsin(sin_y_samples)

        x = x_samples[:, :, None]
        y = y_samples[None, None, :]
        cos_y = np.cos(y)

        coefficient_origin = cos_y * np.cos(x)
        coefficient_east = cos_y * np.sin(x)
        coefficient_north = np.sin(y)

        vector = (
            coefficient_origin[..., None] * origin
            + coefficient_east[..., None] * east
            + coefficient_north[..., None] * north
        )

        yield hp.vec2pix(
            nside,
            vector[..., 0],
            vector[..., 1],
            vector[..., 2],
            nest=False,
        )


def square_area_average(
    hp_maps: np.ndarray,
    lon_center: float,
    lat_center: float,
    height: int = 64,
    width: int = 40,
    pixel_size_deg: float = 6.871 / 60.0,
    nsub: int = 32,
) -> np.ndarray:
    """Average ``[C, Npix]`` RING maps over DC2-shaped square apertures.

    The returned array is float64 with shape ``[height, width, C]``.  Keeping
    float64 here prevents rounding before full-sky monopole subtraction.
    """
    hp_maps = np.asarray(hp_maps)
    if hp_maps.ndim != 2:
        raise ValueError("hp_maps must have shape [channels, HEALPix pixels]")
    if hp_maps.shape[0] == 0:
        raise ValueError("hp_maps must contain at least one channel")

    try:
        nside = int(hp.npix2nside(hp_maps.shape[1]))
    except ValueError as error:
        raise ValueError(
            f"invalid HEALPix map length: {hp_maps.shape[1]}"
        ) from error

    patch = np.empty(
        (height, width, hp_maps.shape[0]),
        dtype=np.float64,
    )

    sample_rows = _iter_square_sample_hpix(
        nside=nside,
        lon_center=lon_center,
        lat_center=lat_center,
        height=height,
        width=width,
        pixel_size_deg=pixel_size_deg,
        nsub=nsub,
    )
    for row, sample_hpix in enumerate(sample_rows):
        # Shape after indexing: [channels, width, nsub, nsub].
        sampled_values = hp_maps[:, sample_hpix]
        patch[row] = sampled_values.mean(
            axis=(-2, -1),
            dtype=np.float64,
        ).T

    return patch


def extract_patch(
    h5_path: Path,
    lon_center: float,
    lat_center: float,
    height: int = 64,
    width: int = 40,
    pixel_size_deg: float = 6.871 / 60.0,
    nsub: int = 32,
    expected_nside: int = 512,
) -> np.ndarray:
    """Read four Stage-III maps and return one monopole-subtracted patch."""
    channels = []
    full_sky_monopoles = []
    map_nside = None

    with h5py.File(h5_path, "r") as handle:
        for dataset_name in STAGE3_DATASETS:
            hp_map = handle[dataset_name][:]
            if hp_map.ndim != 1:
                raise ValueError(
                    f"{h5_path}:{dataset_name} is not a one-dimensional map"
                )

            try:
                channel_nside = int(hp.npix2nside(hp_map.size))
            except ValueError as error:
                raise ValueError(
                    f"{h5_path}:{dataset_name} has invalid HEALPix length "
                    f"{hp_map.size}"
                ) from error

            if map_nside is None:
                map_nside = channel_nside
            elif channel_nside != map_nside:
                raise ValueError(f"HEALPix NSIDE differs between channels in {h5_path}")

            monopole = np.mean(hp_map, dtype=np.float64)
            if not np.isfinite(monopole):
                raise ValueError(
                    f"{h5_path}:{dataset_name} has a non-finite full-sky mean"
                )
            channels.append(hp_map)
            full_sky_monopoles.append(monopole)

    if map_nside != expected_nside:
        raise ValueError(
            f"expected NSIDE={expected_nside}, found NSIDE={map_nside} in {h5_path}"
        )

    patch = square_area_average(
        hp_maps=np.stack(channels, axis=0),
        lon_center=lon_center,
        lat_center=lat_center,
        height=height,
        width=width,
        pixel_size_deg=pixel_size_deg,
        nsub=nsub,
    )

    # Preserve convert_cosmogrid.py's per-channel full-sky mean subtraction.
    patch -= np.asarray(full_sky_monopoles, dtype=np.float64)[None, None, :]
    return patch.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--height", type=positive_int, default=64)
    parser.add_argument("--width", type=positive_int, default=40)
    parser.add_argument(
        "--pixel-size-arcmin",
        type=positive_float,
        default=6.871,
    )
    parser.add_argument(
        "--nsub",
        type=positive_int,
        default=32,
        help="equal-solid-angle midpoint samples per output-pixel axis",
    )
    parser.add_argument("--expected-nside", type=positive_int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-maps",
        type=positive_int,
        default=None,
        help="optional map limit for a quick run",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace combined_batches.pt if it already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output_dir / "combined_batches.pt"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )

    metainfo_path = args.input_dir / "CosmoGridV1_metainfo.h5"
    if not metainfo_path.is_file():
        raise FileNotFoundError(metainfo_path)

    metadata = load_metadata(metainfo_path)
    map_files = sorted((args.input_dir / "maps").glob("cosmo_*_perm*.h5"))
    if not map_files:
        raise FileNotFoundError(f"no CosmoGrid map files found in {args.input_dir / 'maps'}")

    rng = np.random.default_rng(args.seed)
    maps = []
    params = []
    cosmology_ids = []
    permutation_ids = []

    for h5_path in tqdm(map_files):
        match = re.search(r"cosmo_(\d+)_perm(\d+)\.h5$", h5_path.name)
        if match is None:
            continue

        sobol_index = int(match.group(1))
        perm_index = int(match.group(2))
        if sobol_index not in metadata:
            continue

        lon_center = rng.uniform(0.0, 360.0)
        lat_center = rng.uniform(-60.0, 60.0)

        patch = extract_patch(
            h5_path=h5_path,
            lon_center=lon_center,
            lat_center=lat_center,
            height=args.height,
            width=args.width,
            pixel_size_deg=args.pixel_size_arcmin / 60.0,
            nsub=args.nsub,
            expected_nside=args.expected_nside,
        )

        maps.append(patch)
        params.append(params_from_row(metadata[sobol_index]))
        cosmology_ids.append(sobol_index)
        permutation_ids.append(perm_index)

        if args.num_maps is not None and len(maps) >= args.num_maps:
            break

    if not maps:
        raise RuntimeError("no maps matched valid CosmoGrid metadata")

    maps_tensor = torch.from_numpy(np.stack(maps))
    params_tensor = torch.from_numpy(np.stack(params))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "maps": maps_tensor,
            "params": params_tensor,
            "cosmology_ids": torch.tensor(cosmology_ids, dtype=torch.long),
            "permutation_ids": torch.tensor(permutation_ids, dtype=torch.long),
        },
        output_path,
    )

    (args.output_dir / "param_names.txt").write_text(
        ",".join(PARAM_NAMES) + "\n",
        encoding="utf-8",
    )
    conversion_metadata = {
        "estimator": "equal-solid-angle midpoint square-area average",
        "aperture_coordinates": "SkyOffsetFrame longitude/latitude",
        "healpix_ordering": "RING",
        "height": args.height,
        "width": args.width,
        "pixel_size_arcmin": args.pixel_size_arcmin,
        "nsub_per_axis": args.nsub,
        "samples_per_output_pixel": args.nsub**2,
        "full_sky_monopole_subtracted": True,
        "seed": args.seed,
    }
    (args.output_dir / "conversion_metadata.json").write_text(
        json.dumps(conversion_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Saved maps:   {maps_tensor.shape}")
    print(f"Saved params: {params_tensor.shape}")
    print(f"Estimator:    square-area average ({args.nsub}x{args.nsub})")
    print("Monopole:     full-sky channel mean subtracted")
    print(f"Output:       {args.output_dir}")


if __name__ == "__main__":
    main()
