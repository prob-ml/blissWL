"""Gather native CosmoGrid pixels into patches with disjoint face-based splits.

Preserve pixel values, dtypes, and IDs in face y/x order.
Save split batches, shared geometry, and metadata.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import h5py
import healpy as hp
import numpy as np
import torch
import yaml
from tqdm import tqdm


SPLITS = ("train", "val", "test")
PARAM_NAMES = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]
STAGE3_DATASETS = tuple(f"kg/stage3_lensing{i}" for i in range(1, 5))
DEFAULT_CONFIG = (
    Path(__file__).with_name("configs") / "generate_cosmogrid_fixed_regions.yaml"
)


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_config(config):
    """Reject invalid or obsolete angular-reprojection settings before I/O."""
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    nside = positive_int(config["input"]["expected_nside"], "input.expected_nside")
    if nside & (nside - 1) or nside > 2**29:
        raise ValueError("expected_nside must be a HEALPix power of two <= 2**29")
    if config["input"]["ordering"] not in ("RING", "NESTED"):
        raise ValueError("input.ordering must be RING or NESTED")
    patch, layout = config["patch"], config["layout"]
    if set(patch) != {"height", "width"}:
        raise ValueError(
            "patch accepts only height/width; native rearrangement has no nsub"
        )
    allowed_layout = {"max_height", "max_width", "dec_range_deg", "seed"}
    if set(layout) != allowed_layout:
        raise ValueError(
            f"layout must contain {sorted(allowed_layout)}; no angular pixel size/gap is used"
        )
    for axis in ("height", "width"):
        size = positive_int(patch[axis], f"patch.{axis}")
        maximum = positive_int(layout[f"max_{axis}"], f"layout.max_{axis}")
        if maximum > nside or nside % maximum:
            raise ValueError(f"layout.max_{axis} must divide NSIDE={nside} exactly")
        if size > maximum or (maximum - size) % 2:
            raise ValueError(
                f"patch.{axis} must fit the maximum and have the same parity to preserve centers"
            )
    seed = layout["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("layout.seed must be a nonnegative integer")
    dec_range = layout["dec_range_deg"]
    if dec_range is not None:
        if not isinstance(dec_range, (list, tuple)) or len(dec_range) != 2:
            raise ValueError("dec_range_deg must be null or [min,max]")
        lo, hi = map(float, dec_range)
        if not np.isfinite([lo, hi]).all() or not -90 <= lo < hi <= 90:
            raise ValueError("DEC cut must satisfy -90 <= min < max <= 90")
    if set(config["regions"]) != set(SPLITS):
        raise ValueError("regions must contain exactly train, val, and test")
    assigned_faces = set()
    for split in SPLITS:
        region = config["regions"][split]
        if set(region) != {"faces", "num_patches"}:
            raise ValueError(
                f"{split} accepts faces and num_patches, not RA/DEC rectangles"
            )
        faces = region["faces"]
        if not isinstance(faces, list):
            raise ValueError(f"{split}.faces must be a list of HEALPix base-face IDs")
        for face in faces:
            if (
                isinstance(face, bool)
                or not isinstance(face, int)
                or not 0 <= face < 12
            ):
                raise ValueError(f"invalid HEALPix base face: {face}")
            if face in assigned_faces:
                raise ValueError(
                    f"face {face} appears more than once; splits must be disjoint"
                )
            assigned_faces.add(face)
        count = region["num_patches"]
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
        ):
            raise ValueError(
                f"{split}.num_patches must be null (all) or a nonnegative integer"
            )
    limit = config["input"].get("max_files")
    if limit is not None:
        positive_int(limit, "input.max_files")


def cell_pixel_ids(nside, face, x_start, y_start, height, width, ordering):
    """Return native IDs in [height,width] face-y/face-x order."""
    y, x = np.indices((height, width), dtype=np.int64)
    return hp.xyf2pix(nside, x + x_start, y + y_start, face, nest=ordering == "NESTED")


def cell_center(nside, face, x_start, y_start, height, width, ordering):
    """Anchor from central pixels, identical for central crops of matching parity.

    A normalized vector mean gives a representative sky direction, not a
    claim that face coordinates are uniform angular coordinates.
    """
    xs = np.unique([(width - 1) // 2, width // 2]) + x_start
    ys = np.unique([(height - 1) // 2, height // 2]) + y_start
    x, y = np.meshgrid(xs, ys)
    ids = hp.xyf2pix(nside, x, y, face, nest=ordering == "NESTED")
    vector = np.asarray(hp.pix2vec(nside, ids.ravel(), nest=ordering == "NESTED")).mean(
        axis=1
    )
    lon, lat = hp.vec2ang(vector, lonlat=True)
    return float(np.asarray(lon).item()), float(np.asarray(lat).item())


def build_catalog(config):
    """Construct native cells once, with count-independent nested selection."""
    validate_config(config)
    layout = config["layout"]
    nside, ordering = config["input"]["expected_nside"], config["input"]["ordering"]
    height, width = layout["max_height"], layout["max_width"]
    nrows, ncols = nside // height, nside // width
    dec_cut = layout["dec_range_deg"]
    patches, selected, capacities = [], [], {}
    for split_id, split in enumerate(SPLITS):
        region = config["regions"][split]
        candidates = []
        for face in sorted(region["faces"]):
            for row in range(nrows):
                for col in range(ncols):
                    x_start, y_start = col * width, row * height
                    if dec_cut is not None:
                        ids = cell_pixel_ids(
                            nside, face, x_start, y_start, height, width, ordering
                        )
                        _, dec = hp.pix2ang(
                            nside, ids, nest=ordering == "NESTED", lonlat=True
                        )
                        if dec.min() < dec_cut[0] or dec.max() > dec_cut[1]:
                            continue
                    lon, lat = cell_center(
                        nside, face, x_start, y_start, height, width, ordering
                    )
                    candidates.append(
                        {
                            "patch_id": face * nrows * ncols + row * ncols + col,
                            "split": split,
                            "split_id": split_id,
                            "face": face,
                            "max_x_start": x_start,
                            "max_y_start": y_start,
                            "center_face_xy": [
                                x_start + (width - 1) / 2,
                                y_start + (height - 1) / 2,
                            ],
                            "center_lon_deg": lon,
                            "center_lat_deg": lat,
                        }
                    )
        capacities[split] = len(candidates)
        requested = region["num_patches"]
        requested = len(candidates) if requested is None else requested
        if requested > len(candidates):
            raise ValueError(
                f"{split}: requested {requested} patches, but only {len(candidates)} native cells are available"
            )
        rng = np.random.default_rng(np.random.SeedSequence([layout["seed"], split_id]))
        ranks = np.empty(len(candidates), dtype=np.int64)
        ranks[rng.permutation(len(candidates))] = np.arange(len(candidates))
        for entry, rank in zip(candidates, ranks, strict=True):
            entry["selection_rank"] = int(rank)
            patches.append(entry)
            if rank < requested:
                selected.append(entry)
    if not selected:
        raise ValueError("the configuration selects no native patches")
    payload = {
        "algorithm": "native_healpix_face_cells_v1",
        "nside": nside,
        "healpix_ordering": ordering,
        "layout": layout,
        "regions": {
            s: {"faces": sorted(config["regions"][s]["faces"])} for s in SPLITS
        },
        "patches": patches,
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **payload,
        "layout_sha256": signature,
        "capacities": capacities,
        "selected_patch_ids": [p["patch_id"] for p in selected],
    }, selected


def build_geometry(config, catalog, patches):
    """Precompute a shared gather index independent of cosmology/file order."""
    nside = config["input"]["expected_nside"]
    height, width = config["patch"]["height"], config["patch"]["width"]
    dx = (config["layout"]["max_width"] - width) // 2
    dy = (config["layout"]["max_height"] - height) // 2
    indices = np.stack(
        [
            cell_pixel_ids(
                nside,
                p["face"],
                p["max_x_start"] + dx,
                p["max_y_start"] + dy,
                height,
                width,
                config["input"]["ordering"],
            )
            for p in patches
        ]
    )
    if np.unique(indices).size != indices.size:
        raise ValueError("native pixels were assigned more than once")
    payload = {
        "layout_sha256": catalog["layout_sha256"],
        "output_shape": [height, width],
        "selected_patch_ids": catalog["selected_patch_ids"],
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {
        "healpix_pixel_ids": torch.from_numpy(indices),
        "patch_ids": torch.tensor([p["patch_id"] for p in patches], dtype=torch.long),
        "split_ids": torch.tensor([p["split_id"] for p in patches], dtype=torch.long),
        "patch_faces": torch.tensor([p["face"] for p in patches], dtype=torch.long),
        "patch_face_xy_origins": torch.tensor(
            [[p["max_x_start"] + dx, p["max_y_start"] + dy] for p in patches],
            dtype=torch.long,
        ),
        "patch_centers_deg": torch.tensor(
            [[p["center_lon_deg"], p["center_lat_deg"]] for p in patches],
            dtype=torch.float64,
        ),
        "nside": nside,
        "healpix_ordering": config["input"]["ordering"],
        "pixel_axes": "array[row,column] = HEALPix face[y,x]",
        "layout_sha256": catalog["layout_sha256"],
        "geometry_sha256": signature,
    }


def extract_native_patches(hp_maps, pixel_ids):
    """Copy [C,Npix] values into [P,H,W,C] without arithmetic or casting."""
    return np.ascontiguousarray(np.moveaxis(hp_maps[:, pixel_ids], 0, -1))


def load_metadata(path):
    with h5py.File(path, "r") as handle:
        rows = handle["parameters/all"][:]
    return {int(r["sobol_index"]): r for r in rows if int(r["sobol_index"]) >= 0}


def params_from_row(row):
    return np.array(
        [row["O_cdm"], row["Ob"], row["s8"], row["H0"] / 100, row["ns"], row["w0"]],
        dtype=np.float32,
    )


def load_native_sky(path, nside, ordering):
    """Read native channels without monopole subtraction or dtype promotion."""
    channels = []
    with h5py.File(path, "r") as handle:
        for key in STAGE3_DATASETS:
            dataset = handle[key]
            if dataset.shape != (hp.nside2npix(nside),):
                raise ValueError(
                    f"{path}:{key}: expected a 1D NSIDE={nside} map, got {dataset.shape}"
                )
            if dataset.dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError(
                    f"{path}:{key}: unsupported native dtype {dataset.dtype}; refusing a lossy cast"
                )
            for attrs in (handle.attrs, handle["kg"].attrs, dataset.attrs):
                declared = attrs.get("ORDERING", attrs.get("ordering"))
                if declared is not None:
                    declared = (
                        declared.decode()
                        if isinstance(declared, bytes)
                        else str(declared)
                    )
                    declared = declared.upper()
                    if declared == "NEST":
                        declared = "NESTED"
                    if declared != ordering:
                        raise ValueError(
                            f"{path}:{key}: source ordering {declared} != configured {ordering}"
                        )
            channel = dataset[:]
            if channels and channel.dtype != channels[0].dtype:
                raise ValueError(
                    "channels have different native dtypes; refusing implicit promotion"
                )
            channels.append(channel)
    maps = np.stack(channels)
    means = maps.mean(axis=1, dtype=np.float64)
    return maps, means


def find_inputs(config, metadata):
    files = []
    for path in sorted(
        (Path(config["input"]["directory"]) / "maps").glob("cosmo_*_perm*.h5")
    ):
        match = re.fullmatch(r"cosmo_(\d+)_perm(\d+)\.h5", path.name)
        if match and int(match[1]) in metadata:
            files.append((path, int(match[1]), int(match[2])))
    if not files:
        raise FileNotFoundError("no full-sky files matched CosmoGrid metadata")
    limit = config["input"].get("max_files")
    return files if limit is None else files[:limit]


def coverage_summary(config, geometry):
    native_count = hp.nside2npix(config["input"]["expected_nside"])
    stored_count = geometry["healpix_pixel_ids"].numel()
    return {
        "full_sky_native_pixels": native_count,
        "selected_native_pixels": stored_count,
        "selected_sky_fraction": stored_count / native_count,
        "full_sky_covered_exactly_once": stored_count == native_count,
    }


def convert(config, catalog, patches, geometry):
    """Read each sky once, reusing the same gather IDs for all channels/skies."""
    input_dir = Path(config["input"]["directory"])
    metadata = load_metadata(input_dir / "CosmoGridV1_metainfo.h5")
    files = find_inputs(config, metadata)
    output_dir = Path(config["output"]["directory"])
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")
    groups = {
        s: np.array(
            [i for i, p in enumerate(patches) if p["split"] == s], dtype=np.int64
        )
        for s in SPLITS
    }
    pixel_ids = geometry["healpix_pixel_ids"].numpy()
    split_ids = {s: pixel_ids[rows] for s, rows in groups.items() if rows.size}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in groups.items():
        if rows.size:
            (output_dir / split).mkdir()
    (output_dir / "patch_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    torch.save(geometry, output_dir / "patch_geometry.pt")
    manifest = {
        "config": config,
        "layout_sha256": catalog["layout_sha256"],
        "geometry_sha256": geometry["geometry_sha256"],
        "split_names": list(SPLITS),
        "param_names": PARAM_NAMES,
        "map_shape": [config["patch"]["height"], config["patch"]["width"], 4],
        "num_input_files": len(files),
        "maps_per_split": {s: len(rows) * len(files) for s, rows in groups.items()},
        "estimator": "native_healpix_pixel_gather",
        "healpix_ordering": config["input"]["ordering"],
        "coordinate_system": "native HEALPix face x/y; sky longitude/latitude assumed RA/DEC",
        "full_sky_monopole_subtracted": False,
        "map_dtype_preserved": True,
        "full_sky_means_saved": True,
        "source_dtypes": [],
        **coverage_summary(config, geometry),
        "status": "incomplete",
        "completed_input_files": 0,
    }
    manifest_path = output_dir / "conversion_metadata.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for index, (path, cosmo_id, perm_id) in enumerate(tqdm(files, desc="Full skies")):
        sky, means = load_native_sky(
            path, config["input"]["expected_nside"], config["input"]["ordering"]
        )
        params = params_from_row(metadata[cosmo_id])
        for split, rows in groups.items():
            if not rows.size:
                continue
            maps = extract_native_patches(sky, split_ids[split])
            n = len(rows)
            batch = {
                "maps": torch.from_numpy(maps),
                "params": torch.from_numpy(np.repeat(params[None, :], n, axis=0)),
                "cosmology_ids": torch.full((n,), cosmo_id, dtype=torch.long),
                "permutation_ids": torch.full((n,), perm_id, dtype=torch.long),
                "geometry_rows": torch.from_numpy(rows),
                "full_sky_means": torch.from_numpy(means),
                "layout_sha256": catalog["layout_sha256"],
                "geometry_sha256": geometry["geometry_sha256"],
                "source_file": path.name,
            }
            for key in (
                "patch_ids",
                "split_ids",
                "patch_faces",
                "patch_face_xy_origins",
                "patch_centers_deg",
            ):
                batch[key] = geometry[key][rows]
            torch.save(batch, output_dir / split / f"batch_{index:06d}.pt")
            del batch, maps
        dtype = str(sky.dtype)
        if dtype not in manifest["source_dtypes"]:
            manifest["source_dtypes"].append(dtype)
        del sky, means
        manifest["completed_input_files"] = index + 1
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved native HEALPix patches to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and show native geometry without reading maps or writing files",
    )
    parser.add_argument("--max-input-files", type=int, help="override input.max_files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override output.directory; relative paths use the working directory",
    )
    args = parser.parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    if args.max_input_files is not None:
        config["input"]["max_files"] = args.max_input_files
    if args.output_dir is not None:
        config["output"]["directory"] = str(args.output_dir)
    catalog, patches = build_catalog(config)
    geometry = build_geometry(config, catalog, patches)
    print(
        json.dumps(
            {
                "layout_sha256": catalog["layout_sha256"],
                "geometry_sha256": geometry["geometry_sha256"],
                "available_patches": catalog["capacities"],
                "selected_patches_per_full_sky": {
                    s: sum(p["split"] == s for p in patches) for s in SPLITS
                },
                "output_patch_shape": [
                    config["patch"]["height"],
                    config["patch"]["width"],
                    4,
                ],
                "native_pixel_resolution_arcmin": float(
                    hp.nside2resol(config["input"]["expected_nside"], arcmin=True)
                ),
                "pixel_axes": geometry["pixel_axes"],
                "full_sky_monopole_subtracted": False,
                **coverage_summary(config, geometry),
            },
            indent=2,
        )
    )
    if not args.dry_run:
        convert(config, catalog, patches, geometry)


if __name__ == "__main__":
    main()
