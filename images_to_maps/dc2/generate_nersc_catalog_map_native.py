"""Bin DC2 catalog convergence onto a native HEALPix face grid.

Use DC2 truth positions with CosmoDC2 properties, or CosmoDC2 directly.
Apply Smail importance weighting or hard redshift bins; save an NPZ archive.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import re

import h5py
import healpy as hp
import numpy as np
import pyarrow.parquet as pq
import yaml


DEFAULT_CONFIG = Path(__file__).with_name(
    "config_generate_nersc_catalog_map_native.yaml"
)
FILE_PATTERN = re.compile(r"z_(\d+)_(\d+)\.step_all\.healpix_(\d+)\.hdf5")
TRUTH_COLUMNS = ["truth_type", "ra", "dec", "redshift", "cosmodc2_id", "cosmodc2_hp"]


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_config(cfg):
    """Fail before reading galaxy arrays or creating output directories."""
    inp, geom, red, out = (cfg[k] for k in ("input", "geometry", "redshift", "output"))
    for section, key in ((geom, "nside"), (inp, "storage_nside")):
        nside = positive_int(section[key], key)
        if nside & (nside - 1):
            raise ValueError(f"{key} must be a HEALPix power of two")
    if geom["nside"] < inp["storage_nside"]:
        raise ValueError("geometry.nside must be >= input.storage_nside")
    if geom["ordering"] != "RING" or inp["storage_ordering"] != "RING":
        raise ValueError(
            "This generator requires RING ordering, as in the local catalogs"
        )
    if inp["source"] not in {"dc2_truth", "cosmodc2"}:
        raise ValueError("input.source must be dc2_truth or cosmodc2")
    if inp["coordinates"] not in (["ra", "dec"], ["ra_true", "dec_true"]):
        raise ValueError("input.coordinates must be [ra, dec] or [ra_true, dec_true]")
    positive_int(inp["batch_size"], "input.batch_size")
    if inp["tracts"] is not None and not isinstance(inp["tracts"], list):
        raise ValueError("input.tracts must be null or a list")
    if isinstance(geom["face"], bool) or not isinstance(geom["face"], int):
        raise ValueError("geometry.face must be an integer")
    if not 0 <= geom["face"] < 12:
        raise ValueError("geometry.face must be in 0..11")
    for axis in ("height", "width"):
        if positive_int(geom[axis], f"geometry.{axis}") > geom["nside"]:
            raise ValueError(f"geometry.{axis} exceeds the face size")
    for section, key in (
        (geom, "require_full_coverage"),
        (red, "use_smail"),
        (out, "overwrite"),
    ):
        if not isinstance(section[key], bool):
            raise ValueError(f"{key} must be a YAML boolean")
    if Path(out["path"]).suffix.lower() != ".npz":
        raise ValueError("output.path must end in .npz")
    if red["use_smail"]:
        positive_int(red["z_points"], "redshift.z_points")
        if red["z_points"] < 2:
            raise ValueError("redshift.z_points must be >= 2")
        if not np.isfinite([red["z_min"], red["z_max"]]).all():
            raise ValueError("Source-z limits must be finite")
        if not 0 <= red["z_min"] < red["z_max"]:
            raise ValueError("Require 0 <= z_min < z_max")
        pars = np.asarray(red["smail_parameters"], dtype=float)
        if pars.ndim != 2 or pars.shape[1] != 3 or not len(pars):
            raise ValueError("smail_parameters must contain [alpha,beta,z0] rows")
        if not np.isfinite(pars).all() or np.any(pars <= 0):
            raise ValueError("Smail parameters must be positive and finite")
        tol = red["max_missing_target_mass"]
        if not np.isfinite(tol) or not 0 <= tol < 1:
            raise ValueError("max_missing_target_mass must be in [0,1)")
        positive_int(red["warn_proposal_count"], "redshift.warn_proposal_count")
    else:
        edges = np.asarray(red["bin_edges"], dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all():
            raise ValueError("bin_edges must be a finite 1D array of length >= 2")
        if np.any(np.diff(edges) <= 0) or edges[0] < 0:
            raise ValueError("bin_edges must be nonnegative and strictly increasing")


def build_geometry(cfg):
    """Use exactly CosmoGrid's array[row,col] = face[y,x] pixel gather."""
    g = cfg["geometry"]
    center = np.asarray(g["center_face_xy"], dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_face_xy must contain two finite face coordinates")
    start = center - (np.array([g["width"], g["height"]]) - 1) / 2
    rounded = np.rint(start).astype(np.int64)
    if not np.allclose(start, rounded, rtol=0, atol=1e-10):
        raise ValueError("Map dimensions/center parity must give integer face origins")
    x0, y0 = rounded
    if (
        x0 < 0
        or y0 < 0
        or x0 + g["width"] > g["nside"]
        or y0 + g["height"] > g["nside"]
    ):
        raise ValueError("Requested patch crosses a HEALPix base-face boundary")
    y, x = np.indices((g["height"], g["width"]))
    ids = hp.xyf2pix(g["nside"], x + x0, y + y0, g["face"], nest=False)
    lon, lat = hp.pix2ang(g["nside"], ids, nest=False, lonlat=True)
    xs = np.unique([(g["width"] - 1) // 2, g["width"] // 2])
    ys = np.unique([(g["height"] - 1) // 2, g["height"] // 2])
    vector = np.asarray(hp.pix2vec(g["nside"], ids[np.ix_(ys, xs)].ravel())).mean(
        axis=1
    )
    center_lon, center_lat = hp.vec2ang(vector, lonlat=True)
    return dict(
        healpix_pixel_ids=ids,
        ra=lon,
        dec=lat,
        face_xy_origin=np.array([x0, y0]),
        center_ra=float(np.asarray(center_lon).item()),
        center_dec=float(np.asarray(center_lat).item()),
        sort_order=np.argsort(ids.ravel()),
    )


def local_pixels(ra, dec, geometry, nside):
    """Return flattened patch indices; -1 means outside or invalid coordinates."""
    ra, dec = np.asarray(ra), np.asarray(dec)
    valid = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90)
    result = np.full(ra.shape, -1, dtype=np.int32)
    pix = hp.ang2pix(nside, np.mod(ra[valid], 360), dec[valid], lonlat=True, nest=False)
    order = geometry["sort_order"]
    sorted_ids = geometry["healpix_pixel_ids"].ravel()[order]
    positions = np.searchsorted(sorted_ids, pix)
    clipped = np.minimum(positions, len(sorted_ids) - 1)
    inside = (positions < len(sorted_ids)) & (sorted_ids[clipped] == pix)
    result[np.flatnonzero(valid)[inside]] = order[positions[inside]]
    return result


def catalog_files(root):
    files = defaultdict(list)
    partitions = defaultdict(set)
    for path in sorted(Path(root).glob("*.hdf5")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match:
            zlo, zhi, pixel = map(int, match.groups())
            files[pixel].append(path)
            partitions[zlo, zhi].add(pixel)
    if not files:
        raise FileNotFoundError(f"No CosmoDC2 partition files in {root}")
    # A partition is complete only if present in every available redshift block.
    return files, set.intersection(*partitions.values())


def footprint(cfg, geometry, available):
    coarse = hp.ang2pix(
        cfg["input"]["storage_nside"],
        geometry["ra"],
        geometry["dec"],
        lonlat=True,
        nest=False,
    )
    return np.isin(coarse, list(available)), np.unique(coarse)


def truth_bounds(geometry, nside):
    """Conservative spherical envelope, including whole native pixels."""
    radius = np.rad2deg(hp.max_pixrad(nside)) + 0.01
    dmin = max(-90.0, float(geometry["dec"].min()) - radius)
    dmax = min(90.0, float(geometry["dec"].max()) + radius)
    pole_distance = 90 - max(abs(dmin), abs(dmax))
    if pole_distance <= radius:
        return [(0.0, 360.0)], (dmin, dmax)
    margin = np.rad2deg(
        np.arcsin(
            min(
                1.0,
                np.sin(np.deg2rad(radius))
                / np.cos(np.deg2rad(max(abs(dmin), abs(dmax)))),
            )
        )
    )
    delta = (geometry["ra"] - geometry["center_ra"] + 180) % 360 - 180
    lo = geometry["center_ra"] + float(delta.min()) - margin
    hi = geometry["center_ra"] + float(delta.max()) + margin
    if hi - lo >= 360:
        intervals = [(0.0, 360.0)]
    elif lo % 360 <= hi % 360:
        intervals = [(lo % 360, hi % 360)]
    else:
        intervals = [(0.0, hi % 360), (lo % 360, 360.0)]
    return intervals, (dmin, dmax)


def select_truth_files(cfg, geometry):
    root = Path(cfg["input"]["dc2_root"]) / "truth_merged_summary_v1-0-0/match_dr6_v2"
    tracts = cfg["input"]["tracts"]
    if tracts is not None:
        paths = [root / f"truth_tract{int(t)}.parquet" for t in sorted(set(tracts))]
        if not paths or not all(p.is_file() for p in paths):
            raise FileNotFoundError("Explicit truth tracts are empty or missing")
        return paths
    intervals, (dmin, dmax) = truth_bounds(geometry, cfg["geometry"]["nside"])
    selected = []
    for path in sorted(root.glob("truth_tract*.parquet")):
        meta = pq.ParquetFile(path).metadata
        indices = {meta.schema.column(i).name: i for i in range(meta.num_columns)}
        for k in range(meta.num_row_groups):
            group = meta.row_group(k)
            ra = group.column(indices["ra"]).statistics
            dec = group.column(indices["dec"]).statistics
            if ra is None or dec is None or not ra.has_min_max or not dec.has_min_max:
                selected.append(path)
                break
            if (
                dec.min <= dmax
                and dec.max >= dmin
                and any(ra.min <= hi and ra.max >= lo for lo, hi in intervals)
            ):
                selected.append(path)
                break
    if not selected:
        raise FileNotFoundError(f"No overlapping truth parquet files in {root}")
    return selected


def truth_membership(cfg, geometry, paths):
    """Keep compact ID/local-pixel arrays, then deduplicate across all tracts.

    IDs and local positions are retained only for galaxies in this patch. Each
    partition is released after its HDF5 files have been processed, avoiding a
    Python set with hundreds of millions of galaxy-ID objects.
    """
    grouped = defaultdict(list)
    for path in paths:
        logging.info("Reading truth %s", path.name)
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=cfg["input"]["batch_size"],
            columns=TRUTH_COLUMNS,
        ):
            df = batch.to_pandas()
            valid = (
                (df["truth_type"] == 1)
                & (df["cosmodc2_id"] >= 0)
                & (df["cosmodc2_hp"] >= 0)
                & np.isfinite(df["redshift"])
            )
            df = df.loc[valid]
            local = local_pixels(
                df["ra"].to_numpy(),
                df["dec"].to_numpy(),
                geometry,
                cfg["geometry"]["nside"],
            )
            df = df.loc[local >= 0]
            local = local[local >= 0]
            partition = df["cosmodc2_hp"].to_numpy(dtype=np.int64)
            ids = df["cosmodc2_id"].to_numpy(dtype=np.int64)
            for pixel in np.unique(partition):
                mask = partition == pixel
                grouped[int(pixel)].append((ids[mask].copy(), local[mask].copy()))
    result = {}
    for pixel in list(grouped):
        chunks = grouped.pop(pixel)
        ids = np.concatenate([v[0] for v in chunks])
        local = np.concatenate([v[1] for v in chunks])
        unique_ids, first = np.unique(ids, return_index=True)
        result[pixel] = (unique_ids, local[first])
    if not result:
        raise RuntimeError("No DC2 truth galaxies fall inside the native patch")
    logging.info(
        "Selected %d unique truth galaxies", sum(len(v[0]) for v in result.values())
    )
    return result


def source_distribution(red):
    if not red["use_smail"]:
        return np.asarray(red["bin_edges"], dtype=float), None, None
    z = np.linspace(red["z_min"], red["z_max"], red["z_points"])
    alpha, beta, z0 = np.asarray(red["smail_parameters"], dtype=float).T
    nz = z[:, None] ** alpha * np.exp(-((z[:, None] / z0) ** beta))
    norm = np.trapezoid(nz, z, axis=0)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise ValueError(
            "Smail curves have zero or nonfinite mass on the chosen z grid"
        )
    nz /= norm
    mass = 0.5 * (nz[:-1] + nz[1:]) * np.diff(z)[:, None]
    mass /= mass.sum(axis=0)
    return z, nz, mass


def accumulate(sums, counts, local, z, kappa, edges):
    finite = np.isfinite(z) & np.isfinite(kappa)
    bins = np.searchsorted(edges, z, side="right") - 1
    bins[z == edges[-1]] = len(edges) - 2
    good = finite & (bins >= 0) & (bins < len(edges) - 1)
    np.add.at(sums, (local[good], bins[good]), kappa[good])
    np.add.at(counts, (local[good], bins[good]), 1.0)
    return np.array(
        [
            len(z),
            good.sum(),
            (finite & (z < edges[0])).sum(),
            (finite & (z > edges[-1])).sum(),
            (~finite).sum(),
        ],
        dtype=np.int64,
    )


def stream_catalogs(cfg, geometry, files, needed, membership, edges):
    size = geometry["healpix_pixel_ids"].size
    sums = np.zeros((size, len(edges) - 1), dtype=np.float64)
    counts = np.zeros_like(sums)
    totals = np.zeros(5, dtype=np.int64)
    source_files = []
    if membership is not None:
        partitions = sorted(membership)
    else:
        # Stored positions can straddle their nominal coarse partition boundary.
        # Include neighboring storage cells, then make an EXACT fine-pixel cut.
        neighbors = hp.get_all_neighbours(
            cfg["input"]["storage_nside"], needed, nest=False
        )
        partitions = sorted(
            set(needed.tolist()) | set(neighbors[neighbors >= 0].tolist())
        )
    for number, pixel in enumerate(partitions, 1):
        logging.info("CosmoDC2 partition %d (%d/%d)", pixel, number, len(partitions))
        if pixel not in files:
            if membership is not None:
                raise FileNotFoundError(
                    f"Truth refers to missing CosmoDC2 partition {pixel}"
                )
            continue
        members = membership.pop(pixel) if membership is not None else None
        used = np.zeros(len(members[0]), dtype=bool) if members is not None else None
        for path in files[pixel]:
            source_files.append(str(path))
            with h5py.File(path, "r") as handle:
                g = handle["galaxyProperties"]
                for start in range(0, len(g["galaxyID"]), cfg["input"]["batch_size"]):
                    sl = slice(start, start + cfg["input"]["batch_size"])
                    if members is not None:
                        ids = g["galaxyID"][sl]
                        positions = np.searchsorted(members[0], ids)
                        clipped = np.minimum(positions, len(members[0]) - 1)
                        keep = (
                            (positions < len(members[0]))
                            & (members[0][clipped] == ids)
                            & ~used[clipped]
                        )
                        rows = np.flatnonzero(keep)
                        # Defend against duplicate IDs within this chunk as well.
                        _, first = np.unique(positions[rows], return_index=True)
                        rows = rows[first]
                        local = members[1][positions[rows]]
                        used[positions[rows]] = True
                    else:
                        ra_field, dec_field = cfg["input"]["coordinates"]
                        local = local_pixels(
                            g[ra_field][sl],
                            g[dec_field][sl],
                            geometry,
                            cfg["geometry"]["nside"],
                        )
                        rows = np.flatnonzero(local >= 0)
                        local = local[rows]
                    if not len(rows):
                        continue
                    z = g[cfg["input"]["redshift_field"]][sl][rows]
                    kappa = g["convergence"][sl][rows]
                    totals += accumulate(sums, counts, local, z, kappa, edges)
        if used is not None and not used.all():
            raise RuntimeError(
                f"{int((~used).sum())} selected truth IDs unmatched in partition {pixel}"
            )
    if counts.sum() == 0:
        raise RuntimeError("No finite sources inside both patch and source-z grid")
    return sums, counts, totals, source_files


def divide_or_nan(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


def combine(sums, counts, mass, red):
    """Same global-proposal importance estimator and diagnostics as Stage-III."""
    proposal_counts = counts.sum(axis=0)
    proposal_mass = proposal_counts / proposal_counts.sum()
    diagnostics = dict(proposal_counts=proposal_counts, proposal_mass=proposal_mass)
    if mass is not None:
        missing = mass[proposal_counts == 0].sum(axis=0)
        if np.any(missing > red["max_missing_target_mass"]):
            raise RuntimeError(
                f"Missing target source-z mass exceeds tolerance: {missing}"
            )
        if np.any(missing > 0):
            logging.warning(
                "Target mass truncated in unsupported intervals: %s", missing
            )
        relevant = np.any(mass > red["max_missing_target_mass"], axis=1)
        low = relevant & (proposal_counts < red["warn_proposal_count"])
        if low.any():
            logging.warning(
                "%d target-relevant source-z intervals have low counts", low.sum()
            )
        ratio = np.divide(
            mass,
            proposal_mass[:, None],
            out=np.zeros_like(mass),
            where=proposal_mass[:, None] > 0,
        )
        total = sums @ ratio
        weight = counts @ ratio
        squared = counts @ (ratio**2)
        weighted_mass = proposal_counts[:, None] * ratio
        norm = weighted_mass.sum(axis=0)
        if np.any(norm <= 0):
            raise RuntimeError("A target channel has no supported source-z mass")
        weighted_mass /= norm
        supported = mass.copy()
        supported[proposal_counts == 0] = 0
        supported /= supported.sum(axis=0)
        diagnostics.update(
            importance_ratio=ratio,
            missing_target_mass=missing,
            weighted_interval_mass=weighted_mass,
            nz_max_error=np.abs(weighted_mass - supported).max(axis=0),
        )
    else:
        total, weight, squared = sums, counts, counts.copy()
    diagnostics.update(
        map=divide_or_nan(total, weight),
        convergence_sum=total,
        weight=weight,
        weight_squared_sum=squared,
        effective_sample_size=divide_or_nan(weight**2, squared),
        global_effective_sample_size=divide_or_nan(
            weight.sum(axis=0) ** 2, squared.sum(axis=0)
        ),
    )
    return diagnostics


def run(cfg, dry_run=False):
    validate_config(cfg)
    geometry = build_geometry(cfg)
    edges, nz, mass = source_distribution(cfg["redshift"])
    files, available = catalog_files(cfg["input"]["cosmodc2_root"])
    spatial_mask, needed = footprint(cfg, geometry, available)
    truth_paths = (
        select_truth_files(cfg, geometry)
        if cfg["input"]["source"] == "dc2_truth"
        else []
    )
    g = cfg["geometry"]
    channels = mass.shape[1] if mass is not None else len(edges) - 1
    summary = dict(
        map_shape=[g["height"], g["width"], channels],
        nside=g["nside"],
        ordering="RING",
        face=g["face"],
        face_xy_origin=geometry["face_xy_origin"].tolist(),
        center_ra=geometry["center_ra"],
        center_dec=geometry["center_dec"],
        pixel_resolution_arcmin=float(hp.nside2resol(g["nside"], arcmin=True)),
        area_deg2=float(
            geometry["healpix_pixel_ids"].size
            * hp.nside2pixarea(g["nside"], degrees=True)
        ),
        catalog_footprint_fraction=float(spatial_mask.mean()),
        fully_covered_storage_footprint=bool(spatial_mask.all()),
        source=cfg["input"]["source"],
        use_smail=cfg["redshift"]["use_smail"],
        selected_truth_tracts=[
            int(p.stem.removeprefix("truth_tract")) for p in truth_paths
        ],
        slice_accumulator_mib=2
        * geometry["healpix_pixel_ids"].size
        * (len(edges) - 1)
        * 8
        / 1024**2,
        output=str(cfg["output"]["path"]),
    )
    logging.info("Plan: %s", json.dumps(summary, indent=2))
    if dry_run:
        return summary
    if g["require_full_coverage"] and not spatial_mask.all():
        raise RuntimeError(
            "Patch crosses catalog footprint; shrink/move it or set require_full_coverage: false"
        )
    output = Path(cfg["output"]["path"])
    if output.exists() and not cfg["output"]["overwrite"]:
        raise FileExistsError(
            f"{output} exists; use a new path or output.overwrite: true"
        )
    if mass is not None:
        logging.warning(
            "Using embedded inferred Smail fits, not authoritative source-nz-files"
        )
    membership = truth_membership(cfg, geometry, truth_paths) if truth_paths else None
    sums, counts, totals, sources = stream_catalogs(
        cfg, geometry, files, needed, membership, edges
    )
    result = combine(sums, counts, mass, cfg["redshift"])
    shape = (g["height"], g["width"], channels)
    for key in (
        "map",
        "convergence_sum",
        "weight",
        "weight_squared_sum",
        "effective_sample_size",
    ):
        result[key] = result[key].reshape(shape).astype(np.float32)
    valid = np.isfinite(result["map"])
    if g["require_full_coverage"] and not valid.all():
        raise RuntimeError(
            f"Some pixel/channels have no usable sources; coverage={valid.mean(axis=(0, 1))}"
        )
    result.update(
        valid_mask=valid,
        footprint_mask=spatial_mask,
        source_count=counts.sum(axis=1).reshape(shape[:2]).astype(np.int64),
        healpix_pixel_ids=geometry["healpix_pixel_ids"],
        pixel_center_ra=geometry["ra"],
        pixel_center_dec=geometry["dec"],
        center_ra=np.float64(geometry["center_ra"]),
        center_dec=np.float64(geometry["center_dec"]),
        nside=np.int64(g["nside"]),
        healpix_ordering=np.asarray("RING"),
        face=np.int64(g["face"]),
        face_xy_origin=geometry["face_xy_origin"],
        pixel_axes=np.asarray("array[row,column] = HEALPix face[y,x]"),
        pixel_size_arcmin=np.float64(hp.nside2resol(g["nside"], arcmin=True)),
        # This is area-equivalent resolution, not constant angular side lengths.
        source_nz_z=edges if mass is not None else np.array([], dtype=float),
        source_nz=nz if nz is not None else np.empty((0, channels)),
        source_bin_edges=edges,
        source_nz_origin=np.asarray(
            "embedded_inferred_high_precision_smail_fit"
            if mass is not None
            else "catalog_hard_source_z_bins"
        ),
        tomography_mode=np.asarray(
            "stage3_soft_source_nz" if mass is not None else "hard_source_z_bins"
        ),
        target_interval_mass=mass if mass is not None else np.empty((0, channels)),
        target_mean_redshift=np.trapezoid(edges[:, None] * nz, edges, axis=0)
        if mass is not None
        else np.array([]),
        source_nz_files=np.array([], dtype=str),
        embedded_smail_parameters=np.asarray(
            cfg["redshift"]["smail_parameters"], dtype=float
        ),
        redshift_quantity=np.asarray(cfg["input"]["redshift_field"]),
        position_quantity=np.asarray(
            "dc2_truth.ra/dec" if truth_paths else "/".join(cfg["input"]["coordinates"])
        ),
        source_files=np.asarray(sources, dtype=str),
        truth_files=np.asarray([str(p) for p in truth_paths], dtype=str),
        matched_accumulated_below_above_nonfinite=totals,
        base_weight=np.asarray("unity"),
        monopole_subtracted=np.bool_(False),
        configuration_json=np.asarray(json.dumps(cfg, sort_keys=True)),
        summary_json=np.asarray(json.dumps(summary, sort_keys=True)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb" if cfg["output"]["overwrite"] else "xb") as handle:
        np.savez_compressed(handle, **result)
    logging.info(
        "Saved %s shape=%s; channel coverage=%s; accumulated=%d",
        output,
        shape,
        valid.mean(axis=(0, 1)),
        totals[1],
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/report only; no galaxy queries or file writes",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    with args.config.open() as handle:
        cfg = yaml.safe_load(handle)
    run(cfg, args.dry_run)


if __name__ == "__main__":
    main()
