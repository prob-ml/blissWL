"""Kaiser-Squires convergence maps on a DC2 tract's tile grid.

The competitor for NPE in the mass-mapping comparison.  Shear is averaged in
the same tiles and redshift bins as the NPE targets (``generate_dc2_cache.py``):
tile ``(r, c)`` of a tract covers tract pixels ``[256 r, 256 r + 256)`` in y
and ``[256 c, 256 c + 256)`` in x, row 0 is the south edge, and RA decreases
with column.  The grid can extend past the tract (negative indices or indices
beyond the tract's 108 tiles) so that the Fourier inversion's edge effects fall
outside the region being evaluated; the tract's TAN projection is used
throughout.

Conventions (checked against the object catalog's measured moments):
cosmoDC2's ``shear_1`` and ``shear_2`` are in the image (x, y) = (column, row)
frame, so the inversion uses them as given, with ``k1`` along columns and
``k2`` along rows.  The intrinsic shapes rendered in the DC2 images are
``ellipticity_{1,2}_true_dc2``; in the image frame the rendered intrinsic
ellipticity is ``(-ellipticity_1_true_dc2, +ellipticity_2_true_dc2)``.  These
are ``(1 - q) / (1 + q)``-type ellipticities, whose mean over galaxies
estimates the reduced shear ``g = gamma / (1 - kappa)``.

Shear sources (``--shear-source``):

* ``true``: tile means of the true per-galaxy shear of all galaxies (the
  oracle; an upper bound for noiseless shear).
* ``lensed_ellipticity``: "perfect shape measurement".  Tile means of the
  lensed ellipticity ``(eps + g) / (1 + g* eps)`` of selected detected
  galaxies (``--selection``), with no measurement noise, PSF, or calibration
  error.  Also written: the same with intrinsic shapes randomly rotated
  (``--rotations`` shape-noise realizations), noise-only maps (rotated shapes,
  no shear), and the zero-noise limit (``eps = 0``).
* ``metacal``: DESC's DR6-v2 metacalibration shear estimates
  (``--selection metacal`` only), following Sheldon & Huff (2017) and the
  general (not survey-specific) recipe of the DES Y3 metacal catalogue (Gatti
  et al. 2021): inverse-variance weights from (S/N, T / T_psf) cells, the
  per-bin weighted shear response plus selection response (cuts and weights
  re-evaluated on the sheared measurements), element-wise response
  correction, weighted tile means, and subtraction of the per-bin weighted
  mean shear.  Also written: a calibrated version whose per-bin
  multiplicative and additive biases are fitted (weighted) against the true
  shear of galaxies outside the test region (``--test-tiles``), and noise-only
  maps from randomly rotated measured shapes.

Galaxies are binned by true redshift with the NPE target bins.

Metacal tables have no object id: row ``i`` of ``metacal_tract_T.parquet`` is
row ``i`` of the tract's object catalog, which ``load_metacal`` verifies
(fluxes and colours) before joining.  Metacal shears are in a frame where both
components have the opposite sign to the image frame (checked against the true
shear: slopes -0.33 and -0.34, cross terms consistent with zero), so the image
frame shear is ``METACAL_FRAME_SIGN * g``.  With ``--selection metacal``,
galaxies are those matched to a detected object that passes the metacal cuts,
placed in tiles by the object's measured position; objects with no truth match
(stars, spurious detections) are dropped and counted.
"""

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from astropy.io.fits import Header
from astropy.wcs import WCS

DEFAULT_CACHE = Path("/nfs/turbo/lsa-regier/dc2_wl")
DEFAULT_CATALOGS = Path("/nfs/turbo/lsa-regier/dc2_wl_catalog")
DEFAULT_OBJECTS = Path("/nfs/turbo/lsa-regier/dc2_nersc/object_dpdd_only")
DEFAULT_REDSHIFT_QUANTILES = (0.00, 0.762988, 1.120420, 1.592735)
TILE_SLEN = 256
FIELDS = ("shear_1", "shear_2", "convergence")
SHAPE_COLUMNS = ("ellipticity_1_true_dc2", "ellipticity_2_true_dc2")
CUT_COLUMNS = (
    "objectId",
    "clean",
    "good",
    "extendedness",
    "snr_i_cModel",
    "blendedness",
    "Ixx_pixel_i",
    "Iyy_pixel_i",
    "IxxPSF_pixel_i",
    "IyyPSF_pixel_i",
)
SELECTIONS = ("all_detected", "metacal_like", "metacal")

DEFAULT_METACAL = DEFAULT_CATALOGS / (
    "DC2-prod/Run2.2i/dpdd/Run2.2i-dr6-v2/metacal_table_summary"
)
# tract 3828 test region of the four-tract NPE split (crop rows 0-17, cols 9-26:
# tile rows [0, 72), cols [36, 108))
DEFAULT_TEST_TILES = (0, 72, 36, 108)
METACAL_FRAME_SIGN = -1.0
METACAL_STEP = 0.01  # metacal shear step: R = (g_+ - g_-) / (2 * step)
METACAL_VARIANTS = ("", "_1p", "_1m", "_2p", "_2m")
# General shear-measurement cuts of the DES Y3 metacal catalogue (Gatti et al.
# 2021, Sec. 4.2): 10 < S/N < 1000, T / T_psf > 0.5, T < 10 arcsec^2, and no
# objects with both T > 2 arcsec^2 and S/N < 30.  Survey-specific DES cuts
# (footprint flags, magnitude/colour limits, the |e|-based "binaries" cut) are
# not used.
METACAL_CUTS = {
    "s2n_min": 10.0,
    "s2n_max": 1000.0,
    "size_ratio_min": 0.5,
    "T_max": 10.0,
    "large_faint_T_min": 2.0,
    "large_faint_s2n_max": 30.0,
}
# Inverse-variance weights (Gatti et al. 2021, Sec. 4.3, Eqs. 7-9): constant in
# cells of 20 x 20 log-spaced bins of S/N in [10, 300] and T / T_psf in
# [0.5, 5], values beyond the range going to the edge cells.
WEIGHT_GRID_S2N = np.geomspace(10.0, 300.0, 21)
WEIGHT_GRID_SIZE_RATIO = np.geomspace(0.5, 5.0, 21)
METACAL_COLUMNS = (
    "mcal_flags",
    "mcal_psf_T_mean",
    *(f"mcal_gauss_{q}{v}" for q in ("g1", "g2", "T", "s2n") for v in METACAL_VARIANTS),
)
# alignment check: bright objects' metacal vs cModel i flux and r - i colour
ALIGN_MAX_MAD_DEX = 0.05
ALIGN_MIN_COLOUR_R = 0.9
ALIGN_CHUNK = 25_000


def tract_wcs(cache_dir: Path, tract: int) -> tuple[WCS, tuple[int, int]]:
    """The tract's pixel-grid WCS and its shape in tiles, read from crop (0, 0)."""
    datum = torch.load(
        cache_dir / f"tract_{tract}_crop_00_00_size_1.pt", weights_only=False
    )[0]
    wcs = WCS(Header.fromstring(datum["wcs_header_str"]))
    wcs.wcs.crpix = wcs.wcs.crpix + datum["crop_origin_xy"].numpy().astype(float)
    return wcs, tuple(int(v) for v in datum["tract_shape_tiles"])


def tile_indices(ra, dec, wcs: WCS, tile_slen: int = TILE_SLEN):
    """Tile (row, col) of each position on the tract grid.

    Matches the cache generator: BLISS pixel coordinates are the 0-based
    pixel coordinates plus 0.5, and crops start on tile boundaries, so the
    tract tile is ``floor((pixel + 0.5) / tile_slen)``.
    """
    x, y = wcs.all_world2pix(ra, dec, 0)
    row = np.floor((y + 0.5) / tile_slen).astype(np.int64)
    col = np.floor((x + 0.5) / tile_slen).astype(np.int64)
    return row, col


def redshift_bins(redshift, quantiles) -> np.ndarray:
    """Bin index b such that quantiles[b] < z <= quantiles[b + 1] (z <= 0 -> bin 0).

    Same rule as ``LensingDC2Catalog.bin_catalog_by_redshift``.
    """
    return np.clip(
        np.searchsorted(np.asarray(quantiles), redshift, side="left") - 1, 0, None
    )


def read_window(
    catalog_dir: Path,
    wcs: WCS,
    rows: tuple[int, int],
    cols: tuple[int, int],
    columns,
    tile_slen: int = TILE_SLEN,
) -> tuple[pd.DataFrame, list[int]]:
    """Galaxies of the merged catalogs that fall in a tile window.

    ``rows`` and ``cols`` are half-open tile-index ranges on the tract grid.
    Galaxies appearing in several truth tracts are kept once (by
    ``cosmodc2_id``, first file in sorted order).  The returned frame has the
    requested ``columns`` plus ``row`` and ``col`` relative to the window.
    """
    frames, seen, tracts_used = [], [], []
    columns = list(dict.fromkeys(["cosmodc2_id", *columns]))
    for path in sorted(glob.glob(str(catalog_dir / "tract_*.parquet"))):
        pos = pq.read_table(path, columns=["ra", "dec"]).to_pandas()
        r, c = tile_indices(pos["ra"].to_numpy(), pos["dec"].to_numpy(), wcs, tile_slen)
        inside = (r >= rows[0]) & (r < rows[1]) & (c >= cols[0]) & (c < cols[1])
        if not inside.any():
            continue
        tracts_used.append(int(Path(path).stem.split("_")[1]))
        cat = pq.read_table(path, columns=columns).to_pandas()[inside]
        ids = cat["cosmodc2_id"].to_numpy()
        new = ~np.isin(ids, np.concatenate(seen) if seen else [])
        _, first = np.unique(ids, return_index=True)
        unique_in_file = np.zeros(len(ids), dtype=bool)
        unique_in_file[first] = True
        keep = new & unique_in_file
        seen.append(ids[keep])
        cat = cat[keep].copy()
        cat["row"] = r[inside][keep] - rows[0]
        cat["col"] = c[inside][keep] - cols[0]
        frames.append(cat)
        logging.info(
            "tract %s: %d galaxies in the window", tracts_used[-1], int(keep.sum())
        )
    return pd.concat(frames, ignore_index=True), tracts_used


def tile_mean(values, b, r, c, shape) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``values`` per (bin, row, col), skipping NaNs; NaN where empty."""
    v = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(v)
    total = np.zeros(shape)
    count = np.zeros(shape)
    np.add.at(total, (b[ok], r[ok], c[ok]), v[ok])
    np.add.at(count, (b[ok], r[ok], c[ok]), 1.0)
    with np.errstate(invalid="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean, count


def grid_true_shear(
    catalog_dir: Path,
    wcs: WCS,
    rows: tuple[int, int],
    cols: tuple[int, int],
    redshift_quantiles=DEFAULT_REDSHIFT_QUANTILES,
    tile_slen: int = TILE_SLEN,
    cat: pd.DataFrame | None = None,
) -> dict:
    """Tile-mean true shear and convergence of all galaxies per redshift bin.

    Returns ``{field: (n_bins, n_rows, n_cols)}`` means (NaN where a tile has
    no galaxies in a bin) plus ``count`` and the ``tracts`` read.  NaN values
    are skipped field by field, as in the cache generator.  Pass ``cat`` (from
    ``read_window``) to reuse an already-read window.
    """
    tracts = None
    if cat is None:
        cat, tracts = read_window(
            catalog_dir, wcs, rows, cols, ["redshift", *FIELDS], tile_slen
        )
    shape = (len(redshift_quantiles), rows[1] - rows[0], cols[1] - cols[0])
    b = redshift_bins(cat["redshift"].to_numpy(), redshift_quantiles)
    r, c = cat["row"].to_numpy(), cat["col"].to_numpy()
    out = {f: tile_mean(cat[f].to_numpy(), b, r, c, shape)[0] for f in FIELDS}
    out["count"] = np.zeros(shape)
    np.add.at(out["count"], (b, r, c), 1.0)
    out["tracts"] = tracts
    return out


def intrinsic_ellipticity(cat: pd.DataFrame) -> np.ndarray:
    """Rendered intrinsic ellipticity in the image (column, row) frame, as complex."""
    return -cat[SHAPE_COLUMNS[0]].to_numpy() + 1j * cat[SHAPE_COLUMNS[1]].to_numpy()


def reduced_shear(cat: pd.DataFrame) -> np.ndarray:
    gamma = cat["shear_1"].to_numpy() + 1j * cat["shear_2"].to_numpy()
    return gamma / (1.0 - cat["convergence"].to_numpy())


def lens(eps: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Lensed ellipticity (eps + g) / (1 + g* eps) for |g| <= 1.

    For |g| > 1 (strong lensing) the image is mirrored and the lensed
    ellipticity is (1 + g eps*) / (eps* + g*) (Seitz & Schneider 1997).
    """
    eps, g = np.broadcast_arrays(np.asarray(eps, dtype=complex), g)
    out = (eps + g) / (1.0 + np.conj(g) * eps)
    strong = np.abs(g) > 1.0
    if strong.any():
        e, s = np.conj(eps[strong]), g[strong]
        out[strong] = (1.0 + s * e) / (e + np.conj(s))
    return out


def select_galaxies(cat: pd.DataFrame, object_dir: Path, selection: str) -> np.ndarray:
    """Boolean mask of galaxies in a shear catalog built from detected objects.

    ``all_detected``: galaxies matched to a detected object (one truth galaxy
    per object).  ``metacal_like``: additionally clean, good, extended, i-band
    cModel S/N > 10, resolved (convolved T / T_PSF > 1.5 in i, i.e.
    deconvolved T > T_PSF / 2), and blendedness < 0.42.  Cut columns come from
    the object catalog of the tract the object was detected in.
    """
    # match_objectId is exact int64 (-1 if unmatched); objectId is a lossy float64
    detected = (cat["match_objectId"] >= 0).to_numpy()
    if selection == "all_detected":
        return detected
    if selection != "metacal_like":
        raise ValueError(f"unknown selection {selection!r}")

    tracts = cat.loc[detected, "source_object_tract"].dropna().astype(int).unique()
    objects = []
    for tract in sorted(tracts):
        path = object_dir / f"object_dpdd_tract{tract}.parquet"
        obj = pq.read_table(path, columns=list(CUT_COLUMNS)).to_pandas()
        obj["source_object_tract"] = tract
        objects.append(obj)
    objects = pd.concat(objects, ignore_index=True).drop_duplicates(
        subset=["source_object_tract", "objectId"]
    )
    key = pd.DataFrame(
        {
            "objectId": cat["match_objectId"].to_numpy(),
            "source_object_tract": cat["source_object_tract"].astype("Int64"),
        }
    )
    objects["source_object_tract"] = objects["source_object_tract"].astype("Int64")
    objects["objectId"] = objects["objectId"].astype(np.int64)
    joined = key.merge(objects, on=["objectId", "source_object_tract"], how="left")
    assert len(joined) == len(cat)
    found = joined["snr_i_cModel"].notna().to_numpy() | ~detected
    logging.info(
        "object-catalog rows found for %.4f of detected galaxies",
        found[detected].mean(),
    )
    t_gal = joined["Ixx_pixel_i"] + joined["Iyy_pixel_i"]
    t_psf = joined["IxxPSF_pixel_i"] + joined["IyyPSF_pixel_i"]
    cuts = (
        (joined["clean"] == True)  # noqa: E712  (NaN -> False)
        & (joined["good"] == True)  # noqa: E712
        & (joined["extendedness"] == 1)
        & (joined["snr_i_cModel"] > 10)
        & (t_gal / t_psf > 1.5)
        & (joined["blendedness"] < 0.42)
    )
    return detected & cuts.fillna(False).to_numpy(dtype=bool)


def grid_ellipticity(cat, selected, shape, redshift_quantiles, eps=None, shear=True):
    """Tile means of (lensed) ellipticity of the selected galaxies.

    ``eps`` overrides the intrinsic ellipticities (e.g. rotated, or zero);
    ``shear=False`` gives noise-only maps.  Returns ``(e1, e2, count)``.
    """
    sub = cat[selected]
    if eps is None:
        eps = intrinsic_ellipticity(sub)
    e = lens(eps, reduced_shear(sub)) if shear else eps
    b = redshift_bins(sub["redshift"].to_numpy(), redshift_quantiles)
    r, c = sub["row"].to_numpy(), sub["col"].to_numpy()
    e1, count = tile_mean(e.real, b, r, c, shape)
    e2, _ = tile_mean(e.imag, b, r, c, shape)
    return e1, e2, count


def check_metacal_alignment(mcal: pd.DataFrame, obj: pd.DataFrame, tract: int) -> dict:
    """Raise unless metacal rows are the object catalog's rows in the same order.

    Uses unflagged objects with i-band cModel S/N > 20: the metacal / cModel
    i-flux ratio must be constant up to a small scatter in every chunk of
    ``ALIGN_CHUNK`` rows (it differs by a unit factor), and the two r - i
    colours must correlate.  Misaligned rows give a scatter of ~0.4 dex and no
    colour correlation.
    """
    if len(mcal) != len(obj):
        raise ValueError(f"tract {tract}: {len(mcal)} metacal rows, {len(obj)} objects")
    fm_i, fm_r = (
        mcal["mcal_gauss_flux_i"].to_numpy(),
        mcal["mcal_gauss_flux_r"].to_numpy(),
    )
    fo_i, fo_r = obj["cModelFlux_i"].to_numpy(), obj["cModelFlux_r"].to_numpy()
    with np.errstate(invalid="ignore"):
        bright = (
            (mcal["mcal_flags"].to_numpy() == 0)
            & (obj["snr_i_cModel"].to_numpy() > 20)
            & (fm_i > 0)
            & (fm_r > 0)
            & (fo_i > 0)
            & (fo_r > 0)
        )
    idx = np.flatnonzero(bright)
    if len(idx) < 1000:
        raise ValueError(
            f"tract {tract}: only {len(idx)} bright objects to check alignment"
        )
    log_ratio = np.log10(fm_i[idx] / fo_i[idx])
    offset = np.median(log_ratio)
    mad = np.median(np.abs(log_ratio - offset))
    colour_r = np.corrcoef(
        np.log10(fm_r[idx] / fm_i[idx]), np.log10(fo_r[idx] / fo_i[idx])
    )[0, 1]
    chunk_mads = []
    for start in range(0, len(mcal), ALIGN_CHUNK):
        in_chunk = (idx >= start) & (idx < start + ALIGN_CHUNK)
        if in_chunk.sum() >= 100:
            chunk_mads.append(np.median(np.abs(log_ratio[in_chunk] - offset)))
    worst = max(chunk_mads)
    if (
        mad > ALIGN_MAX_MAD_DEX
        or worst > ALIGN_MAX_MAD_DEX
        or colour_r < ALIGN_MIN_COLOUR_R
    ):
        raise ValueError(
            f"tract {tract}: metacal rows do not match the object catalog "
            f"(flux MAD {mad:.3f} dex, worst chunk {worst:.3f}, colour r {colour_r:.3f})"
        )
    return {
        "mad_dex": mad,
        "worst_chunk_mad_dex": worst,
        "colour_r": colour_r,
        "n_bright": len(idx),
    }


def load_metacal(tract: int, object_dir: Path, metacal_dir: Path) -> pd.DataFrame:
    """One tract's objects (id, measured position) with their metacal columns."""
    mcal = pq.read_table(
        metacal_dir / f"metacal_tract_{tract}.parquet",
        columns=[*METACAL_COLUMNS, "mcal_gauss_flux_i", "mcal_gauss_flux_r"],
    ).to_pandas()
    obj = pq.read_table(
        object_dir / f"object_dpdd_tract{tract}.parquet",
        columns=[
            "objectId",
            "ra",
            "dec",
            "cModelFlux_i",
            "cModelFlux_r",
            "snr_i_cModel",
        ],
    ).to_pandas()
    stats = check_metacal_alignment(mcal, obj, tract)
    logging.info(
        "tract %d: metacal rows aligned with objects (flux MAD %.3f dex, worst chunk %.3f, "
        "colour r %.3f, %d bright objects)",
        tract,
        stats["mad_dex"],
        stats["worst_chunk_mad_dex"],
        stats["colour_r"],
        stats["n_bright"],
    )
    out = pd.concat(
        [
            obj[["objectId", "ra", "dec"]].reset_index(drop=True),
            mcal[list(METACAL_COLUMNS)].reset_index(drop=True),
        ],
        axis=1,
    )
    out["objectId"] = out["objectId"].astype(np.int64)
    out["source_object_tract"] = tract
    return out


def metacal_cuts(df: pd.DataFrame, variant: str = "") -> np.ndarray:
    """Metacal selection, optionally on a sheared variant (``_1p``, ...) for R_S.

    Unflagged, ``s2n_min < S/N < s2n_max``, ``T / T_psf > size_ratio_min``,
    ``T < T_max`` (arcsec^2), and not both ``T > large_faint_T_min`` and
    ``S/N < large_faint_s2n_max``.  The table has one flag column, used for
    every variant.  NaNs fail the cuts.
    """
    s2n = df[f"mcal_gauss_s2n{variant}"].to_numpy()
    size = df[f"mcal_gauss_T{variant}"].to_numpy()
    with np.errstate(invalid="ignore"):
        large_faint = (size > METACAL_CUTS["large_faint_T_min"]) & (
            s2n < METACAL_CUTS["large_faint_s2n_max"]
        )
        return (
            (df["mcal_flags"].to_numpy() == 0)
            & (s2n > METACAL_CUTS["s2n_min"])
            & (s2n < METACAL_CUTS["s2n_max"])
            & (size / df["mcal_psf_T_mean"].to_numpy() > METACAL_CUTS["size_ratio_min"])
            & (size < METACAL_CUTS["T_max"])
            & ~large_faint
        )


def count_cross_tract_duplicates(
    objects: pd.DataFrame, radius_arcsec: float = 0.2
) -> int:
    """Pairs of objects from different tracts closer than ``radius_arcsec``."""
    from scipy.spatial import cKDTree

    ra, dec = (
        np.deg2rad(objects["ra"].to_numpy()),
        np.deg2rad(objects["dec"].to_numpy()),
    )
    xyz = np.column_stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )
    pairs = cKDTree(xyz).query_pairs(
        np.deg2rad(radius_arcsec / 3600), output_type="ndarray"
    )
    tract = objects["source_object_tract"].to_numpy()
    return int((tract[pairs[:, 0]] != tract[pairs[:, 1]]).sum())


def metacal_window(cat, wcs, rows, cols, object_dir, metacal_dir, tile_slen=TILE_SLEN):
    """Matched galaxies with their object's metacal columns, tiled by object position.

    ``cat`` is a ``read_window`` frame.  Each galaxy matched to a detected
    object (``match_objectId >= 0``, one galaxy per object) is joined to that
    object in its detection tract; its ``row``/``col`` become the tile of the
    object's measured position, and galaxies whose object falls outside the
    window are dropped.  Returns ``(frame, info)``, where ``info`` counts the
    objects in the window that pass the metacal cuts but have no truth match
    (they are not used), away from the window edge.
    """
    det = cat[cat["match_objectId"] >= 0].drop(columns=["row", "col"])
    if det["source_object_tract"].isna().any():
        raise ValueError("matched galaxies without a detection tract")
    det = det.assign(
        source_object_tract=det["source_object_tract"].astype(int),
        objectId=det["match_objectId"].astype(np.int64),
    )
    if det.duplicated(["source_object_tract", "objectId"]).any():
        raise ValueError("an object is matched to several galaxies")

    objects = []
    for tract in sorted(det["source_object_tract"].unique()):
        obj = load_metacal(int(tract), object_dir, metacal_dir)
        r, c = tile_indices(obj["ra"].to_numpy(), obj["dec"].to_numpy(), wcs, tile_slen)
        inside = (r >= rows[0]) & (r < rows[1]) & (c >= cols[0]) & (c < cols[1])
        obj = obj[inside].copy()
        obj["row"] = r[inside] - rows[0]
        obj["col"] = c[inside] - cols[0]
        objects.append(obj)
    objects = pd.concat(objects, ignore_index=True)
    duplicates = count_cross_tract_duplicates(objects)

    joined = det.merge(
        objects.drop(columns=["ra", "dec"]),
        on=["source_object_tract", "objectId"],
        how="inner",
        validate="one_to_one",
    )
    keys = objects.merge(
        det[["source_object_tract", "objectId"]],
        on=["source_object_tract", "objectId"],
        how="left",
        indicator=True,
    )
    n_rows, n_cols = rows[1] - rows[0], cols[1] - cols[0]
    interior = (
        (keys["row"] >= 2)
        & (keys["row"] < n_rows - 2)
        & (keys["col"] >= 2)
        & (keys["col"] < n_cols - 2)
    ).to_numpy()
    passes = metacal_cuts(keys) & interior
    unmatched = (keys["_merge"] == "left_only").to_numpy()
    info = {
        "matched_galaxies": len(det),
        "joined_galaxies": len(joined),
        "objects_in_window": len(objects),
        "cross_tract_duplicate_pairs": duplicates,
        "selected_objects_interior": int(passes.sum()),
        "selected_unmatched_fraction": float(unmatched[passes].mean()),
    }
    logging.info("metacal window: %s", info)
    return joined, info


def shear_response_matrix(df: pd.DataFrame) -> np.ndarray:
    """Per-galaxy metacal shear response ``R_ij = (g_i^{jp} - g_i^{jm}) / (2 step)``."""
    r = np.empty((len(df), 2, 2))
    for i in (1, 2):
        for j in (1, 2):
            r[:, i - 1, j - 1] = (
                df[f"mcal_gauss_g{i}_{j}p"].to_numpy()
                - df[f"mcal_gauss_g{i}_{j}m"].to_numpy()
            ) / (2 * METACAL_STEP)
    return r


def weight_cells(df: pd.DataFrame, variant: str = "") -> tuple[np.ndarray, np.ndarray]:
    """(S/N, T / T_psf) weight-grid cell of each galaxy, edge cells for outliers."""
    s2n = df[f"mcal_gauss_s2n{variant}"].to_numpy()
    ratio = df[f"mcal_gauss_T{variant}"].to_numpy() / df["mcal_psf_T_mean"].to_numpy()
    n_s, n_t = len(WEIGHT_GRID_S2N) - 1, len(WEIGHT_GRID_SIZE_RATIO) - 1
    with np.errstate(invalid="ignore"):
        i = np.clip(np.searchsorted(WEIGHT_GRID_S2N, s2n, side="right") - 1, 0, n_s - 1)
        j = np.clip(
            np.searchsorted(WEIGHT_GRID_SIZE_RATIO, ratio, side="right") - 1, 0, n_t - 1
        )
    return i, j


def metacal_weight_table(df: pd.DataFrame, selected: np.ndarray) -> dict:
    """Inverse-variance weight per (S/N, T / T_psf) cell (Gatti et al. 2021, Sec. 4.3).

    ``w = <R_gamma>^2 / sigma_e^2`` (their Eqs. 7-8), with ``<R_gamma>`` the
    cell mean of the mean diagonal shear response and ``sigma_e^2 =
    (<e_1^2> + <e_2^2>) / 2`` the per-galaxy variance of the measured shape in
    the cell.  Their Eq. 9 as printed divides the sums of squares by
    ``n_gal^2``, which is the variance of the cell mean and would make weights
    grow with cell occupancy; we use the per-galaxy variance (divide by
    ``n_gal``), as their Eq. 7 describes.  Computed from the selected galaxies'
    unsheared measurements only (no truth).  Empty cells get weight 0.
    """
    i, j = weight_cells(df)
    r = shear_response_matrix(df)
    r_diag = 0.5 * (r[:, 0, 0] + r[:, 1, 1])
    e_sq = 0.5 * (
        df["mcal_gauss_g1"].to_numpy() ** 2 + df["mcal_gauss_g2"].to_numpy() ** 2
    )
    shape = (len(WEIGHT_GRID_S2N) - 1, len(WEIGHT_GRID_SIZE_RATIO) - 1)
    count, r_sum, e_sq_sum = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    s = selected
    np.add.at(count, (i[s], j[s]), 1.0)
    np.add.at(r_sum, (i[s], j[s]), r_diag[s])
    np.add.at(e_sq_sum, (i[s], j[s]), e_sq[s])
    with np.errstate(invalid="ignore", divide="ignore"):
        r_mean = r_sum / count
        sigma_e_sq = e_sq_sum / count
        weight = np.where(count > 0, r_mean**2 / sigma_e_sq, 0.0)
    return {"weight": weight, "count": count, "R": r_mean, "sigma_e_sq": sigma_e_sq}


def metacal_weights(df: pd.DataFrame, table: dict, variant: str = "") -> np.ndarray:
    """Per-galaxy weight from the cell of its (optionally sheared) S/N and size."""
    i, j = weight_cells(df, variant)
    return table["weight"][i, j]


def metacal_response(df, bins, n_bins, selected, table):
    """Per-bin weighted shear response and selection response, each (n_bins, 2, 2).

    ``R_shear[b]`` is the weighted mean of the per-galaxy response over the
    selected galaxies of bin ``b``.  ``R_select[b, i, j]`` is ``(<g_i>^{S+} -
    <g_i>^{S-}) / (2 step)``: weighted means of the unsheared shape over the
    galaxies selected by the cuts evaluated on the ``jp`` / ``jm`` measurements,
    with the weights also looked up from those measurements (shear-dependent
    weights must be, Gatti et al. 2021, Sec. 4.3).  Metacal frame; flipping
    both components leaves both unchanged.  Binning is by true redshift, so it
    adds no selection response.
    """
    g = [df["mcal_gauss_g1"].to_numpy(), df["mcal_gauss_g2"].to_numpy()]
    r = shear_response_matrix(df)
    w = metacal_weights(df, table)
    r_shear = np.zeros((n_bins, 2, 2))
    r_select = np.zeros((n_bins, 2, 2))
    for b in range(n_bins):
        in_bin = bins == b
        m = selected & in_bin
        r_shear[b] = np.einsum("n,nij->ij", w[m], r[m]) / w[m].sum()
        for j in (1, 2):
            means = []
            for sign in "pm":
                v = f"_{j}{sign}"
                sel = metacal_cuts(df, v) & in_bin
                wv = metacal_weights(df, table, v)[sel]
                means.append([np.sum(wv * gi[sel]) / wv.sum() for gi in g])
            for i in (1, 2):
                r_select[b, i - 1, j - 1] = (means[0][i - 1] - means[1][i - 1]) / (
                    2 * METACAL_STEP
                )
    return r_shear, r_select


def metacal_shear(g: np.ndarray, bins: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Response-corrected per-galaxy shear in the image frame, as complex.

    ``g`` is the complex metacal shape ``g1 + i g2`` (metacal frame).  As in
    Sheldon & Huff (2017) and DES Y1/Y3, the ensemble response is taken as
    diagonal and the correction is element-wise: ``METACAL_FRAME_SIGN * g_i /
    R_b,ii`` with ``R_b`` the galaxy's bin response.
    """
    diag = np.stack([response[:, 0, 0], response[:, 1, 1]], axis=1)[bins]
    return METACAL_FRAME_SIGN * (g.real / diag[:, 0] + 1j * g.imag / diag[:, 1])


def weighted_mean_shear(e, w, bins, n_bins) -> np.ndarray:
    """Per-bin weighted mean of complex per-galaxy shear, as (n_bins, 2)."""
    out = np.zeros((n_bins, 2))
    for b in range(n_bins):
        m = bins == b
        out[b] = [
            np.sum(w[m] * e[m].real) / w[m].sum(),
            np.sum(w[m] * e[m].imag) / w[m].sum(),
        ]
    return out


def fit_shear_calibration(e, g_true, w, bins, n_bins, use):
    """Per-bin weighted fit ``e_i = c_i + A_i1 g1 + A_i2 g2`` on galaxies in ``use``.

    Weighted least squares with the inverse-variance weights ``w``.  The cell
    weights only approximate each galaxy's inverse variance, so standard
    errors use the heteroskedasticity-robust sandwich covariance
    ``(X^T W X)^{-1} (sum_n w_n^2 r_n^2 x_n x_n^T) (X^T W X)^{-1}``.
    Returns ``A`` (n_bins, 2, 2), its standard errors, ``c``
    (n_bins, 2), its standard errors, and the galaxy counts.  The
    multiplicative bias of component ``i`` is ``A_ii - 1``.
    """
    a = np.zeros((n_bins, 2, 2))
    a_err = np.zeros((n_bins, 2, 2))
    c = np.zeros((n_bins, 2))
    c_err = np.zeros((n_bins, 2))
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = use & (bins == b)
        counts[b] = m.sum()
        x = np.column_stack([np.ones(m.sum()), g_true[m].real, g_true[m].imag])
        wb = w[m]
        xtwx_inv = np.linalg.inv(x.T @ (wb[:, None] * x))
        for i, y in enumerate((e[m].real, e[m].imag)):
            coef = xtwx_inv @ (x.T @ (wb * y))
            u = (wb * (y - x @ coef))[:, None] * x  # per-galaxy score contributions
            cov = xtwx_inv @ (u.T @ u) @ xtwx_inv
            se = np.sqrt(np.diag(cov))
            c[b, i], a[b, i] = coef[0], coef[1:]
            c_err[b, i], a_err[b, i] = se[0], se[1:]
    return a, a_err, c, c_err, counts


def kaiser_squires(gamma1: np.ndarray, gamma2: np.ndarray, pad_factor: int = 2):
    """Flat-sky Kaiser-Squires inversion of gridded shear.

    ``gamma1`` and ``gamma2`` are ``(..., n_rows, n_cols)`` arrays in the
    (column, row) frame; NaNs are treated as zero shear.  The maps are
    zero-padded by ``pad_factor`` before the FFT to suppress wrap-around.
    The k = 0 mode is set to zero, so the returned E-mode map has zero mean
    over the padded field.  As in SMPy, modes at the Nyquist frequency of one
    axis that also vary along the other axis are dropped, since the sign of
    the ``k1 k2`` term is undefined there; with ``pad_factor=1`` the result
    equals SMPy's ``convergence_from_shear``.  Returns ``(kappa_E, kappa_B)``.
    """
    g = np.nan_to_num(gamma1) + 1j * np.nan_to_num(gamma2)
    n_rows, n_cols = g.shape[-2:]
    pr, pc = pad_factor * n_rows, pad_factor * n_cols
    padded = np.zeros(g.shape[:-2] + (pr, pc), dtype=complex)
    padded[..., :n_rows, :n_cols] = g
    k2, k1 = np.meshgrid(np.fft.fftfreq(pr), np.fft.fftfreq(pc), indexing="ij")
    ksq = k1**2 + k2**2
    ksq[0, 0] = 1.0
    # kappa_hat = conj(D) gamma_hat / |k|^2 with D = (k1^2 - k2^2 + 2i k1 k2)
    d_conj = (k1**2 - k2**2 - 2j * k1 * k2) / ksq
    d_conj[0, 0] = 0.0
    d_conj[((k1 == -0.5) & (k2 != 0)) | ((k2 == -0.5) & (k1 != 0))] = 0.0
    kappa = np.fft.ifft2(d_conj * np.fft.fft2(padded))[..., :n_rows, :n_cols]
    return kappa.real, kappa.imag


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tract", type=int, default=3828)
    parser.add_argument(
        "--pad-tiles",
        type=int,
        default=40,
        help="tiles added on every side of the tract grid",
    )
    parser.add_argument("--fft-pad-factor", type=int, default=2)
    parser.add_argument(
        "--shear-source",
        choices=("true", "lensed_ellipticity", "metacal"),
        default="true",
    )
    parser.add_argument("--selection", choices=SELECTIONS, default="metacal_like")
    parser.add_argument(
        "--rotations",
        type=int,
        default=20,
        help="shape-noise realizations (random rotations of intrinsic shapes)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOGS)
    parser.add_argument("--object-dir", type=Path, default=DEFAULT_OBJECTS)
    parser.add_argument("--metacal-dir", type=Path, default=DEFAULT_METACAL)
    parser.add_argument(
        "--test-tiles",
        type=int,
        nargs=4,
        default=DEFAULT_TEST_TILES,
        metavar=("ROW0", "ROW1", "COL0", "COL1"),
        help="half-open tract-tile ranges of the test region; the metacal "
        "calibration uses only galaxies outside it",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shear_source == "metacal" and args.selection != "metacal":
        parser.error("--shear-source metacal requires --selection metacal")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    wcs, (n_rows, n_cols) = tract_wcs(args.cache_dir, args.tract)
    rows = (-args.pad_tiles, n_rows + args.pad_tiles)
    cols = (-args.pad_tiles, n_cols + args.pad_tiles)
    quantiles = DEFAULT_REDSHIFT_QUANTILES
    shape = (len(quantiles), rows[1] - rows[0], cols[1] - cols[0])
    meta = dict(
        row0=rows[0],
        col0=cols[0],
        tract=args.tract,
        pad_tiles=args.pad_tiles,
        fft_pad_factor=args.fft_pad_factor,
        shear_source=args.shear_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.shear_source == "true":
        truth = grid_true_shear(args.catalog_dir, wcs, rows, cols, quantiles)
        kappa_e, kappa_b = kaiser_squires(
            truth["shear_1"], truth["shear_2"], args.fft_pad_factor
        )
        np.savez(
            args.output,
            kappa_ks=kappa_e,
            kappa_ks_b=kappa_b,
            **{f"{f}_true": truth[f] for f in FIELDS},
            count=truth["count"],
            tracts_read=np.array(truth["tracts"]),
            **meta,
        )
        logging.info("wrote %s", args.output)
        return

    columns = [
        "redshift",
        *FIELDS,
        *SHAPE_COLUMNS,
        "match_objectId",
        "source_object_tract",
    ]
    cat, tracts = read_window(args.catalog_dir, wcs, rows, cols, columns)
    truth = grid_true_shear(args.catalog_dir, wcs, rows, cols, quantiles, cat=cat)
    extra = {}
    if args.selection == "metacal":
        # matched galaxies only, tiled by their object's measured position
        cat, info = metacal_window(
            cat, wcs, rows, cols, args.object_dir, args.metacal_dir
        )
        selected = metacal_cuts(cat)
        extra = {
            **{f"window_{k}": v for k, v in info.items()},
            **{f"cut_{k}": v for k, v in METACAL_CUTS.items()},
            "metacal_dir": str(args.metacal_dir),
        }
    else:
        selected = select_galaxies(cat, args.object_dir, args.selection)
    if args.shear_source == "metacal":
        write_metacal(
            args, cat, selected, truth, tracts, shape, rows, cols, meta, extra
        )
        return
    sub = cat[selected]
    g = reduced_shear(sub)
    logging.info(
        "selection %s: %d of %d galaxies; %d with |g| > 1",
        args.selection,
        int(selected.sum()),
        len(cat),
        int((np.abs(g) > 1).sum()),
    )

    def realization(eps=None, shear=True):
        e1, e2, count = grid_ellipticity(cat, selected, shape, quantiles, eps, shear)
        k_e, k_b = kaiser_squires(e1, e2, args.fft_pad_factor)
        return e1, e2, k_e, k_b, count

    eps_actual = intrinsic_ellipticity(sub)
    actual = realization()
    zero_noise = realization(eps=np.zeros_like(eps_actual))
    rng = np.random.default_rng(args.seed)
    rotated, noise_only = [], []
    for k in range(args.rotations):
        phase = np.exp(2j * rng.uniform(0.0, np.pi, size=len(eps_actual)))
        rotated.append(realization(eps=eps_actual * phase))
        noise_only.append(realization(eps=eps_actual * phase, shear=False))
        logging.info("rotation %d of %d done", k + 1, args.rotations)

    def stack(results, i):
        return np.stack([res[i] for res in results])

    sigma_e = np.array(
        [
            np.std(eps_actual.real[redshift_bins(sub["redshift"], quantiles) == b])
            for b in range(len(quantiles))
        ]
    )
    np.savez(
        args.output,
        e1=actual[0],
        e2=actual[1],
        kappa_ks=actual[2],
        kappa_ks_b=actual[3],
        count=actual[4],
        e1_zero_noise=zero_noise[0],
        e2_zero_noise=zero_noise[1],
        kappa_ks_zero_noise=zero_noise[2],
        e1_rot=stack(rotated, 0),
        e2_rot=stack(rotated, 1),
        kappa_ks_rot=stack(rotated, 2),
        kappa_ks_b_rot=stack(rotated, 3),
        e1_noise=stack(noise_only, 0),
        e2_noise=stack(noise_only, 1),
        kappa_ks_noise=stack(noise_only, 2),
        **{f"{f}_true": truth[f] for f in FIELDS},
        count_all=truth["count"],
        sigma_e=sigma_e,
        selection=args.selection,
        rotations=args.rotations,
        seed=args.seed,
        tracts_read=np.array(tracts),
        **meta,
        **extra,
    )
    logging.info("wrote %s", args.output)


def tile_weighted_mean(values, weights, b, r, c, shape):
    """Weighted mean of ``values`` per (bin, row, col); NaN where the weight sum is 0.

    Returns ``(mean, weight_sum, count)``.
    """
    total, wsum, count = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    np.add.at(total, (b, r, c), weights * values)
    np.add.at(wsum, (b, r, c), weights)
    np.add.at(count, (b, r, c), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(wsum > 0, total / wsum, np.nan)
    return mean, wsum, count


def write_metacal(args, cat, selected, truth, tracts, shape, rows, cols, meta, extra):
    """Metacal KS maps following the metacal literature, plus noise-only maps.

    Per galaxy: inverse-variance weight ``w`` from its (S/N, T / T_psf) cell;
    shear ``METACAL_FRAME_SIGN * g_i / R_b,ii`` with ``R_b`` the bin's weighted
    shear-plus-selection response.  Per tile: the weighted mean.  Main version
    (``kappa_ks``): the per-bin weighted mean shear of all selected galaxies is
    subtracted first (Gatti et al. 2021, Sec. 4.4).  Calibrated version
    (``kappa_ks_cal``): ``(e - c) / (1 + m)`` with ``m`` and ``c`` from a
    weighted fit against the true reduced shear of galaxies outside the test
    region.  Noise-only maps rotate each measured shape by a random angle.
    """
    quantiles = DEFAULT_REDSHIFT_QUANTILES
    n_bins = len(quantiles)
    bins_all = redshift_bins(cat["redshift"].to_numpy(), quantiles)
    table = metacal_weight_table(cat, selected)
    logging.info(
        "weight table: %d of %d cells populated; weights %.3g to %.3g",
        int((table["count"] > 0).sum()),
        table["count"].size,
        table["weight"][table["count"] > 0].min(),
        table["weight"].max(),
    )
    r_shear, r_select = metacal_response(cat, bins_all, n_bins, selected, table)
    response = r_shear + r_select
    logging.info("weighted metacal response per bin (shear + selection):\n%s", response)

    sub = cat[selected]
    bins = bins_all[selected]
    tile_r, tile_c = sub["row"].to_numpy(), sub["col"].to_numpy()
    g_meas = sub["mcal_gauss_g1"].to_numpy() + 1j * sub["mcal_gauss_g2"].to_numpy()
    w = metacal_weights(sub, table)
    if not (np.isfinite(g_meas).all() and np.isfinite(w).all() and (w > 0).all()):
        raise ValueError(
            "selected galaxies with non-finite shapes or non-positive weights"
        )
    e = metacal_shear(g_meas, bins, response)
    g_true = reduced_shear(sub)
    mean_shear = weighted_mean_shear(e, w, bins, n_bins)
    logging.info("weighted mean shear per bin (subtracted):\n%s", mean_shear)
    e_main = e - (mean_shear[bins, 0] + 1j * mean_shear[bins, 1])

    t = args.test_tiles
    in_test = (
        (tile_r + rows[0] >= t[0])
        & (tile_r + rows[0] < t[1])
        & (tile_c + cols[0] >= t[2])
        & (tile_c + cols[0] < t[3])
    )
    a, a_err, add, add_err, n_cal = fit_shear_calibration(
        e, g_true, w, bins, n_bins, ~in_test
    )
    mult = np.stack([a[:, 0, 0], a[:, 1, 1]], axis=1) - 1.0  # (n_bins, 2)
    mult_err = np.stack([a_err[:, 0, 0], a_err[:, 1, 1]], axis=1)
    logging.info(
        "calibration outside the test region: m per bin\n%s\n+- %s\nc\n%s",
        mult,
        mult_err,
        add,
    )
    e_cal = (e.real - add[bins, 0]) / (1.0 + mult[bins, 0]) + 1j * (
        (e.imag - add[bins, 1]) / (1.0 + mult[bins, 1])
    )

    def grid(values):
        e1, wsum, count = tile_weighted_mean(
            values.real, w, bins, tile_r, tile_c, shape
        )
        e2, _, _ = tile_weighted_mean(values.imag, w, bins, tile_r, tile_c, shape)
        k_e, k_b = kaiser_squires(e1, e2, args.fft_pad_factor)
        return e1, e2, k_e, k_b, wsum, count

    actual = grid(e_main)
    calibrated = grid(e_cal)
    truth_same = grid(g_true)  # weighted tile mean of the same galaxies' true g
    # expected per-component noise variance of each tile mean: sum w^2 |e|^2/2 / (sum w)^2
    var_sum = np.zeros(shape)
    np.add.at(var_sum, (bins, tile_r, tile_c), w**2 * np.abs(e) ** 2 / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        noise_var = np.where(actual[4] > 0, var_sum / actual[4] ** 2, np.nan)
    rng = np.random.default_rng(args.seed)
    noise = []
    for k in range(args.rotations):
        phase = np.exp(2j * rng.uniform(0.0, np.pi, size=len(g_meas)))
        noise.append(grid(metacal_shear(g_meas * phase, bins, response)))
        logging.info("noise rotation %d of %d done", k + 1, args.rotations)
    sigma_e = np.array([np.std(e.real[bins == b]) for b in range(n_bins)])

    np.savez(
        args.output,
        e1=actual[0],
        e2=actual[1],
        kappa_ks=actual[2],
        kappa_ks_b=actual[3],
        weight_sum=actual[4],
        count=actual[5],
        noise_var=noise_var,
        e1_true_same=truth_same[0],
        e2_true_same=truth_same[1],
        e1_cal=calibrated[0],
        e2_cal=calibrated[1],
        kappa_ks_cal=calibrated[2],
        kappa_ks_b_cal=calibrated[3],
        e1_noise=np.stack([n[0] for n in noise]),
        e2_noise=np.stack([n[1] for n in noise]),
        kappa_ks_noise=np.stack([n[2] for n in noise]),
        R_shear=r_shear,
        R_select=r_select,
        mean_shear=mean_shear,
        weight_table=table["weight"],
        weight_table_count=table["count"],
        weight_table_R=table["R"],
        weight_table_sigma_e_sq=table["sigma_e_sq"],
        weight_grid_s2n=WEIGHT_GRID_S2N,
        weight_grid_size_ratio=WEIGHT_GRID_SIZE_RATIO,
        calib_A=a,
        calib_A_err=a_err,
        m=mult,
        m_err=mult_err,
        c=add,
        c_err=add_err,
        calib_counts=n_cal,
        test_tiles=np.array(t),
        frame_sign=METACAL_FRAME_SIGN,
        **{f"{f}_true": truth[f] for f in FIELDS},
        count_all=truth["count"],
        sigma_e=sigma_e,
        selection=args.selection,
        rotations=args.rotations,
        seed=args.seed,
        tracts_read=np.array(tracts),
        **meta,
        **extra,
    )
    logging.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
