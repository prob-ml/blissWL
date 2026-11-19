"""Global tract-level flow-matching sampler for padded DC2 crops."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from images_to_maps.dc2.dc2 import padded_crop_position  # noqa: E402

LOGGER = logging.getLogger(__name__)

FIELDS = ("shear_1", "shear_2", "convergence")
DEFAULT_CKPT_DIR = (
    "images_to_maps/results/dc2/flowmatching/"
    "padded_nanpsf_512_True_False_scale100.0/checkpoints"
)
DEFAULT_OUTPUT = (
    "images_to_maps/results/dc2/flowmatching/"
    "padded_nanpsf_512_True_False_scale100.0/global_stitched_samples"
)


@dataclass
class CropRecord:
    path_name: str
    valid_local_rows: torch.Tensor
    valid_local_cols: torch.Tensor
    valid_global_rows: torch.Tensor
    valid_global_cols: torch.Tensor
    interior_local_rows: torch.Tensor
    interior_local_cols: torch.Tensor
    interior_global_rows: torch.Tensor
    interior_global_cols: torch.Tensor
    invalid_local_rows: torch.Tensor
    invalid_local_cols: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a global tract-level flow-matching sampler by stitching local "
            "crop velocity evaluations at every ODE step."
        )
    )
    parser.add_argument("--config-name", default="config_train_flowmatching")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tracts",
        nargs="+",
        type=int,
        default=None,
        help="Restrict sampling to these tracts. Defaults to surveys.dc2.tracts.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
        help="Configured split to sample, or all files from the configured tracts.",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--num-ode-steps",
        type=int,
        default=None,
        help=(
            "Override the checkpoint/config ODE step count for inference. "
            "Defaults to the value saved in the loaded model."
        ),
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=2,
        help="Number of global posterior samples evaluated together.",
    )
    parser.add_argument(
        "--crop-batch-size",
        type=int,
        default=64,
        help="Number of local crop velocity evaluations batched per sample chunk.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save coverage and convergence diagnostic PNGs.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser.parse_args()


def load_config(config_name: str) -> Any:
    config_dir = Path(__file__).resolve().parent
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def load_file_paths(
    cfg: Any, cache_override: str | None, split_name: str
) -> list[Path]:
    """Select cache files with the configured data module, so the split matches training.

    ``split_name="all"`` returns every crop in the configured tracts, including
    buffer crops that are excluded from train/val/test.
    """
    dc2_cfg = cfg.surveys.dc2.copy()
    if cache_override is not None:
        dc2_cfg.cached_data_path = cache_override
    data_module = instantiate(dc2_cfg)
    if split_name == "all":
        names = data_module._list_file_names()
        return [Path(data_module.cached_data_path) / name for name in names]

    data_module._load_file_paths_and_slices()
    split_index = {"train": 0, "val": 1, "test": 2}[split_name]
    paths = data_module.file_paths[data_module.slices[split_index]]
    return [Path(path) for path in paths]


def file_tract(path: Path) -> int:
    return padded_crop_position(path.name)[0]


def load_datum(path: Path) -> dict[str, Any]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, list):
        if len(loaded) != 1:
            raise ValueError(f"Expected one datum in {path}, found {len(loaded)}")
        loaded = loaded[0]
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict datum in {path}, found {type(loaded)}")
    return loaded


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"encoder_epoch=(\d+)", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def resolve_checkpoint_path(ckpt_arg: str | None) -> Path:
    if ckpt_arg is not None:
        ckpt_path = Path(ckpt_arg)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
        return ckpt_path

    ckpt_dir = Path(DEFAULT_CKPT_DIR)
    checkpoints = sorted(ckpt_dir.glob("encoder_epoch=*.ckpt"), key=checkpoint_epoch)
    if not checkpoints:
        raise FileNotFoundError(
            f"No encoder_epoch=*.ckpt checkpoints found in {ckpt_dir}"
        )
    return checkpoints[-1]


def load_model(cfg: Any, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = cfg.train.encoder
    model_cfg.mode_metrics = None
    model_cfg.sample_metrics = None
    model_cfg.sample_image_renders = None
    model = instantiate(model_cfg)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint state dict did not match model. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )
    model.to(device)
    model.eval()
    return model


def make_inference_batch(
    datum: dict[str, Any], device: torch.device, batch_size: int
) -> dict[str, torch.Tensor]:
    images = datum["images"].unsqueeze(0).to(device)
    psf_params = datum["psf_params"].unsqueeze(0).to(device)
    if batch_size > 1:
        images = images.expand(batch_size, -1, -1, -1).contiguous()
        psf_params = psf_params.expand(batch_size, -1, -1).contiguous()
    return {"images": images, "psf_params": psf_params}


def crop_interior_indices(datum: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if "interior_mask" in datum:
        mask = datum["interior_mask"].bool()
    elif "loss_mask" in datum:
        mask = datum["loss_mask"].bool()
    else:
        raise KeyError("Datum has neither interior_mask nor loss_mask")

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D interior mask, got shape {tuple(mask.shape)}")

    local_rc = mask.nonzero(as_tuple=False)
    tract_rc = datum["tract_tile_row_col"][mask].to(torch.long)
    if local_rc.shape != tract_rc.shape:
        raise ValueError(
            "interior_mask and tract_tile_row_col selected different shapes: "
            f"{tuple(local_rc.shape)} vs {tuple(tract_rc.shape)}"
        )
    return local_rc, tract_rc


def infer_shape(datum: dict[str, Any]) -> tuple[int, int, int]:
    """Tract tile grid covered by crop interiors, and the number of redshift bins."""
    tract_shape = datum["tract_shape_tiles"].to(torch.long).tolist()
    if len(tract_shape) != 2:
        raise ValueError(f"Expected two tract tile dimensions, got {tract_shape}")
    full_h, full_w = (int(tract_shape[0]), int(tract_shape[1]))
    n_bins = int(datum["tile_catalog"]["convergence"].shape[-1])
    return full_h, full_w, n_bins


def finite_extent(ra: torch.Tensor, dec: torch.Tensor) -> tuple[tuple[float, ...], str]:
    ra_2d = ra[..., 0]
    dec_2d = dec[..., 0]
    finite = torch.isfinite(ra_2d) & torch.isfinite(dec_2d)
    if not finite.any():
        return (), "lower"

    ra_min = float(ra_2d[finite].min())
    ra_max = float(ra_2d[finite].max())
    dec_min = float(dec_2d[finite].min())
    dec_max = float(dec_2d[finite].max())

    first_row = torch.nanmean(dec_2d[0])
    last_row = torch.nanmean(dec_2d[-1])
    origin = "lower"
    if torch.isfinite(first_row) and torch.isfinite(last_row) and first_row > last_row:
        origin = "upper"
    return (ra_min, ra_max, dec_min, dec_max), origin


def plot_coverage(tract: int, coverage: torch.Tensor, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(coverage.numpy(), origin="lower", interpolation="nearest")
    ax.set_title(f"Tract {tract} stitched coverage")
    ax.set_xlabel("stitched tile column")
    ax.set_ylabel("stitched tile row")
    fig.colorbar(im, ax=ax, label="interior crop writes")
    fig.savefig(output_dir / f"tract_{tract}_coverage.png", dpi=180)
    plt.close(fig)


def imshow_map(
    ax: plt.Axes,
    array: torch.Tensor,
    title: str,
    vmin: float,
    vmax: float,
    extent: tuple[float, ...],
    origin: str,
) -> Any:
    kwargs: dict[str, Any] = {
        "origin": origin,
        "interpolation": "nearest",
        "vmin": vmin,
        "vmax": vmax,
        "cmap": "coolwarm",
    }
    if extent:
        kwargs["extent"] = extent
    im = ax.imshow(array.numpy(), **kwargs)
    if extent:
        ax.invert_xaxis()
        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
    else:
        ax.set_xlabel("stitched tile column")
        ax.set_ylabel("stitched tile row")
    ax.set_title(title)
    return im


def finite_min_max(*arrays: torch.Tensor) -> tuple[float, float] | None:
    values = torch.cat([array.reshape(-1) for array in arrays])
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return float(values.min()), float(values.max())


def finite_abs_max(array: torch.Tensor) -> float | None:
    values = array.reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return float(values.abs().max())


def plot_convergence_summary(
    tract: int,
    target: torch.Tensor,
    posterior_mean: torch.Tensor,
    ra: torch.Tensor,
    dec: torch.Tensor,
    output_dir: Path,
) -> None:
    n_bins = target.shape[-1]
    extent, origin = finite_extent(ra, dec)
    fig, axes = plt.subplots(
        n_bins,
        3,
        figsize=(12, 3.2 * n_bins),
        squeeze=False,
        constrained_layout=True,
    )

    for bin_idx in range(n_bins):
        true_map = target[..., bin_idx]
        mean_map = posterior_mean[..., bin_idx]
        residual = mean_map - true_map

        signal_limits = finite_min_max(true_map, mean_map)
        if signal_limits is None:
            signal_min, signal_max = -1.0, 1.0
        else:
            signal_min, signal_max = signal_limits

        res_abs = finite_abs_max(residual)
        if res_abs is None:
            res_abs = 1.0
        elif res_abs == 0:
            res_abs = 1.0

        im_true = imshow_map(
            axes[bin_idx, 0],
            true_map,
            f"z bin {bin_idx}: target convergence",
            signal_min,
            signal_max,
            extent,
            origin,
        )
        im_mean = imshow_map(
            axes[bin_idx, 1],
            mean_map,
            f"z bin {bin_idx}: posterior mean",
            signal_min,
            signal_max,
            extent,
            origin,
        )
        im_res = imshow_map(
            axes[bin_idx, 2],
            residual,
            f"z bin {bin_idx}: mean - target",
            -res_abs,
            res_abs,
            extent,
            origin,
        )
        fig.colorbar(im_true, ax=axes[bin_idx, 0], shrink=0.8)
        fig.colorbar(im_mean, ax=axes[bin_idx, 1], shrink=0.8)
        fig.colorbar(im_res, ax=axes[bin_idx, 2], shrink=0.8)

    fig.savefig(output_dir / f"tract_{tract}_convergence_summary.png", dpi=180)
    plt.close(fig)


def rmse_and_corr(
    target: torch.Tensor, prediction: torch.Tensor, valid: torch.Tensor
) -> dict[str, float]:
    target_valid = target[valid]
    pred_valid = prediction[valid]
    finite = torch.isfinite(target_valid) & torch.isfinite(pred_valid)
    if finite.sum() == 0:
        return {"rmse": math.nan, "pearson": math.nan}

    target_valid = target_valid[finite]
    pred_valid = pred_valid[finite]
    rmse = torch.sqrt(torch.mean((pred_valid - target_valid) ** 2)).item()
    if target_valid.numel() < 2 or target_valid.std() == 0 or pred_valid.std() == 0:
        pearson = math.nan
    else:
        pearson = torch.corrcoef(torch.stack([target_valid, pred_valid]))[0, 1].item()
    return {"rmse": rmse, "pearson": pearson}


def compute_metrics(
    targets: dict[str, torch.Tensor],
    posterior_mean: dict[str, torch.Tensor],
    coverage: torch.Tensor,
) -> dict[str, dict[str, list[float]]]:
    valid = coverage > 0
    metrics: dict[str, dict[str, list[float]]] = {}
    for field in FIELDS:
        field_metrics = {"rmse": [], "pearson": []}
        for bin_idx in range(targets[field].shape[-1]):
            values = rmse_and_corr(
                targets[field][..., bin_idx],
                posterior_mean[field][..., bin_idx],
                valid,
            )
            field_metrics["rmse"].append(values["rmse"])
            field_metrics["pearson"].append(values["pearson"])
        metrics[field] = field_metrics
    return metrics


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as output_file:
        json.dump(data, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def group_paths_by_tract(paths: list[Path]) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[file_tract(path)].append(path)
    return dict(sorted(grouped.items()))


def empty_targets(
    full_h: int, full_w: int, n_bins: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = {
        field: torch.full((full_h, full_w, n_bins), torch.nan, dtype=torch.float32)
        for field in FIELDS
    }
    ra = torch.full((full_h, full_w, n_bins), torch.nan, dtype=torch.float32)
    dec = torch.full((full_h, full_w, n_bins), torch.nan, dtype=torch.float32)
    coverage = torch.zeros((full_h, full_w), dtype=torch.int16)
    return targets, ra, dec, coverage


def build_crop_record(
    path: Path,
    datum: dict[str, Any],
    full_h: int,
    full_w: int,
) -> CropRecord:
    tract_rc = datum["tract_tile_row_col"].to(torch.long)
    global_rows = tract_rc[..., 0]
    global_cols = tract_rc[..., 1]
    valid = (
        (global_rows >= 0)
        & (global_rows < full_h)
        & (global_cols >= 0)
        & (global_cols < full_w)
    )
    valid_local = valid.nonzero(as_tuple=False)
    invalid_local = (~valid).nonzero(as_tuple=False)

    interior_local, interior_tract = crop_interior_indices(datum)
    interior_global_rows = interior_tract[:, 0]
    interior_global_cols = interior_tract[:, 1]
    interior_valid = (
        (interior_global_rows >= 0)
        & (interior_global_rows < full_h)
        & (interior_global_cols >= 0)
        & (interior_global_cols < full_w)
    )
    if not bool(interior_valid.all()):
        bad = interior_local[~interior_valid].tolist()
        raise ValueError(f"Interior tiles outside global tract for {path.name}: {bad}")

    return CropRecord(
        path_name=path.name,
        valid_local_rows=valid_local[:, 0].to(torch.long),
        valid_local_cols=valid_local[:, 1].to(torch.long),
        valid_global_rows=global_rows[valid].to(torch.long),
        valid_global_cols=global_cols[valid].to(torch.long),
        interior_local_rows=interior_local[:, 0].to(torch.long),
        interior_local_cols=interior_local[:, 1].to(torch.long),
        interior_global_rows=interior_global_rows.to(torch.long),
        interior_global_cols=interior_global_cols.to(torch.long),
        invalid_local_rows=invalid_local[:, 0].to(torch.long),
        invalid_local_cols=invalid_local[:, 1].to(torch.long),
    )


def place_targets(
    datum: dict[str, Any],
    record: CropRecord,
    targets: dict[str, torch.Tensor],
    ra: torch.Tensor,
    dec: torch.Tensor,
    coverage: torch.Tensor,
) -> None:
    lr = record.interior_local_rows
    lc = record.interior_local_cols
    gr = record.interior_global_rows
    gc = record.interior_global_cols
    if bool(coverage[gr, gc].any()):
        duplicate = torch.stack(
            [gr[coverage[gr, gc] > 0], gc[coverage[gr, gc] > 0]], dim=1
        )
        raise ValueError(f"Duplicate interior target placement: {duplicate.tolist()}")
    for field in FIELDS:
        targets[field][gr, gc, :] = datum["tile_catalog"][field][lr, lc, :]
    ra[gr, gc, :] = datum["tile_catalog"]["ra"][lr, lc, :]
    dec[gr, gc, :] = datum["tile_catalog"]["dec"][lr, lc, :]
    coverage[gr, gc] += 1


def precompute_crop_records_and_embeddings(
    paths: list[Path],
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[
    list[CropRecord],
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
]:
    first = load_datum(paths[0])
    full_h, full_w, n_bins = infer_shape(first)
    targets, ra, dec, coverage = empty_targets(full_h, full_w, n_bins)
    records: list[CropRecord] = []
    embeddings = []

    LOGGER.info("Precomputing image embeddings for %d crop files", len(paths))
    for index, path in enumerate(paths, start=1):
        datum = first if index == 1 else load_datum(path)
        record = build_crop_record(path, datum, full_h, full_w)
        place_targets(datum, record, targets, ra, dec, coverage)
        with torch.inference_mode():
            batch = make_inference_batch(datum, device, batch_size=1)
            embeddings.append(
                model.encode_conditioning(batch).detach().cpu().squeeze(0)
            )
        records.append(record)
        if index % 25 == 0 or index == len(paths):
            LOGGER.info(
                "Precomputed %d/%d embeddings; coverage=%d/%d tiles",
                index,
                len(paths),
                int((coverage > 0).sum()),
                coverage.numel(),
            )

    return (
        records,
        torch.stack(embeddings, dim=0),
        targets,
        ra,
        dec,
        coverage,
        (full_h, full_w, n_bins),
    )


def evaluate_global_velocity(
    model: torch.nn.Module,
    global_z: torch.Tensor,
    halo_z: torch.Tensor,
    embeddings: torch.Tensor,
    records: list[CropRecord],
    t_value: float,
    sample_batch_size: int,
    crop_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_velocity = torch.zeros_like(global_z)
    halo_velocity = torch.zeros_like(halo_z)
    n_samples, z_dim, _full_h, _full_w = global_z.shape
    n_crops = len(records)

    for sample_start in range(0, n_samples, sample_batch_size):
        sample_stop = min(sample_start + sample_batch_size, n_samples)
        batch_n = sample_stop - sample_start
        global_chunk = global_z[sample_start:sample_stop]
        halo_chunk = halo_z[sample_start:sample_stop]
        global_velocity_chunk = torch.zeros_like(global_chunk)
        halo_velocity_chunk = torch.zeros_like(halo_chunk)

        for crop_start in range(0, n_crops, crop_batch_size):
            crop_stop = min(crop_start + crop_batch_size, n_crops)
            crop_records = records[crop_start:crop_stop]
            crop_n = crop_stop - crop_start
            local_z = halo_chunk[:, crop_start:crop_stop].clone()

            for crop_offset, record in enumerate(crop_records):
                if record.valid_local_rows.numel() == 0:
                    continue
                local_z[
                    :,
                    crop_offset,
                    :,
                    record.valid_local_rows.to(device),
                    record.valid_local_cols.to(device),
                ] = global_chunk[
                    :,
                    :,
                    record.valid_global_rows.to(device),
                    record.valid_global_cols.to(device),
                ]

            local_z_flat = local_z.reshape(batch_n * crop_n, z_dim, 8, 8)
            x_embedding = embeddings[crop_start:crop_stop].to(device)
            x_embedding = (
                x_embedding.unsqueeze(0)
                .expand(batch_n, -1, -1, -1, -1)
                .reshape(batch_n * crop_n, *x_embedding.shape[1:])
            )
            t = torch.full((batch_n * crop_n, 1), t_value, device=device)
            with torch.inference_mode():
                local_velocity = model.velocity(local_z_flat, x_embedding, t)
            local_velocity = local_velocity.reshape(batch_n, crop_n, z_dim, 8, 8)

            for crop_offset, record in enumerate(crop_records):
                global_velocity_chunk[
                    :,
                    :,
                    record.interior_global_rows.to(device),
                    record.interior_global_cols.to(device),
                ] = local_velocity[
                    :,
                    crop_offset,
                    :,
                    record.interior_local_rows.to(device),
                    record.interior_local_cols.to(device),
                ]
                if record.invalid_local_rows.numel() > 0:
                    halo_velocity_chunk[
                        :,
                        crop_start + crop_offset,
                        :,
                        record.invalid_local_rows.to(device),
                        record.invalid_local_cols.to(device),
                    ] = local_velocity[
                        :,
                        crop_offset,
                        :,
                        record.invalid_local_rows.to(device),
                        record.invalid_local_cols.to(device),
                    ]

        global_velocity[sample_start:sample_stop] = global_velocity_chunk
        halo_velocity[sample_start:sample_stop] = halo_velocity_chunk

    return global_velocity, halo_velocity


def integrate_global_ode(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    records: list[CropRecord],
    num_samples: int,
    full_h: int,
    full_w: int,
    sample_batch_size: int,
    crop_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    global_z = torch.randn(num_samples, model.z_dim, full_h, full_w, device=device)
    halo_z = torch.randn(num_samples, len(records), model.z_dim, 8, 8, device=device)
    dt = 1.0 / model.num_ode_steps

    for step in range(model.num_ode_steps):
        t = step * dt
        LOGGER.info("Global ODE step %d/%d at t=%.3f", step + 1, model.num_ode_steps, t)
        global_velocity, halo_velocity = evaluate_global_velocity(
            model=model,
            global_z=global_z,
            halo_z=halo_z,
            embeddings=embeddings,
            records=records,
            t_value=t,
            sample_batch_size=sample_batch_size,
            crop_batch_size=crop_batch_size,
            device=device,
        )
        global_mid = global_z + 0.5 * dt * global_velocity
        halo_mid = halo_z + 0.5 * dt * halo_velocity
        global_mid_velocity, halo_mid_velocity = evaluate_global_velocity(
            model=model,
            global_z=global_mid,
            halo_z=halo_mid,
            embeddings=embeddings,
            records=records,
            t_value=t + 0.5 * dt,
            sample_batch_size=sample_batch_size,
            crop_batch_size=crop_batch_size,
            device=device,
        )
        global_z = global_z + dt * global_mid_velocity
        halo_z = halo_z + dt * halo_mid_velocity

    return global_z / model.scale_factor


def z_samples_to_field_maps(
    model: torch.nn.Module, z_samples: torch.Tensor
) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        return {
            field: value.detach().cpu()
            for field, value in model.z_to_catalog(z_samples).items()
        }


def sample_global_tract(
    *,
    tract: int,
    paths: list[Path],
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    ckpt_path: Path,
    config_name: str,
    num_samples: int,
    sample_batch_size: int,
    crop_batch_size: int,
    allow_incomplete: bool,
    overwrite: bool,
    save_plots: bool,
) -> None:
    out_path = output_dir / f"tract_{tract}_global_flowmatching_stitched_samples.pt"
    metrics_path = (
        output_dir / f"tract_{tract}_global_flowmatching_stitched_metrics.json"
    )
    if not overwrite and (out_path.exists() or metrics_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out_path}")

    (
        records,
        embeddings,
        targets,
        ra,
        dec,
        coverage,
        (full_h, full_w, _n_bins),
    ) = precompute_crop_records_and_embeddings(paths, model, device)

    duplicate_count = int((coverage > 1).sum())
    missing_count = int((coverage == 0).sum())
    if duplicate_count:
        raise RuntimeError(
            f"Tract {tract} has {duplicate_count} duplicate target tiles"
        )
    if missing_count and not allow_incomplete:
        raise RuntimeError(
            f"Tract {tract} is missing {missing_count} target tiles; "
            "pass --allow-incomplete for smoke tests."
        )

    z_samples = integrate_global_ode(
        model=model,
        embeddings=embeddings,
        records=records,
        num_samples=num_samples,
        full_h=full_h,
        full_w=full_w,
        sample_batch_size=sample_batch_size,
        crop_batch_size=crop_batch_size,
        device=device,
    )
    samples = z_samples_to_field_maps(model, z_samples)
    posterior_mean = {field: value.mean(dim=0) for field, value in samples.items()}
    metrics = compute_metrics(targets, posterior_mean, coverage)

    output = {
        "tract": tract,
        "paths": [path.name for path in paths],
        "fields": FIELDS,
        "num_samples": num_samples,
        "num_ode_steps": int(model.num_ode_steps),
        "checkpoint": str(ckpt_path),
        "config_name": config_name,
        "sampler": "global_velocity_stitching",
        "coverage": coverage,
        "ra": ra,
        "dec": dec,
        "samples": samples,
        "posterior_mean": posterior_mean,
        "target": targets,
        "metrics": metrics,
    }
    torch.save(output, out_path)
    save_json(
        metrics_path,
        {
            "tract": tract,
            "num_crop_files": len(paths),
            "num_samples": num_samples,
            "num_ode_steps": int(model.num_ode_steps),
            "coverage_tiles": int((coverage > 0).sum()),
            "duplicate_tiles": duplicate_count,
            "missing_tiles": missing_count,
            "metrics": metrics,
            "output": str(out_path),
            "sampler": "global_velocity_stitching",
        },
    )
    if save_plots:
        plot_coverage(tract, coverage, output_dir)
        plot_convergence_summary(
            tract,
            targets["convergence"],
            posterior_mean["convergence"],
            ra,
            dec,
            output_dir,
        )
    LOGGER.info("Wrote %s", out_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if args.sample_batch_size < 1:
        raise ValueError("--sample-batch-size must be positive")
    if args.crop_batch_size < 1:
        raise ValueError("--crop-batch-size must be positive")
    if args.num_ode_steps is not None and args.num_ode_steps < 1:
        raise ValueError("--num-ode-steps must be positive")

    cfg = load_config(args.config_name)
    selected_paths = load_file_paths(cfg, args.cache, args.split)
    if args.tracts is not None:
        wanted = set(args.tracts)
        selected_paths = [path for path in selected_paths if file_tract(path) in wanted]
    if args.max_files is not None:
        selected_paths = selected_paths[: args.max_files]
    if not selected_paths:
        raise FileNotFoundError("No files selected after split and tract filtering")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = resolve_checkpoint_path(args.ckpt)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    LOGGER.info("Using device %s", device)
    LOGGER.info("Using checkpoint %s", ckpt_path)
    LOGGER.info("Selected %d crop files from split %s", len(selected_paths), args.split)

    model = load_model(cfg, ckpt_path, device)
    if args.num_ode_steps is not None:
        LOGGER.info(
            "Overriding model.num_ode_steps from %d to %d",
            model.num_ode_steps,
            args.num_ode_steps,
        )
        model.num_ode_steps = args.num_ode_steps
    grouped = group_paths_by_tract(selected_paths)
    LOGGER.info("Selected tracts: %s", sorted(grouped))
    for tract, paths in grouped.items():
        sample_global_tract(
            tract=tract,
            paths=paths,
            model=model,
            device=device,
            output_dir=output_dir,
            ckpt_path=ckpt_path,
            config_name=args.config_name,
            num_samples=args.num_samples,
            sample_batch_size=args.sample_batch_size,
            crop_batch_size=args.crop_batch_size,
            allow_incomplete=args.allow_incomplete,
            overwrite=args.overwrite,
            save_plots=args.save_plots,
        )


if __name__ == "__main__":
    main()
