"""Load native CosmoGrid patches using preassigned spatial splits.

Return float32 [C,H,W] maps and six cosmological parameters.
Support mean subtraction, smoothing, and training-only augmentation.
Compute parameter statistics from training data only.
"""

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from itertools import accumulate
import json
import math
import os
from pathlib import Path

from lightning import LightningDataModule
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import gaussian_blur


SPLITS = ("train", "val", "test")
PARAM_NAMES = ("omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0")


@dataclass(frozen=True)
class _BatchInfo:
    path: Path
    count: int
    parameters: torch.Tensor
    source_file: str
    cosmology_id: int
    permutation_id: int


def _load_batch(path, mmap=True):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _build_train_region_reference(manifest):
    """Return a fixed spherical reference, never the selected patch footprint."""
    import healpy as hp

    from maps_to_cosmology.convert_cosmogrid_fixed_regions import (
        cell_pixel_ids,
        validate_config,
    )

    config = manifest["config"]
    validate_config(config)
    nside = config["input"]["expected_nside"]
    ordering = config["input"]["ordering"]
    faces = sorted(config["regions"]["train"]["faces"])
    dec_range = config["layout"]["dec_range_deg"]
    parts = []
    for face in faces:
        ids = cell_pixel_ids(nside, face, 0, 0, nside, nside, ordering).ravel()
        if dec_range is not None:
            _, dec = hp.pix2ang(nside, ids, nest=ordering == "NESTED", lonlat=True)
            ids = ids[(dec >= dec_range[0]) & (dec <= dec_range[1])]
        parts.append(ids)
    _require(
        bool(parts) and sum(ids.size for ids in parts) > 0,
        "empty train reference region",
    )
    ids = np.sort(np.concatenate(parts))
    reference = {
        "definition": "base_faces_and_pixel_center_dec_v1",
        "nside": nside,
        "ordering": ordering,
        "faces": faces,
        "dec_range_deg": dec_range,
        "num_pixels": int(ids.size),
        "pixel_ids_sha256": hashlib.sha256(
            ids.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }
    return reference, ids


def _train_region_stats_sha256(stats):
    digest = hashlib.sha256(
        json.dumps(
            {k: stats[k] for k in ("schema_version", "region", "source_ids")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for source in sorted(stats["means"]):
        digest.update(json.dumps(source).encode())
        digest.update(
            stats["means"][source].detach().cpu().numpy().astype("<f8").tobytes()
        )
    return digest.hexdigest()


def _validate_train_region_stats(stats, reference, source_ids):
    _require(
        isinstance(stats, dict) and stats.get("schema_version") == 1,
        "invalid train-region statistics",
    )
    _require(stats.get("region") == reference, "train reference region mismatch")
    _require(
        stats.get("source_ids") == source_ids,
        "train-region statistics sources differ from dataset",
    )
    means = stats.get("means")
    _require(
        isinstance(means, dict) and set(means) == set(source_ids),
        "missing/extra train-region source means",
    )
    for source, value in means.items():
        _require(
            isinstance(value, torch.Tensor)
            and value.dtype == torch.float64
            and value.shape == (4,)
            and bool(torch.isfinite(value).all()),
            f"invalid train-region mean: {source}",
        )
    _require(
        stats.get("stats_sha256") == _train_region_stats_sha256(stats),
        "train-region statistics checksum mismatch",
    )
    return {
        **stats,
        "means": {k: v.detach().cpu().clone() for k, v in means.items()},
    }


def prepare_train_region_stats(data_dir, output_file=None):
    """Read original full skies once and write only a new statistics file."""
    from tqdm import tqdm

    from maps_to_cosmology.convert_cosmogrid_fixed_regions import load_native_sky

    data_dir = Path(data_dir)
    output_file = (
        Path(output_file)
        if output_file is not None
        else data_dir / "train_region_means.pt"
    )
    if output_file.exists():
        raise FileExistsError(output_file)
    dm = NativeCosmoGridMapsModule(
        data_dir=str(data_dir), num_workers=0, standardize_params=False
    )
    dm.setup()
    manifest = json.loads((data_dir / "conversion_metadata.json").read_text())
    reference, ids = _build_train_region_reference(manifest)
    source_dir = Path(manifest["config"]["input"]["directory"]) / "maps"
    means, source_ids = {}, {}
    for record in tqdm(dm.train_dataset.records, desc="Train-region means"):
        _require(
            Path(record.source_file).name == record.source_file,
            "expected a source file basename",
        )
        sky, _ = load_native_sky(
            source_dir / record.source_file, reference["nside"], reference["ordering"]
        )
        means[record.source_file] = torch.from_numpy(
            np.asarray(
                [channel[ids].mean(dtype=np.float64) for channel in sky],
                dtype=np.float64,
            )
        )
        source_ids[record.source_file] = (record.cosmology_id, record.permutation_id)
        del sky
    stats = {
        "schema_version": 1,
        "region": reference,
        "source_ids": source_ids,
        "means": means,
    }
    stats["stats_sha256"] = _train_region_stats_sha256(stats)
    stats = _validate_train_region_stats(stats, reference, source_ids)
    # Exclusive creation also prevents overwriting a file created during the scan.
    with output_file.open("xb") as handle:
        torch.save(stats, handle)
    print(f"Saved {len(means)} source means over {ids.size} pixels to {output_file}")


def _scan_split(paths, split_id, manifest, geometry, subtract_mean, mmap):
    """Check headers/small tensors without touching every map data page."""
    records = []
    rows = torch.nonzero(geometry["split_ids"] == split_id, as_tuple=True)[0]
    _require(rows.numel() > 0, f"{SPLITS[split_id]} has no selected patches")
    expected_ids = geometry["patch_ids"][rows]
    expected_shape = (len(rows), *manifest["map_shape"])
    for path in paths:
        batch = _load_batch(path, mmap)
        prefix = f"{path}: "
        for key in (
            "maps",
            "params",
            "geometry_rows",
            "patch_ids",
            "split_ids",
            "cosmology_ids",
            "permutation_ids",
            "full_sky_means",
        ):
            _require(
                isinstance(batch.get(key), torch.Tensor),
                prefix + f"missing tensor {key}",
            )
        _require(
            batch.get("geometry_sha256") == manifest["geometry_sha256"],
            prefix + "geometry hash mismatch",
        )
        _require(
            batch.get("layout_sha256") == manifest["layout_sha256"],
            prefix + "layout hash mismatch",
        )
        maps, parameters = batch["maps"], batch["params"]
        _require(
            tuple(maps.shape) == expected_shape,
            prefix + "map shape does not match shared geometry",
        )
        _require(
            maps.dtype in (torch.float32, torch.float64),
            prefix + "unsupported map dtype",
        )
        _require(
            tuple(parameters.shape) == (len(rows), len(PARAM_NAMES)),
            prefix + "expected params [N,6]",
        )
        _require(
            parameters.is_floating_point() and bool(torch.isfinite(parameters).all()),
            prefix + "invalid cosmology parameters",
        )
        _require(
            torch.equal(parameters, parameters[:1].expand_as(parameters)),
            prefix + "expected one cosmology per source batch",
        )
        _require(
            torch.equal(batch["geometry_rows"], rows),
            prefix + "wrong shared geometry rows",
        )
        _require(
            torch.equal(batch["patch_ids"], expected_ids),
            prefix + "wrong patch IDs/order",
        )
        _require(
            tuple(batch["split_ids"].shape) == (len(rows),)
            and bool((batch["split_ids"] == split_id).all()),
            prefix + "wrong spatial split",
        )
        for key in ("cosmology_ids", "permutation_ids"):
            values = batch[key]
            _require(
                tuple(values.shape) == (len(rows),) and values.dtype == torch.long,
                prefix + f"expected integer {key} [N]",
            )
            _require(
                bool((values == values[0]).all()),
                prefix + f"multiple {key} in one batch",
            )
        means = batch["full_sky_means"]
        _require(
            tuple(means.shape) == (manifest["map_shape"][-1],),
            prefix + "wrong full_sky_means shape",
        )
        if subtract_mean:
            _require(
                means.is_floating_point() and bool(torch.isfinite(means).all()),
                prefix + "cannot subtract nonfinite means",
            )
        source_file = batch.get("source_file")
        _require(
            isinstance(source_file, str) and bool(source_file),
            prefix + "missing source_file",
        )
        records.append(
            _BatchInfo(
                path=path,
                count=len(rows),
                parameters=parameters[0].clone(),
                source_file=source_file,
                cosmology_id=int(batch["cosmology_ids"][0]),
                permutation_id=int(batch["permutation_ids"][0]),
            )
        )
        del batch
    return records


def _train_statistics(records, eps):
    """Weighted Welford statistics for parameters repeated within each file."""
    mean = torch.zeros(len(PARAM_NAMES), dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    count = 0
    for record in records:
        new_count = count + record.count
        delta = record.parameters.to(torch.float64) - mean
        mean += delta * (record.count / new_count)
        m2 += delta.square() * (count * record.count / new_count)
        count = new_count
    return mean.float(), (m2 / count).sqrt().clamp_min(eps).float()


class NativeCosmoGridDataset(Dataset):
    """Map-style dataset with bounded per-process batch-file caches."""

    def __init__(
        self,
        records,
        *,
        param_mean=None,
        param_std=None,
        subtract_full_sky_mean=False,
        smoothing_sigma=None,
        augment=False,
        cache_size=4,
        mmap=True,
        train_region_means=None,
    ):
        self.records = tuple(records)
        self.cumulative_sizes = list(accumulate(r.count for r in records))
        self.param_mean = param_mean
        self.param_std = param_std
        self.subtract_full_sky_mean = subtract_full_sky_mean
        self.train_region_means = train_region_means
        self.smoothing_sigma = smoothing_sigma
        self.augment = augment
        self.cache_size = cache_size
        self.mmap = mmap
        self._cache = OrderedDict()
        self._cache_pid = None

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def clear_cache(self):
        self._cache.clear()
        self._cache_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        state["_cache_pid"] = None
        return state

    def _get_batch(self, file_index):
        pid = os.getpid()
        if self._cache_pid != pid:
            self.clear_cache()
            self._cache_pid = pid
        if file_index in self._cache:
            self._cache.move_to_end(file_index)
            return self._cache[file_index]
        batch = _load_batch(self.records[file_index].path, self.mmap)
        if self.cache_size:
            self._cache[file_index] = batch
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return batch

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = bisect_right(self.cumulative_sizes, index)
        offset = self.cumulative_sizes[file_index - 1] if file_index else 0
        local_index = index - offset
        batch = self._get_batch(file_index)
        raw = batch["maps"][local_index]
        if self.train_region_means is not None:
            mean = self.train_region_means[batch["source_file"]]
            maps = (raw.double() - mean.view(1, 1, -1)).float()
        elif self.subtract_full_sky_mean:
            maps = (
                raw.double() - batch["full_sky_means"].double().view(1, 1, -1)
            ).float()
        else:
            # A float32 .to() alone could alias the mmap; always own the sample.
            maps = raw.to(torch.float32).clone()
        maps = maps.permute(2, 0, 1).contiguous()
        if not bool(torch.isfinite(maps).all()):
            raise ValueError(
                f"nonfinite map in {self.records[file_index].path}, row {local_index}"
            )
        if self.smoothing_sigma is not None:
            kernel_size = 2 * round(2 * self.smoothing_sigma) + 1
            maps = gaussian_blur(
                maps,
                [kernel_size, kernel_size],
                [self.smoothing_sigma, self.smoothing_sigma],
            )
        if self.augment:
            choice = int(torch.randint(4, ()).item())
            if choice == 1:
                maps = maps.flip(-2)
            elif choice == 2:
                maps = maps.flip(-1)
            elif choice == 3:
                maps = maps.rot90(2, (-2, -1))
        parameters = batch["params"][local_index].to(torch.float32).clone()
        if self.param_mean is not None:
            parameters = (parameters - self.param_mean) / self.param_std
        return maps.contiguous(), parameters


class NativeCosmoGridMapsModule(LightningDataModule):
    """Use native train/val/test directories without splitting them again.

    cache_size counts mapped files PER worker. With mmap=True, map payloads
    are paged in on demand; setup reads only metadata and small label tensors.
    Source batches must come from one completed, consistent conversion.
    subtract_full_sky_mean defaults to False to retain native input values.
    Train-region subtraction reads a prepared table, or restores it from a
    checkpoint; setup never generates statistics or rewrites map files.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        standardize_params: bool = True,
        subtract_full_sky_mean: bool = False,
        smoothing_sigma: float | None = None,
        augment: bool = False,
        cache_size: int = 4,
        mmap: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        eps: float = 1e-8,
        subtract_train_region_mean: bool = False,
        train_region_stats_file: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        for name, value, minimum in (
            ("batch_size", batch_size, 1),
            ("num_workers", num_workers, 0),
            ("cache_size", cache_size, 0),
            ("seed", seed, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if smoothing_sigma is not None and (
            not math.isfinite(smoothing_sigma) or smoothing_sigma <= 0
        ):
            raise ValueError("smoothing_sigma must be None or finite and positive")
        self.data_dir = Path(data_dir)
        if subtract_full_sky_mean and subtract_train_region_mean:
            raise ValueError("enable only one mean-subtraction method")
        self.subtract_train_region_mean = subtract_train_region_mean
        self.train_region_stats_file = (
            Path(train_region_stats_file)
            if train_region_stats_file is not None
            else self.data_dir / "train_region_means.pt"
        )
        self.train_region_stats = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.standardize_params = standardize_params
        self.subtract_full_sky_mean = subtract_full_sky_mean
        self.smoothing_sigma = smoothing_sigma
        self.augment = augment
        self.cache_size = cache_size
        self.mmap = mmap
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.eps = eps
        self.param_mean = None
        self.param_std = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.geometry_sha256 = None
        self.dataset_sha256 = None
        self._restored_state = None

    def setup(self, stage: str | None = None):
        if self.train_dataset is not None:
            return
        manifest_path = self.data_dir / "conversion_metadata.json"
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        _require(
            manifest.get("status") == "complete",
            "native conversion is incomplete; wait until status=complete",
        )
        _require(
            manifest.get("estimator") == "native_healpix_pixel_gather",
            "expected native HEALPix converter output",
        )
        _require(
            manifest.get("full_sky_monopole_subtracted") is False,
            "expected unmodified native maps; refusing possible double mean subtraction",
        )
        file_count = manifest.get("num_input_files")
        _require(
            isinstance(file_count, int)
            and file_count > 0
            and manifest.get("completed_input_files") == file_count,
            "manifest input-file counts are inconsistent",
        )
        _require(
            tuple(manifest.get("param_names", ())) == PARAM_NAMES,
            "unexpected cosmological parameter order",
        )
        shape = manifest.get("map_shape", [])
        _require(
            len(shape) == 3
            and all(isinstance(x, int) and x > 0 for x in shape)
            and shape[-1] == 4,
            "expected manifest map_shape [H,W,4]",
        )
        if self.smoothing_sigma is not None:
            _require(
                round(2 * self.smoothing_sigma) < min(shape[:2]),
                "smoothing kernel is too large for the map",
            )
        geometry = _load_batch(self.data_dir / "patch_geometry.pt", self.mmap)
        for key in ("geometry_sha256", "layout_sha256", "healpix_ordering"):
            _require(
                geometry.get(key) is not None
                and geometry.get(key) == manifest.get(key),
                f"shared geometry {key} mismatch",
            )
        indices = geometry["healpix_pixel_ids"]
        _require(
            indices.dtype == torch.long
            and indices.ndim == 3
            and tuple(indices.shape[1:]) == tuple(shape[:2]),
            "invalid shared native pixel IDs",
        )
        n = indices.shape[0]
        for key in ("patch_ids", "split_ids"):
            _require(
                geometry[key].dtype == torch.long
                and tuple(geometry[key].shape) == (n,),
                f"invalid geometry {key}",
            )
        _require(
            bool(
                (
                    (geometry["split_ids"] >= 0) & (geometry["split_ids"] < len(SPLITS))
                ).all()
            ),
            "invalid geometry split IDs",
        )
        native_ids = indices.numpy()
        _require(
            native_ids.size > 0
            and native_ids.min() >= 0
            and native_ids.max() < 12 * geometry["nside"] ** 2,
            "native pixel ID outside HEALPix map",
        )
        _require(
            np.unique(native_ids).size == native_ids.size,
            "shared geometry reuses native pixels across patches/splits",
        )
        _require(
            torch.unique(geometry["patch_ids"]).numel() == n,
            "duplicate geometry patch IDs",
        )
        expected_names = [f"batch_{i:06d}.pt" for i in range(file_count)]
        records = {}
        for split_id, split in enumerate(SPLITS):
            paths = sorted((self.data_dir / split).glob("batch_*.pt"))
            _require(
                [p.name for p in paths] == expected_names,
                f"{split}: missing/extra batch files relative to completed manifest",
            )
            records[split] = _scan_split(
                paths,
                split_id,
                manifest,
                geometry,
                self.subtract_full_sky_mean,
                self.mmap,
            )
            _require(
                sum(r.count for r in records[split])
                == manifest["maps_per_split"][split],
                f"{split}: map count mismatch",
            )

        # Every source must appear in all splits, as intended by the fixed layout.
        def identities(records_for_split):
            return [
                (r.source_file, r.cosmology_id, r.permutation_id)
                for r in records_for_split
            ]

        reference = identities(records["train"])
        _require(len(set(reference)) == len(reference), "duplicate full-sky sources")
        for split in ("val", "test"):
            _require(
                identities(records[split]) == reference,
                f"{split}: full-sky sources differ from train",
            )
            _require(
                all(
                    torch.equal(a.parameters, b.parameters)
                    for a, b in zip(records["train"], records[split], strict=True)
                ),
                f"{split}: cosmology labels differ from matching train sources",
            )
        self.geometry_sha256 = manifest["geometry_sha256"]
        signature = {
            "geometry": self.geometry_sha256,
            "sources": {
                s: [
                    (
                        r.source_file,
                        r.cosmology_id,
                        r.permutation_id,
                        r.count,
                        r.parameters.tolist(),
                    )
                    for r in records[s]
                ]
                for s in SPLITS
            },
        }
        self.dataset_sha256 = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()
        if self.subtract_train_region_mean:
            reference, train_ids = _build_train_region_reference(manifest)
            _require(
                reference["nside"] == geometry["nside"]
                and reference["ordering"] == geometry["healpix_ordering"],
                "train reference differs from native geometry",
            )
            split_ids = geometry["split_ids"].numpy()
            _require(
                bool(np.isin(native_ids[split_ids == 0], train_ids).all())
                and not bool(np.isin(native_ids[split_ids != 0], train_ids).any()),
                "train reference conflicts with spatial splits",
            )
            self._expected_train_region = reference
            self._expected_source_ids = {
                r.source_file: (r.cosmology_id, r.permutation_id)
                for r in records["train"]
            }
            del train_ids
            if self._restored_state is None:
                if not self.train_region_stats_file.is_file():
                    raise FileNotFoundError(
                        f"{self.train_region_stats_file}: prepare statistics with "
                        "python -B -m maps_to_cosmology.datamodule_cosmogrid_native "
                        f"--prepare-train-region-means {self.data_dir}"
                    )
                self._set_train_region_stats(
                    _load_batch(self.train_region_stats_file, mmap=False)
                )
        if self.standardize_params:
            self.param_mean, self.param_std = _train_statistics(
                records["train"], self.eps
            )
        if self._restored_state is not None:
            self._restore_statistics(self._restored_state)
        options = dict(
            param_mean=self.param_mean,
            param_std=self.param_std,
            subtract_full_sky_mean=self.subtract_full_sky_mean,
            smoothing_sigma=self.smoothing_sigma,
            cache_size=self.cache_size,
            mmap=self.mmap,
            train_region_means=(
                self.train_region_stats["means"]
                if self.subtract_train_region_mean
                else None
            ),
        )
        self.train_dataset = NativeCosmoGridDataset(
            records["train"], augment=self.augment, **options
        )
        self.val_dataset = NativeCosmoGridDataset(records["val"], **options)
        self.test_dataset = NativeCosmoGridDataset(records["test"], **options)

    def _loader(self, split, shuffle):
        dataset = getattr(self, f"{split}_dataset")
        if dataset is None:
            raise RuntimeError("call setup() before requesting a DataLoader")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            generator=torch.Generator().manual_seed(self.seed + SPLITS.index(split)),
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)

    def _preprocessing_state(self):
        return {
            "standardize_params": self.standardize_params,
            "subtract_full_sky_mean": self.subtract_full_sky_mean,
            "subtract_train_region_mean": self.subtract_train_region_mean,
            "smoothing_sigma": self.smoothing_sigma,
            "augment": self.augment,
        }

    def state_dict(self):
        return {
            "geometry_sha256": self.geometry_sha256,
            "dataset_sha256": self.dataset_sha256,
            "preprocessing": self._preprocessing_state(),
            "param_mean": self.param_mean,
            "param_std": self.param_std,
            "train_region_stats": self.train_region_stats,
        }

    def _set_train_region_stats(self, stats):
        self.train_region_stats = _validate_train_region_stats(
            stats, self._expected_train_region, self._expected_source_ids
        )

    def _restore_statistics(self, state):
        preprocessing = dict(state.get("preprocessing", {}))
        preprocessing.setdefault("subtract_train_region_mean", False)
        _require(
            preprocessing == self._preprocessing_state(),
            "checkpoint preprocessing differs from datamodule settings",
        )
        _require(
            state.get("geometry_sha256") == self.geometry_sha256
            and state.get("dataset_sha256") == self.dataset_sha256,
            "checkpoint geometry/dataset differs from current data",
        )
        if self.subtract_train_region_mean:
            self._set_train_region_stats(state.get("train_region_stats"))
        if self.standardize_params:
            mean, std = state.get("param_mean"), state.get("param_std")
            _require(
                isinstance(mean, torch.Tensor)
                and isinstance(std, torch.Tensor)
                and mean.shape == std.shape == (len(PARAM_NAMES),)
                and bool(torch.isfinite(mean).all())
                and bool(torch.isfinite(std).all())
                and bool((std > 0).all()),
                "invalid checkpoint parameter statistics",
            )
            self.param_mean = mean.detach().cpu().float().clone()
            self.param_std = std.detach().cpu().float().clone()
        for split in SPLITS:
            dataset = getattr(self, f"{split}_dataset")
            if dataset is not None:
                dataset.param_mean, dataset.param_std = self.param_mean, self.param_std
                dataset.train_region_means = (
                    self.train_region_stats["means"]
                    if self.subtract_train_region_mean
                    else None
                )

    def load_state_dict(self, state_dict):
        self._restored_state = state_dict
        if self.train_dataset is not None:
            self._restore_statistics(state_dict)

    def teardown(self, stage: str | None = None):
        for split in SPLITS:
            dataset = getattr(self, f"{split}_dataset")
            if dataset is not None:
                dataset.clear_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare fixed train-region means")
    parser.add_argument(
        "--prepare-train-region-means", metavar="DATA_DIR", required=True
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    prepare_train_region_stats(args.prepare_train_region_means, args.output)
