from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from einops import rearrange
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur
from tqdm.auto import tqdm

from maps_to_cosmology.datamodule import ConvergenceMapsDataset


class CosmoGridDC2MapsModule(LightningDataModule):
    def __init__(
        self,
        cosmogrid_data_dir: str,
        dc2_data_dir: str,
        train_dc2_per_cosmogrid: float = 0.5,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        standardize_params: bool = True,
        eps: float = 1e-8,
        smoothing_sigma: float | None = None,
        augment: bool = False,
    ):
        super().__init__()
        self.cosmogrid_data_dir = Path(cosmogrid_data_dir)
        self.dc2_data_dir = Path(dc2_data_dir)
        self.train_dc2_per_cosmogrid = train_dc2_per_cosmogrid
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.standardize_params = standardize_params
        self.eps = eps
        self.smoothing_sigma = smoothing_sigma
        self.augment = augment

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.param_mean = None
        self.param_std = None

    def _load_maps_and_params(self, data_dir: Path):
        combined_path = data_dir / "combined_batches.pt"
        if combined_path.exists():
            data = torch.load(combined_path, weights_only=True)
            return data

        pt_files = sorted(data_dir.glob("batch_*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No data files found in {data_dir}")

        def load_file(path):
            batch = torch.load(path, weights_only=True)
            return batch["maps"], batch["params"]

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(load_file, pt_files), total=len(pt_files)))

        maps = torch.cat([m for m, _ in results], dim=0)
        params = torch.cat([p for _, p in results], dim=0)

        torch.save({"maps": maps, "params": params}, combined_path)
        return maps, params

    def _smooth(self, maps):
        if self.smoothing_sigma is None:
            return maps

        sigma = self.smoothing_sigma
        kernel_size = 2 * round(2 * sigma) + 1
        maps = rearrange(maps, "n h w c -> n c h w")
        maps = gaussian_blur(
            maps,
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma, sigma],
        )
        return rearrange(maps, "n c h w -> n h w c")
    
    def _augment_training_maps(self, maps, params):
        aug_maps = [
            maps,
            torch.flip(maps, dims=[1]),
            torch.flip(maps, dims=[2]),
            torch.rot90(maps, k=2, dims=[1, 2]),
        ]
        return torch.cat(aug_maps, dim=0), params.repeat(len(aug_maps), 1)

    def setup(self, stage: str | None = None):
        if self.train_dataset is not None:
            return

        cosmo_data = self._load_maps_and_params(self.cosmogrid_data_dir)
        dc2_data = self._load_maps_and_params(self.dc2_data_dir)

        cosmo_maps = cosmo_data["maps"]
        cosmo_params = cosmo_data["params"]
        cosmology_ids = cosmo_data["cosmology_ids"]

        dc2_maps = dc2_data["maps"][..., :4]
        dc2_params = dc2_data["params"]

        cosmology_ids = cosmo_data["cosmology_ids"].to(torch.long)

        unique_cosmos = torch.unique(cosmology_ids)
        n_total_cosmos = len(unique_cosmos)
        n_val_cosmos = int(n_total_cosmos * self.val_split)
        n_test_cosmos = int(n_total_cosmos * self.test_split)
        n_train_cosmos = n_total_cosmos - n_val_cosmos - n_test_cosmos

        generator = torch.Generator().manual_seed(self.seed)
        cosmo_perm = unique_cosmos[torch.randperm(n_total_cosmos, generator=generator)]

        train_cosmos = cosmo_perm[:n_train_cosmos]
        val_cosmos = cosmo_perm[n_train_cosmos : n_train_cosmos + n_val_cosmos]
        test_cosmos = cosmo_perm[n_train_cosmos + n_val_cosmos :]

        train_idx = torch.nonzero(torch.isin(cosmology_ids, train_cosmos), as_tuple=True)[0]
        val_idx = torch.nonzero(torch.isin(cosmology_ids, val_cosmos), as_tuple=True)[0]
        test_idx = torch.nonzero(torch.isin(cosmology_ids, test_cosmos), as_tuple=True)[0]

        train_maps = cosmo_maps[train_idx]
        train_params = cosmo_params[train_idx]

        val_maps = cosmo_maps[val_idx]
        val_params = cosmo_params[val_idx]

        test_maps = cosmo_maps[test_idx]
        test_params = cosmo_params[test_idx]

        n_dc2_train = int(round(self.train_dc2_per_cosmogrid * len(train_idx)))
        if n_dc2_train > len(dc2_maps):
            raise ValueError(
                f"Need {n_dc2_train} DC2 maps, but only {len(dc2_maps)} available."
            )

        dc2_idx = torch.randperm(len(dc2_maps), generator=generator)[:n_dc2_train]

        train_maps = torch.cat([train_maps, dc2_maps[dc2_idx]], dim=0)
        train_params = torch.cat([train_params, dc2_params[dc2_idx]], dim=0)

        train_maps = self._smooth(train_maps)
        val_maps = self._smooth(val_maps)
        test_maps = self._smooth(test_maps)

        if self.augment:
            train_maps, train_params = self._augment_training_maps(train_maps, train_params)

        shuffle_idx = torch.randperm(len(train_maps), generator=generator)
        train_maps = train_maps[shuffle_idx]
        train_params = train_params[shuffle_idx]

        if self.standardize_params:
            self.param_mean = train_params.mean(dim=0)
            self.param_std = train_params.std(dim=0, unbiased=False).clamp_min(self.eps)

            train_params = (train_params - self.param_mean) / self.param_std
            val_params = (val_params - self.param_mean) / self.param_std
            test_params = (test_params - self.param_mean) / self.param_std

        self.train_dataset = ConvergenceMapsDataset(train_maps, train_params)
        self.val_dataset = ConvergenceMapsDataset(val_maps, val_params)
        self.test_dataset = ConvergenceMapsDataset(test_maps, test_params)

        print(f"CosmoGrid train cosmologies: {n_train_cosmos}")
        print(f"CosmoGrid train maps: {len(train_idx)}")
        print(f"DC2 train: {n_dc2_train}")
        print(f"Train total: {len(self.train_dataset)}")
        print(f"Val: {len(self.val_dataset)} CosmoGrid only")
        print(f"Test: {len(self.test_dataset)} CosmoGrid only")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )