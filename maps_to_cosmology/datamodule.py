from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from einops import rearrange
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm


class ConvergenceMapsDataset(Dataset):
    """Dataset for convergence maps and cosmological parameters."""

    def __init__(self, maps: torch.Tensor, params: torch.Tensor):
        """Initialize dataset.

        Args:
            maps: Convergence maps [N, 256, 256, 5]
            params: Cosmological parameters [N, 6]
        """
        self.maps = maps
        self.params = params

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Transpose from [256, 256, 5] to [5, 256, 256] for PyTorch conv layers
        maps = rearrange(self.maps[idx], "h w b -> b h w")
        params = self.params[idx]
        return maps, params


class ConvergenceMapsModule(LightningDataModule):
    """Lightning DataModule for convergence maps."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        """Initialize data module.

        Args:
            data_dir: Directory containing batch_*.pt files
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            seed: Random seed for train/val/test split
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):  # noqa: ARG002
        """Load data and create train/val/test splits."""
        if self.train_dataset is not None:
            return  # Already set up

        # Check for combined file first (fast path)
        combined_path = self.data_dir / "combined_batches.pt"
        if combined_path.exists():
            print(f"Loading combined_batches.pt from {self.data_dir}")
            data = torch.load(combined_path, weights_only=True)
            all_maps = data["maps"]
            all_params = data["params"]
        else:
            # Fall back to loading individual batch files (slow path)
            pt_files = sorted(self.data_dir.glob("batch_*.pt"))
            if not pt_files:
                raise FileNotFoundError(
                    f"No combined_batches.pt or batch_*.pt files found in {self.data_dir}"
                )

            # Load files in parallel
            def load_file(filepath):
                batch = torch.load(filepath, weights_only=True)
                return batch["maps"], batch["params"]

            all_maps_list = []
            all_params_list = []

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    tqdm(
                        executor.map(load_file, pt_files),
                        total=len(pt_files),
                        desc="Loading data",
                    )
                )

            for maps, params in results:
                all_maps_list.append(maps)
                all_params_list.append(params)

            # Concatenate all batches
            all_maps = torch.cat(all_maps_list, dim=0)
            all_params = torch.cat(all_params_list, dim=0)

        print(f"Loaded {len(all_maps)} samples")
        print(f"Maps shape: {all_maps.shape}")
        print(f"Params shape: {all_params.shape}")

        # Create full dataset and split into train/val/test
        full_dataset = ConvergenceMapsDataset(all_maps, all_params)
        n_total = len(full_dataset)
        n_val = int(n_total * self.val_split)
        n_test = int(n_total * self.test_split)
        n_train = n_total - n_val - n_test

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed),
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
