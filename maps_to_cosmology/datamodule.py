from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
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
        maps = self.maps[idx].permute(2, 0, 1)
        params = self.params[idx]
        return maps, params


class ConvergenceMapsModule(LightningDataModule):
    """Lightning DataModule for convergence maps."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        train_val_split: float = 0.9,
    ):
        """Initialize data module.

        Args:
            data_dir: Directory containing batch_*.pt files
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_val_split: Fraction of data to use for training
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None):  # noqa: ARG002
        """Load data and create train/val splits."""
        if self.train_dataset is not None:
            return  # Already set up

        # Find all batch files
        pt_files = sorted(self.data_dir.glob("batch_*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No batch_*.pt files found in {self.data_dir}")

        # Load files in parallel
        def load_file(filepath):
            batch = torch.load(filepath, weights_only=True)
            return batch["maps"], batch["params"]

        all_maps = []
        all_params = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                tqdm(
                    executor.map(load_file, pt_files),
                    total=len(pt_files),
                    desc="Loading data",
                )
            )

        for maps, params in results:
            all_maps.append(maps)
            all_params.append(params)

        # Concatenate all batches
        all_maps = torch.cat(all_maps, dim=0)
        all_params = torch.cat(all_params, dim=0)

        print(f"Loaded {len(all_maps)} samples")
        print(f"Maps shape: {all_maps.shape}")
        print(f"Params shape: {all_params.shape}")

        # Create full dataset and split
        full_dataset = ConvergenceMapsDataset(all_maps, all_params)
        train_size = int(len(full_dataset) * self.train_val_split)
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

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
