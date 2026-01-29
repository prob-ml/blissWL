#!/usr/bin/env python
"""Compute credible intervals for NPE shear predictions.

Usage:
    python compute_npe_credibleintervals.py
    python compute_npe_credibleintervals.py ckpt=/path/to/ckpt cached_data_path=/path/to/data output=out.pt

Configure settings in config_compute_npe_credibleintervals.yaml or override via command line.
"""

import concurrent.futures

import hydra
import lightning as L
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="config_compute_npe_credibleintervals",
)
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Clear global hydra to allow re-initialization for training config
    GlobalHydra.instance().clear()

    # Initialize hydra config for training (to get data source and encoder config)
    with initialize(config_path="..", version_base=None):
        hydra_cfg = compose(
            "config_train_npe",
            overrides=[
                f"train.pretrained_weights={cfg.ckpt}",
                f"paths.cached_data={cfg.cached_data_path}",
            ],
        )

    L.seed_everything(hydra_cfg.train.seed)

    # Setup data source
    print("Setting up data source...")
    data_source = instantiate(hydra_cfg.train.data_source)
    data_source.setup("test")

    # Load encoder
    print("Loading encoder...")
    encoder = instantiate(hydra_cfg.encoder).to(device)
    state_dict = torch.load(cfg.ckpt, map_location=device)["state_dict"]
    encoder.load_state_dict(state_dict)
    encoder = encoder.eval()

    # Confidence levels and quantiles
    confidence_levels = torch.linspace(0.05, 0.95, steps=19)
    ci_quantiles = (
        torch.distributions.Normal(0, 1)
        .icdf(1 - (1 - confidence_levels) / 2)
        .to(device)
    )

    # Get test files
    test_files = data_source.test_dataset.file_paths
    transform = data_source.test_dataset.transform
    print(f"Processing {len(test_files)} files...")

    def load_chunk(chunk_files):
        """Load files in parallel and flatten samples."""

        def load_one(path):
            data_list = torch.load(path, weights_only=False)
            return [transform(s) for s in data_list]

        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_workers) as ex:
            file_data = list(ex.map(load_one, chunk_files))
        return [s for samples in file_data for s in samples]

    # Process chunks
    files_per_chunk = cfg.files_per_chunk
    chunk_lists = [
        test_files[i : i + files_per_chunk]
        for i in range(0, len(test_files), files_per_chunk)
    ]

    results = {
        "shear1_true": [],
        "shear2_true": [],
        "shear1_posteriormean": [],
        "shear2_posteriormean": [],
        "shear1_ci_lower": [],
        "shear1_ci_upper": [],
        "shear2_ci_lower": [],
        "shear2_ci_upper": [],
    }

    with torch.no_grad():
        for chunk_files in tqdm(chunk_lists, desc="Processing"):
            chunk_data = load_chunk(chunk_files)
            if not chunk_data:
                continue

            images = torch.stack([d["images"] for d in chunk_data]).to(device)
            shear1 = torch.tensor([d["tile_catalog"]["shear_1"] for d in chunk_data])
            shear2 = torch.tensor([d["tile_catalog"]["shear_2"] for d in chunk_data])

            # Forward pass
            input_lst = [
                inorm.get_input_tensor({"images": images})
                for inorm in encoder.image_normalizers
            ]
            inputs = torch.cat(input_lst, dim=2).squeeze(2)
            x = encoder.net(inputs).squeeze()  # (batch, 1, 1, 4) -> (batch, 4) or (4,)
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Ensure 2D: (batch, 4)

            # Compute credible intervals
            std1, std2 = x[:, 1].exp().sqrt(), x[:, 3].exp().sqrt()

            results["shear1_true"].append(shear1)
            results["shear2_true"].append(shear2)
            results["shear1_posteriormean"].append(x[:, 0].cpu())
            results["shear2_posteriormean"].append(x[:, 2].cpu())
            results["shear1_ci_lower"].append(
                (x[:, 0:1] - ci_quantiles * std1.unsqueeze(1)).cpu()
            )
            results["shear1_ci_upper"].append(
                (x[:, 0:1] + ci_quantiles * std1.unsqueeze(1)).cpu()
            )
            results["shear2_ci_lower"].append(
                (x[:, 2:3] - ci_quantiles * std2.unsqueeze(1)).cpu()
            )
            results["shear2_ci_upper"].append(
                (x[:, 2:3] + ci_quantiles * std2.unsqueeze(1)).cpu()
            )

            del images, inputs, x, chunk_data

    # Save results
    output = {
        "confidence_levels": confidence_levels,
        "shear1_true": torch.cat(results["shear1_true"]),
        "shear2_true": torch.cat(results["shear2_true"]),
        "shear1_posteriormean": torch.cat(results["shear1_posteriormean"]),
        "shear2_posteriormean": torch.cat(results["shear2_posteriormean"]),
        "shear1_ci_lower": torch.cat(results["shear1_ci_lower"]),
        "shear1_ci_upper": torch.cat(results["shear1_ci_upper"]),
        "shear2_ci_lower": torch.cat(results["shear2_ci_lower"]),
        "shear2_ci_upper": torch.cat(results["shear2_ci_upper"]),
    }

    print(f"Processed {len(output['shear1_true'])} samples")
    torch.save(output, cfg.output)
    print(f"Saved to {cfg.output}")


if __name__ == "__main__":
    main()
