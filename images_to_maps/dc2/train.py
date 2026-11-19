"""Train MassMapEncoder for DC2 weak lensing inference.

Usage:
    python -m images_to_maps.dc2.train
"""

import hydra
import torch
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=".", config_name="config_train_npe")
def main(cfg: DictConfig) -> None:
    """Train the MassMapEncoder model."""
    seed_everything(cfg.train.seed, workers=True)

    matmul_precision = cfg.train.get("matmul_precision")
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)

    # Instantiate data module and encoder from config
    data_module = instantiate(cfg.train.data_source)
    encoder = instantiate(cfg.train.encoder)
    callbacks = list(instantiate(cfg.train.callbacks).values())

    # Create trainer
    trainer = instantiate(cfg.train.trainer, callbacks=callbacks)

    # Train (ckpt_path resumes model, optimizer, epoch and callback state)
    trainer.fit(encoder, datamodule=data_module, ckpt_path=cfg.train.get("ckpt_path"))


if __name__ == "__main__":
    main()
