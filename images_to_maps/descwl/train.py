"""Train ScalarShearEncoder for descwl-shear-sims weak lensing inference.

Usage:
    python -m images_to_maps.descwl.train
"""

import torch
import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=".", config_name="config_train_npe")
def main(cfg: DictConfig) -> None:
    """Train the ScalarShearEncoder model."""
    seed_everything(cfg.train.seed, workers=True)

    # Instantiate data module and encoder from config
    data_module = instantiate(cfg.train.data_source)
    encoder = instantiate(cfg.train.encoder)
    callbacks = list(instantiate(cfg.train.callbacks).values())

    # Load pretrained weights if specified
    if cfg.train.pretrained_weights is not None:
        print(f"Loading pretrained weights from: {cfg.train.pretrained_weights}")
        checkpoint = torch.load(cfg.train.pretrained_weights, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        encoder.load_state_dict(state_dict, strict=True)

    # Create trainer
    trainer = instantiate(cfg.train.trainer, callbacks=callbacks)

    # Train (ckpt_path resumes full training state including optimizer)
    trainer.fit(encoder, datamodule=data_module, ckpt_path=cfg.train.ckpt_path)


if __name__ == "__main__":
    main()
