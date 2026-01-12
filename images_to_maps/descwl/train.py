"""Train ScalarShearEncoder for descwl-shear-sims weak lensing inference.

Usage:
    python -m images_to_maps.descwl.train
"""

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

    # Create trainer
    trainer = instantiate(cfg.train.trainer, callbacks=callbacks)

    # Train
    trainer.fit(encoder, datamodule=data_module)


if __name__ == "__main__":
    main()
