import matplotlib.pyplot as plt

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig

from maps_to_cosmology.datamodule import ConvergenceMapsModule
from maps_to_cosmology.encoder import Encoder


class SaveBestScatterplot(Callback):
    """Save omega_c scatterplot when a new best val_loss is achieved."""

    def __init__(self, dirpath: str):
        super().__init__()
        self.dirpath = dirpath
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            fig = pl_module.val_scatter.create_omega_c_scatter()
            fig.savefig(f"{self.dirpath}/best_omega_c_scatter.png", dpi=150)
            plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="train_npe")
def main(cfg: DictConfig) -> None:
    """Train the neural posterior estimation model."""
    seed_everything(cfg.seed, workers=True)

    # Create data module
    datamodule = ConvergenceMapsModule(
        data_dir=cfg.paths.data_dir,
        batch_size=cfg.convergence_maps.batch_size,
        num_workers=cfg.convergence_maps.num_workers,
        val_split=cfg.convergence_maps.val_split,
        test_split=cfg.convergence_maps.test_split,
        seed=cfg.seed,
    )

    # Create model
    model = Encoder(
        hidden_dim=cfg.encoder.hidden_dim,
        num_cosmo_params=cfg.encoder.num_cosmo_params,
        lr=cfg.encoder.lr,
    )

    # Create logger
    logger = instantiate(cfg.train.logger)

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logger.log_dir}/checkpoints",
        filename=cfg.train.callbacks.checkpoint.filename,
        monitor=cfg.train.callbacks.checkpoint.monitor,
        mode=cfg.train.callbacks.checkpoint.mode,
        save_top_k=cfg.train.callbacks.checkpoint.save_top_k,
    )

    early_stopping_callback = EarlyStopping(
        monitor=cfg.train.callbacks.early_stopping.monitor,
        patience=cfg.train.callbacks.early_stopping.patience,
        mode=cfg.train.callbacks.early_stopping.mode,
    )

    scatterplot_callback = SaveBestScatterplot(dirpath=logger.log_dir)

    # Create trainer
    trainer = Trainer(
        max_epochs=cfg.train.trainer.max_epochs,
        accelerator=cfg.train.trainer.accelerator,
        devices=cfg.train.trainer.devices,
        log_every_n_steps=cfg.train.trainer.log_every_n_steps,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, scatterplot_callback],
    )

    # Train
    trainer.fit(model, datamodule)

    print(f"\nBest model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
