import hydra
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig

from maps_to_cosmology.datamodule import ConvergenceMapsModule
from maps_to_cosmology.encoder import Encoder


@hydra.main(version_base=None, config_path="configs", config_name="train_npe")
def main(cfg: DictConfig) -> None:
    """Train the neural posterior estimation model."""
    # Create data module
    datamodule = ConvergenceMapsModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_val_split=cfg.data.train_val_split,
    )

    # Create model
    model = Encoder(
        hidden_dim=cfg.model.hidden_dim,
        num_params=cfg.model.num_params,
        lr=cfg.model.lr,
    )

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename=cfg.checkpoint.filename,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
    )

    early_stopping_callback = EarlyStopping(
        monitor=cfg.early_stopping.monitor,
        patience=cfg.early_stopping.patience,
        mode=cfg.early_stopping.mode,
    )

    # Create trainer
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # Train
    trainer.fit(model, datamodule)

    print(f"\nBest model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
