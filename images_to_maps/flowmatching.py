from pathlib import Path
from typing import Optional

import lightning as L
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from torchmetrics import MetricCollection

from images_to_maps.convnet import MassMapNet


class TimeEncoder(L.LightningModule):
    def __init__(self, t_embed_dim: int, expand_dim: int):
        super().__init__()
        self.expand_dim = expand_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, t_embed_dim // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(t_embed_dim // 2, t_embed_dim),
        )

    def forward(self, t):
        return rearrange(self.net(t), "b d -> b d 1 1").expand(
            -1, -1, self.expand_dim, self.expand_dim
        )


class VelocityNet(L.LightningModule):
    def __init__(self, z_dim: int, x_embed_dim: int, t_embed_dim: int, channels: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                z_dim + x_embed_dim + t_embed_dim, channels, kernel_size=3, padding=1
            ),
            torch.nn.SiLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(channels, z_dim, kernel_size=3, padding=1),
        )

    def forward(self, zt, x_embedding, t_embedding):
        inputs = torch.cat([zt, x_embedding, t_embedding], dim=1)
        return self.net(inputs)


class FlowMatching(L.LightningModule):
    def __init__(
        self,
        n_bands: int,
        res_init: int,
        res_midpoint: int,
        res_final: int,
        ch_init: int,
        ch_max: int,
        ch_final: int,
        initial_downsample: bool,
        more_up_layers: bool,
        num_bottleneck_layers: int,
        image_normalizers: list,
        t_embed_dim: int,
        num_redshift_bins: int,
        velo_net_channels: int,
        scale_factor: float = 1.0,
        optimizer_params: Optional[dict] = None,
        mode_metrics: Optional[MetricCollection] = None,
        sample_metrics: Optional[MetricCollection] = None,
        sample_image_renders: Optional[MetricCollection] = None,
        num_samples_for_mode: int = 10,
        num_ode_steps: int = 10,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.res_init = res_init
        self.res_midpoint = res_midpoint
        self.res_final = res_final
        self.ch_init = ch_init
        self.ch_max = ch_max
        self.ch_final = ch_final
        self.initial_downsample = initial_downsample
        self.more_up_layers = more_up_layers
        self.num_bottleneck_layers = num_bottleneck_layers
        self.image_normalizers = torch.nn.ModuleList(image_normalizers.values())
        self.t_embed_dim = t_embed_dim
        self.num_redshift_bins = num_redshift_bins
        self.z_dim = 3 * num_redshift_bins  # shear1, shear2, convergence for each bin
        self.velo_net_channels = velo_net_channels
        self.scale_factor = scale_factor
        self.optimizer_params = optimizer_params
        self.mode_metrics = mode_metrics
        self.sample_metrics = sample_metrics
        self.sample_image_renders = sample_image_renders
        self.num_samples_for_mode = num_samples_for_mode
        self.num_ode_steps = num_ode_steps

        self.initialize_networks()

        # Epoch loss tracking
        self.epoch_train_losses = []
        self.current_epoch_train_loss = 0.0
        self.current_epoch_train_batches = 0
        self.epoch_val_losses = []
        self.current_epoch_val_loss = 0.0
        self.current_epoch_val_batches = 0
        self.best_val_loss = float("inf")

    def initialize_networks(self):
        ch_per_band = sum(
            inorm.num_channels_per_band() for inorm in self.image_normalizers
        )
        self.image_encoder = MassMapNet(
            n_bands=self.n_bands,
            res_init=self.res_init,
            res_midpoint=self.res_midpoint,
            res_final=self.res_final,
            ch_per_band=ch_per_band,
            ch_init=self.ch_init,
            ch_max=self.ch_max,
            ch_final=self.ch_final,
            initial_downsample=self.initial_downsample,
            more_up_layers=self.more_up_layers,
            num_bottleneck_layers=self.num_bottleneck_layers,
            map_to_var_params=False,
        )
        self.time_encoder = TimeEncoder(
            t_embed_dim=self.t_embed_dim, expand_dim=self.res_final
        )
        self.velocity_net = VelocityNet(
            z_dim=self.z_dim,
            x_embed_dim=self.ch_final,
            t_embed_dim=self.t_embed_dim,
            channels=self.velo_net_channels,
        )

    def compute_loss(self, batch):
        x_list = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        x = torch.cat(x_list, dim=2)
        x_embedding = self.image_encoder(x)

        t = torch.rand([x.shape[0], 1], device=x.device)
        t_embedding = self.time_encoder(t)

        shear1 = self.scale_factor * rearrange(
            batch["tile_catalog"]["shear_1"], "b h w r -> b r h w"
        )
        shear2 = self.scale_factor * rearrange(
            batch["tile_catalog"]["shear_2"], "b h w r -> b r h w"
        )
        convergence = self.scale_factor * rearrange(
            batch["tile_catalog"]["convergence"], "b h w r -> b r h w"
        )
        z1 = torch.cat([shear1, shear2, convergence], dim=1)

        z0 = torch.randn_like(z1, device=z1.device)

        zt = (1 - t) * z0 + t * z1

        velo_true = z1 - z0

        velo_pred = self.velocity_net(zt, x_embedding, t_embedding)

        return ((velo_true - velo_pred) ** 2).mean()

    def sample_path(self, zt, x_embedding, t, t_next, method="midpoint"):
        t_embedding = self.time_encoder(t)
        velo_t = self.velocity_net(zt, x_embedding, t_embedding)

        t_diff = t_next.unique() - t.unique()

        if method == "midpoint":
            half_step = 0.5 * t_diff
            z_mid = zt + half_step * velo_t
            t_mid_embedding = self.time_encoder(t + half_step)
            u_mid = self.velocity_net(z_mid, x_embedding, t_mid_embedding)
            return zt + t_diff * u_mid
        if method == "euler":
            return zt + t_diff * velo_t
        raise ValueError("method should be euler or midpoint")

    def sample(self, batch, use_mode=False):
        """Generate samples via ODE integration.

        Args:
            batch: Input batch with images
            use_mode: If True, return mean of multiple samples

        Returns:
            Dict with 'shear_1', 'shear_2', 'convergence' keys
        """
        x_list = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        x = torch.cat(x_list, dim=2)
        x_embedding = self.image_encoder(x)

        n_samples = self.num_samples_for_mode if use_mode else 1
        samples = []
        for _ in range(n_samples):
            z = self._integrate_ode(x_embedding)
            samples.append(z)

        z_final = torch.stack(samples).mean(0) if use_mode else samples[0]
        return self._z_to_catalog(z_final)

    def _integrate_ode(self, x_embedding):
        """Integrate ODE from t=0 (noise) to t=1 (data)."""
        batch_size = x_embedding.shape[0]
        h, w = x_embedding.shape[2], x_embedding.shape[3]

        z = torch.randn(batch_size, self.z_dim, h, w, device=x_embedding.device)
        dt = 1.0 / self.num_ode_steps

        for i in range(self.num_ode_steps):
            t = torch.full((batch_size, 1), i * dt, device=z.device)
            t_next = torch.full((batch_size, 1), (i + 1) * dt, device=z.device)
            z = self.sample_path(z, x_embedding, t, t_next, method="midpoint")

        return z / self.scale_factor

    def _z_to_catalog(self, z):
        """Convert z tensor (b, z_dim, h, w) to catalog dict."""
        n_bins = self.num_redshift_bins
        shear1 = z[:, :n_bins, :, :]
        shear2 = z[:, n_bins : 2 * n_bins, :, :]
        convergence = z[:, 2 * n_bins :, :, :]

        return {
            "shear_1": rearrange(shear1, "b r h w -> b h w r"),
            "shear_2": rearrange(shear2, "b r h w -> b h w r"),
            "convergence": rearrange(convergence, "b r h w -> b h w r"),
        }

    def update_metrics(self, batch, batch_idx):
        """Update metrics with current batch predictions."""
        if self.mode_metrics is None:
            return

        target_cat = batch["tile_catalog"]

        mode_cat = self.sample(batch, use_mode=True)
        self.mode_metrics.update(target_cat, mode_cat, None)

        if self.sample_metrics is not None:
            sample_cat = self.sample(batch, use_mode=False)
            self.sample_metrics.update(target_cat, sample_cat, None)

        if self.sample_image_renders is not None:
            self.sample_image_renders.update(
                target_cat, mode_cat, self.current_epoch, batch_idx
            )

    def report_metrics(self, metrics, logging_name):
        """Report metrics to logger."""
        computed = metrics.compute()
        for k, v in computed.items():
            if torch.is_tensor(v) and v.numel() > 1:
                for i in range(v.numel()):
                    self.log(f"{logging_name}/{k}/bin_{i}", v[i].item(), sync_dist=True)
            else:
                self.log(
                    f"{logging_name}/{k}",
                    v.item() if torch.is_tensor(v) else v,
                    sync_dist=True,
                )

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log(
            "train_loss", loss, batch_size=batch["images"].shape[0], sync_dist=True
        )
        self.current_epoch_train_loss += loss.item()
        self.current_epoch_train_batches += 1
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, batch_size=batch["images"].shape[0], sync_dist=True)
        self.current_epoch_val_loss += loss.item()
        self.current_epoch_val_batches += 1
        self.update_metrics(batch, batch_idx)
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch_train_batches > 0:
            avg_train_loss = (
                self.current_epoch_train_loss / self.current_epoch_train_batches
            )
            self.epoch_train_losses.append(avg_train_loss)
            print(f"Epoch {self.current_epoch}: avg train loss = {avg_train_loss:.6f}")
        self.current_epoch_train_loss = 0.0
        self.current_epoch_train_batches = 0

    def on_validation_epoch_end(self):
        avg_val_loss = None
        if self.current_epoch_val_batches > 0:
            avg_val_loss = self.current_epoch_val_loss / self.current_epoch_val_batches
            self.epoch_val_losses.append(avg_val_loss)
            print(f"Epoch {self.current_epoch}: avg val loss = {avg_val_loss:.6f}")
        self.current_epoch_val_loss = 0.0
        self.current_epoch_val_batches = 0

        if self.mode_metrics is not None:
            self.report_metrics(self.mode_metrics, "val/mode")
            self.mode_metrics.reset()
        if self.sample_metrics is not None:
            self.report_metrics(self.sample_metrics, "val/sample")
            self.sample_metrics.reset()
        if self.sample_image_renders is not None:
            # Only save plots if this is the best val loss so far
            if avg_val_loss is not None and avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                for metric in self.sample_image_renders.values():
                    metric.plot()
            self.sample_image_renders.reset()

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("test_loss", loss, batch_size=batch["images"].shape[0], sync_dist=True)
        self.update_metrics(batch, batch_idx)
        return loss

    def on_test_epoch_end(self):
        if self.mode_metrics is not None:
            self.report_metrics(self.mode_metrics, "test/mode")
            self.mode_metrics.reset()
        if self.sample_metrics is not None:
            self.report_metrics(self.sample_metrics, "test/sample")
            self.sample_metrics.reset()

    def on_train_end(self):
        # Get save location from the plots metric
        save_local = None
        if self.sample_image_renders is not None:
            for metric in self.sample_image_renders.values():
                if hasattr(metric, "save_local") and metric.save_local:
                    save_local = metric.save_local
                    break

        if save_local is None:
            print("No save_local found, skipping loss plots")
            return

        # Create directory if needed
        save_path = Path(save_local)
        save_path.mkdir(parents=True, exist_ok=True)

        # Plot both on same axes
        if self.epoch_train_losses and self.epoch_val_losses:
            fig, ax = plt.subplots()
            ax.plot(self.epoch_train_losses, label="Train")
            ax.plot(self.epoch_val_losses, label="Validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training and Validation Loss")
            ax.legend()
            fig.savefig(save_path / "loss_curves.png")
            plt.close(fig)

        print(f"Loss plots saved to {save_path}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        return optimizer
