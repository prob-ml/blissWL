import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning import LightningModule

from maps_to_cosmology.metrics import (
    PearsonCorrelationCoefficient,
    RootMeanSquaredError,
    ScatterPlot,
)
from maps_to_cosmology.networks import ResNet


class TimeEncoder(nn.Module):
    """MLP mapping scalar t in [0,1] to vector (B, t_embed_dim)."""

    def __init__(self, t_embed_dim: int):
        super().__init__()
        # TODO: Build an MLP that maps (B, 1) -> (B, t_embed_dim).
        # Suggested: Linear(1, t_embed_dim // 2) -> SiLU -> Linear(t_embed_dim // 2, t_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(1, t_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(t_embed_dim // 2, t_embed_dim),
        )

    def forward(self, t):
        # Input:  t of shape (B, 1)
        # Output: t_embedding of shape (B, t_embed_dim)
        return self.net(t)


class VelocityNet(nn.Module):
    """MLP predicting 6D velocity from concat(z_t, x_embed, t_embed)."""

    def __init__(
        self, num_cosmo_params: int, x_embed_dim: int, t_embed_dim: int, hidden_dim: int
    ):
        super().__init__()
        # TODO: Build an MLP that maps (B, num_cosmo_params + x_embed_dim + t_embed_dim) -> (B, num_cosmo_params).
        # Suggested: concat inputs, then alternate Linear and SiLU layers (ending with Linear)
        in_dim = num_cosmo_params + x_embed_dim + t_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_cosmo_params),
        )

    def forward(self, zt, x_embedding, t_embedding):
        # Inputs:
        #   zt          — (B, num_cosmo_params)  e.g. (B, 6), the interpolated state
        #   x_embedding — (B, x_embed_dim), map encoding
        #   t_embedding — (B, t_embed_dim), time encoding
        # Output: predicted velocity of shape (B, num_cosmo_params)
        return self.net(torch.cat([zt, x_embedding, t_embedding], dim=-1))


class FlowMatching(LightningModule):
    """Flow matching model for cosmological parameter inference."""

    def __init__(
        self,
        num_bins: int,
        x_embed_dim: int,
        t_embed_dim: int,
        velocity_hidden_dim: int,
        num_cosmo_params: int,
        lr: float,
        num_ode_steps: int = 20,
        num_samples_for_mode: int = 10,
        ode_method: str = "midpoint",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_cosmo_params = num_cosmo_params
        self.num_ode_steps = num_ode_steps
        self.num_samples_for_mode = num_samples_for_mode
        self.ode_method = ode_method
        self.param_names = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]

        self.map_encoder = ResNet(
            num_bins=num_bins,
            output_dim=x_embed_dim,
        )
        self.time_encoder = TimeEncoder(t_embed_dim=t_embed_dim)
        self.velocity_net = VelocityNet(
            num_cosmo_params=num_cosmo_params,
            x_embed_dim=x_embed_dim,
            t_embed_dim=t_embed_dim,
            hidden_dim=velocity_hidden_dim,
        )

        self.val_rmse = RootMeanSquaredError(self.param_names)
        self.val_scatter = ScatterPlot()
        self.val_pcc = PearsonCorrelationCoefficient(self.param_names)
        self.test_rmse = RootMeanSquaredError(self.param_names)
        self.test_scatter = ScatterPlot()
        self.test_pcc = PearsonCorrelationCoefficient(self.param_names)

    def compute_loss(self, maps, params):
        """Compute flow matching loss: MSE between true and predicted velocity."""
        # 1. Encode maps
        x_embedding = self.map_encoder(maps)
        # 2. Sample time
        t = torch.rand(maps.shape[0], 1, device=maps.device)
        # 3. Encode time
        t_embedding = self.time_encoder(t)
        # 4. Set z1 = params and sample z0
        z1 = params
        z0 = torch.randn_like(z1)
        # 5. Define zt as interpolation of z1 and z0
        zt = (1 - t) * z0 + t * z1
        # 6. Define true velocity
        true_velocity = z1 - z0
        # 7. Predict velocity with self.velocity_net
        predicted_velocity = self.velocity_net(zt, x_embedding, t_embedding)
        # 8. Return MSE
        return nn.functional.mse_loss(predicted_velocity, true_velocity)

    def _sample_path(self, zt, x_embedding, t, t_next):
        """One ODE step from t to t_next."""
        # Compute dt = t_next - t, encode t, predict velocity at (zt, x_embedding, t).
        # Use Euler method or Midpoint method to update zt
        # Return updated z.  Use self.ode_method to determine which integrator to use.
        dt = t_next - t
        t_embedding = self.time_encoder(t)
        velocity = self.velocity_net(zt, x_embedding, t_embedding)
        if self.ode_method == "euler":
            z_next = zt + velocity * dt
            return z_next
        elif self.ode_method == "midpoint":
            z_mid = zt + velocity * (dt / 2)
            t_mid = t + dt / 2
            t_mid_embedding = self.time_encoder(t_mid)
            velocity_mid = self.velocity_net(z_mid, x_embedding, t_mid_embedding)
            z_next = zt + velocity_mid * dt
            return z_next
        else:
            raise ValueError(f"Unknown ODE method: {self.ode_method}")

    def _integrate_ode(self, x_embedding):
        """Integrate ODE from t=0 (noise) to t=1 (data)."""
        # 1. Sample z ~ N(0, I) of shape (B, num_cosmo_params) at t=0.
        z = torch.randn(
            x_embedding.shape[0], self.num_cosmo_params, device=x_embedding.device
        )
        # 2. Compute dt = 1 / self.num_ode_steps.
        dt = 1 / self.num_ode_steps
        # 3. For each step i in range(self.num_ode_steps):
        #      t      = i * dt       (broadcast to (B, 1))
        #      t_next = (i + 1) * dt
        #      z = self._sample_path(z, x_embedding, t, t_next)
        for i in range(self.num_ode_steps):
            t = torch.full((x_embedding.shape[0], 1), i * dt, device=x_embedding.device)
            t_next = torch.full(
                (x_embedding.shape[0], 1), (i + 1) * dt, device=x_embedding.device
            )
            z = self._sample_path(z, x_embedding, t, t_next)
        # 4. Return z at t=1, shape (B, num_cosmo_params).
        return z

    def predict(self, maps, use_mode=False):
        """Generate predictions via ODE integration.

        Args:
            maps: Convergence maps (B, num_bins, H, W)
            use_mode: If True, average multiple samples

        Returns:
            Predicted cosmological parameters (B, 6)
        """
        x_embedding = self.map_encoder(maps)
        n_samples = self.num_samples_for_mode if use_mode else 1
        samples = [self._integrate_ode(x_embedding) for _ in range(n_samples)]
        return torch.stack(samples).mean(0) if use_mode else samples[0]

    def training_step(self, batch, batch_idx):
        maps, params = batch
        loss = self.compute_loss(maps, params)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        maps, params = batch
        loss = self.compute_loss(maps, params)
        self.log("val_loss", loss, prog_bar=True)

        # ODE prediction for metrics
        preds = self.predict(maps, use_mode=True)
        self.val_rmse.update(preds, params)
        self.val_scatter.update(preds, params)
        self.val_pcc.update(preds, params)
        return loss

    def on_validation_epoch_end(self):
        rmse = self.val_rmse.compute()
        for name, value in rmse.items():
            self.log(f"val/rmse/{name}", value)
        pcc = self.val_pcc.compute()
        for name, value in pcc.items():
            self.log(f"val/pcc/{name}", value)

        for i, name in enumerate(self.param_names):
            fig = self.val_scatter.create_param_scatter(i, name)
            self.logger.experiment.add_figure(
                f"val/scatter/{name}", fig, self.current_epoch
            )
            plt.close(fig)

        self.val_rmse.reset()
        self.val_scatter.reset()
        self.val_pcc.reset()

    def test_step(self, batch, batch_idx):
        maps, params = batch
        loss = self.compute_loss(maps, params)
        self.log("test_loss", loss, prog_bar=True)

        preds = self.predict(maps, use_mode=True)
        self.test_rmse.update(preds, params)
        self.test_scatter.update(preds, params)
        self.test_pcc.update(preds, params)
        return loss

    def on_test_epoch_end(self):
        rmse = self.test_rmse.compute()
        for name, value in rmse.items():
            self.log(f"test/rmse/{name}", value)
        pcc = self.test_pcc.compute()
        for name, value in pcc.items():
            self.log(f"test/pcc/{name}", value)

        for i, name in enumerate(self.param_names):
            fig = self.test_scatter.create_param_scatter(i, name)
            self.logger.experiment.add_figure(
                f"test/scatter/{name}", fig, self.current_epoch
            )
            plt.close(fig)

        self.test_rmse.reset()
        self.test_scatter.reset()
        self.test_pcc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
