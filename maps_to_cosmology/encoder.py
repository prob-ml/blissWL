import matplotlib.pyplot as plt
import torch
from lightning import LightningModule

from maps_to_cosmology.metrics import RootMeanSquaredError, ScatterPlot
from maps_to_cosmology.networks import TwoLayerMLP


class Encoder(LightningModule):
    """Encoder that maps convergence maps to variational posterior parameters.

    Uses a TwoLayerMLP to process convergence maps [B, 5, 256, 256] and output
    [B, 12] tensor with alternating loc/scale parameters for 6 independent
    Normal distributions over cosmological parameters.
    """

    def __init__(
        self,
        num_bins: int,
        map_slen: int,
        hidden_dim: int,
        num_cosmo_params: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_cosmo_params = num_cosmo_params
        self.param_names = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]

        # Metrics (separate instances for val/test)
        self.val_rmse = RootMeanSquaredError(self.param_names)
        self.val_scatter = ScatterPlot()
        self.test_rmse = RootMeanSquaredError(self.param_names)
        self.test_scatter = ScatterPlot()

        self.net = TwoLayerMLP(
            num_bins=num_bins,
            map_slen=map_slen,
            hidden_dim=hidden_dim,
            output_dim=num_cosmo_params * 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Convergence maps [B, 5, 256, 256]

        Returns:
            Variational parameters [B, 12] with alternating loc/scale
        """
        return self.net(x)

    def compute_loss(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            x: Convergence maps [B, 5, 256, 256]
            params: True cosmological parameters [B, 6]

        Returns:
            Mean NLL loss
        """
        out = self.forward(x)  # [B, 12]
        loc = out[:, 0::2]  # [B, 6]
        scale = torch.clamp(out[:, 1::2], -10, 10).exp().sqrt()  # [B, 6]
        dist = torch.distributions.Normal(loc, scale)
        nll = -dist.log_prob(params).sum(dim=-1).mean()
        return nll

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        maps, params = batch
        loss = self.compute_loss(maps, params)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        maps, params = batch
        out = self.forward(maps)
        loc = out[:, 0::2]  # Posterior means [B, 6]

        # Update metrics
        self.val_rmse.update(loc, params)
        self.val_scatter.update(loc, params)

        loss = self.compute_loss(maps, params)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """Compute RMSE and generate scatterplot at end of validation epoch."""
        # Log per-parameter RMSE
        rmse = self.val_rmse.compute()
        for name, value in rmse.items():
            self.log(f"val_rmse_{name}", value)

        # Log scatterplot
        fig = self.val_scatter.create_omega_c_scatter()
        self.logger.experiment.add_figure(
            "val_omega_c_scatter", fig, self.current_epoch
        )
        plt.close(fig)

        # Reset metrics
        self.val_rmse.reset()
        self.val_scatter.reset()

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        maps, params = batch
        out = self.forward(maps)
        loc = out[:, 0::2]  # Posterior means [B, 6]

        # Update metrics
        self.test_rmse.update(loc, params)
        self.test_scatter.update(loc, params)

        loss = self.compute_loss(maps, params)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        """Compute RMSE and generate scatterplot for test set."""
        # Log per-parameter RMSE
        rmse = self.test_rmse.compute()
        for name, value in rmse.items():
            self.log(f"test_rmse_{name}", value)

        # Log scatterplot
        fig = self.test_scatter.create_omega_c_scatter()
        self.logger.experiment.add_figure(
            "test_omega_c_scatter", fig, self.current_epoch
        )
        plt.close(fig)

        # Reset metrics
        self.test_rmse.reset()
        self.test_scatter.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
