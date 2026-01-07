import torch
import torch.nn as nn
from lightning import LightningModule


class Encoder(LightningModule):
    """Simple MLP that maps convergence maps to variational posterior parameters.

    Takes convergence maps [B, 5, 256, 256] and outputs [B, 12] tensor with
    alternating loc/scale parameters for 6 independent Normal distributions
    over cosmological parameters.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_cosmo_params: int = 6,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_cosmo_params = num_cosmo_params

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 256 * 256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_cosmo_params * 2),
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

        # Print predictions vs truth for first batch of each validation epoch
        if batch_idx == 0:
            out = self.forward(maps)
            loc = out[:, 0::2]  # Posterior means [B, 6]
            param_names = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]
            print(f"\n{'Sample':<8} {'Param':<10} {'Predicted':>12} {'True':>12}")
            print("-" * 44)
            for sample_idx in range(loc.shape[0]):
                for i, name in enumerate(param_names):
                    pred = loc[sample_idx, i].item()
                    true = params[sample_idx, i].item()
                    print(f"{sample_idx:<8} {name:<10} {pred:>12.4f} {true:>12.4f}")
                print()

        loss = self.compute_loss(maps, params)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        maps, params = batch
        loss = self.compute_loss(maps, params)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
