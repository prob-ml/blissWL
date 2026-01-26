import matplotlib.pyplot as plt
import torch
from torchmetrics import Metric


class RootMeanSquaredError(Metric):
    """Compute RMSE for each cosmological parameter separately."""

    def __init__(self, param_names: list[str]):
        super().__init__()
        self.param_names = param_names
        self.add_state(
            "sum_squared_error",
            default=torch.zeros(len(param_names)),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets [B, num_params]."""
        self.sum_squared_error += ((preds - target) ** 2).sum(dim=0)
        self.total += preds.shape[0]

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute per-parameter RMSE."""
        rmse = (self.sum_squared_error / self.total).sqrt()
        return {name: rmse[i] for i, name in enumerate(self.param_names)}


class ScatterPlot(Metric):
    """Collect predictions and targets for scatterplot generation."""

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Store predictions and targets."""
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated predictions and targets."""
        return torch.cat(self.preds, dim=0), torch.cat(self.targets, dim=0)

    def create_param_scatter(self, param_idx: int, param_name: str) -> plt.Figure:
        """Create scatterplot of true vs predicted for a selected parameter."""
        preds, targets = self.compute()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            targets[:, param_idx].cpu(), preds[:, param_idx].cpu(), alpha=0.5, s=10
        )

        # Add diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel(f"True {param_name}")
        ax.set_ylabel(f"Predicted {param_name}")
        ax.set_title(f"{param_name}: True vs Predicted")
        ax.legend()
        plt.tight_layout()
        return fig


class PearsonCorrelationCoefficient(Metric):
    """Compute Pearson Correlation for each cosmological parameter separately."""

    def __init__(self, param_names: list[str], eps: float = 1e-8):
        super().__init__()
        self.param_names = param_names
        self.eps = eps  # avoid zero division
        N = len(param_names)
        self.add_state("sum_x", default=torch.zeros(N), dist_reduce_fx="sum")
        self.add_state("sum_y", default=torch.zeros(N), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=torch.zeros(N), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=torch.zeros(N), dist_reduce_fx="sum")
        self.add_state("sum_xy", default=torch.zeros(N), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets [B, num_params]."""
        self.sum_x += preds.sum(dim=0)
        self.sum_y += target.sum(dim=0)
        self.sum_x2 += (preds**2).sum(dim=0)
        self.sum_y2 += (target**2).sum(dim=0)
        self.sum_xy += (preds * target).sum(dim=0)
        self.total += preds.shape[0]

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute per-parameter Pearson Correlation coefficient."""
        n = self.total.to(self.sum_x.dtype)

        num = n * self.sum_xy - self.sum_x * self.sum_y
        den_x = n * self.sum_x2 - self.sum_x**2
        den_y = n * self.sum_y2 - self.sum_y**2
        denom = (
            torch.sqrt(torch.clamp(den_x, min=0.0) * torch.clamp(den_y, min=0.0))
            + self.eps
        )  # eps
        results = num / denom
        return {name: results[i] for i, name in enumerate(self.param_names)}
