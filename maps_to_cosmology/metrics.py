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


class PearsonCorrelationCoefficient(Metric):
    """Compute Pearson correlation coefficient for each cosmological parameter separately."""

    def __init__(self, param_names: list[str], eps: float = 1e-12):
        super().__init__()
        self.param_names = param_names
        self.eps = eps
        p = len(param_names)

        self.add_state("sum_x",  default=torch.zeros(p), dist_reduce_fx="sum")
        self.add_state("sum_y",  default=torch.zeros(p), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=torch.zeros(p), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=torch.zeros(p), dist_reduce_fx="sum")
        self.add_state("sum_xy", default=torch.zeros(p), dist_reduce_fx="sum")
        self.add_state("total",  default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets [B, num_params]."""
        x = preds.detach().float()
        y = target.detach().float()

        self.sum_x  += x.sum(dim=0)
        self.sum_y  += y.sum(dim=0)
        self.sum_x2 += (x * x).sum(dim=0)
        self.sum_y2 += (y * y).sum(dim=0)
        self.sum_xy += (x * y).sum(dim=0)
        self.total  += x.shape[0]

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute per-parameter Pearson correlation coefficient."""
        n = self.total.float()

        num = n * self.sum_xy - self.sum_x * self.sum_y
        den_x = n * self.sum_x2 - self.sum_x * self.sum_x
        den_y = n * self.sum_y2 - self.sum_y * self.sum_y

        den = torch.sqrt(torch.clamp(den_x, min=0.0) * torch.clamp(den_y, min=0.0)) + self.eps
        r = num / den
        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

        return {name: r[i] for i, name in enumerate(self.param_names)}


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
    
    def create_scatterplots(
        self,
        param_names: list[str],
        nrows: int = 2,
        ncols: int = 3,
        figsize: tuple[int, int] = (12, 7),
        alpha: float = 0.5,
        s: int = 10,
    ) -> plt.Figure:
        preds, targets = self.compute()
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        axes = axes.ravel()

        for j, name in enumerate(param_names):
            ax = axes[j]
            x = targets[:, j]
            y = preds[:, j]
            ax.scatter(x, y, alpha=alpha, s=s)

            lo = min(x.min().item(), y.min().item())
            hi = max(x.max().item(), y.max().item())
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            ax.set_xlabel(f"True {name}")
            ax.set_ylabel(f"Predicted {name}")
            ax.set_title(f"{name}: True vs Predicted")

        return fig
    
    def create_omega_c_scatter(self) -> plt.Figure:
    
        return self.create_scatterplots(["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"], figsize=(12, 7))

