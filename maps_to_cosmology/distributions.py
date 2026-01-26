import torch
from torch.distributions import Independent, Normal


class IndependentMVN:
    """Independent multivariate normal variational distribution.

    Parameterizes independent Normal distributions from network output.
    Network outputs [B, num_params * 2] with alternating (loc, log_var) pairs.
    """

    def __init__(self, num_params: int, low_clamp: float = -10, high_clamp: float = 10):
        self.num_params = num_params
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp

    def get_dist(self, params: torch.Tensor) -> Independent:
        loc = params[:, 0::2]
        scale = params[:, 1::2].clamp(self.low_clamp, self.high_clamp).exp().sqrt()
        return Independent(Normal(loc, scale), 1)

    def log_prob(self, params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dist = self.get_dist(params)
        return dist.log_prob(targets)

    def sample(self, params: torch.Tensor, use_mode: bool = False) -> torch.Tensor:
        if use_mode:
            return params[:, 0::2]
        dist = self.get_dist(params)
        return dist.sample()


class FullRankMVN:
    """Full-rank multivariate normal variational distribution.

    Models cosmological parameters as a MVN with full-rank covariance matrix,
    allowing correlations between parameters.

    TODO: Implement full-rank MVN architecture.
    """

    def __init__(self, num_params: int):
        self.num_params = num_params
        raise NotImplementedError

    def get_dist(self, params: torch.Tensor):
        raise NotImplementedError

    def log_prob(self, params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, params: torch.Tensor, use_mode: bool = False) -> torch.Tensor:
        raise NotImplementedError
