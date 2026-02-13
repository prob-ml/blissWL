import torch
from torch.distributions import Independent, Normal
from torch.distributions import MultivariateNormal


class IndependentMVN:
    """Independent multivariate normal variational distribution.

    Parameterizes independent Normal distributions from network output.
    Network outputs [B, num_params * 2] with alternating (loc, log_var) pairs.
    """

    def __init__(self, num_params: int, low_clamp: float = -10, high_clamp: float = 10):
        self.num_params = num_params
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp
        self.output_dim = num_params * 2

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

    def __init__(self, num_params: int, low_clamp: float = -14, high_clamp: float = 10, eps: float = 1e-6):
        self.num_params = num_params
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp
        self.eps = eps
        D = num_params
        self.output_dim = D + D * (D + 1) // 2

    def get_dist(self, params: torch.Tensor) -> MultivariateNormal:
        B = params.shape[0]
        D = self.num_params

        mu = params[:, :D]
        tril_params = params[:, D:]

        L = params.new_zeros(B, D, D)
        tril_idx = torch.tril_indices(D, D, 0, device=params.device)

        L[:, tril_idx[0], tril_idx[1]] = tril_params

        diag = L[:, range(D), range(D)]
        L[:, range(D), range(D)] = diag.exp() + self.eps

        return MultivariateNormal(loc=mu, scale_tril=L)

    def log_prob(self, params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dist = self.get_dist(params)
        return dist.log_prob(targets)

    def sample(self, params: torch.Tensor, use_mode: bool = False) -> torch.Tensor:
        if use_mode:
            return params[:, :self.num_params]
        dist = self.get_dist(params)
        return dist.sample()
