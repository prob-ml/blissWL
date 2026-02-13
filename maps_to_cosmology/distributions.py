import torch
from torch.distributions import Independent, Normal, MultivariateNormal


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

    Network outputs [B, n + n(n+1)/2] where:
      - first n entries are the mean vector μ
      - remaining entries parameterize the lower-triangular Cholesky factor L
        (including diagonal), so Σ = L L^T.
    We constrain diag(L) > 0 via exp(.) + eps for numerical stability.
    """
    def __init__(
        self,
        num_params: int,
        diag_eps: float = 1e-6,
        diag_clamp_low: float = -10.0,
        diag_clamp_high: float = 10.0,
    ):
        self.num_params = num_params
        self.diag_eps = diag_eps
        self.diag_clamp_low = diag_clamp_low
        self.diag_clamp_high = diag_clamp_high
        self.num_tril_params = num_params * (num_params + 1) // 2

    def get_dist(self, params: torch.Tensor):
        B = params.shape[0]
        n = self.num_params
        # 1) split mean and tril params
        mu = params[:, :n]
        tril_params = params[:, n:]
        # 2) fill lower-triangular matrix L for each batch element
        L = torch.zeros(B, n, n, device=params.device, dtype=params.dtype)
        tril_idx = torch.tril_indices(row=n, col=n, offset=0, device=params.device)
        L[:, tril_idx[0], tril_idx[1]] = tril_params
        # 3) constrain diagonal to be positive
        diag = torch.arange(n, device=params.device)
        diag_raw = L[:, diag, diag].clamp(self.diag_clamp_low, self.diag_clamp_high)
        L[:, diag, diag] = diag_raw.exp() + self.diag_eps
        # 4) MVN distribution
        return MultivariateNormal(loc=mu, scale_tril=L)

    def log_prob(self, params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dist = self.get_dist(params)
        return dist.log_prob(targets)

    def sample(self, params: torch.Tensor, use_mode: bool = False) -> torch.Tensor:
        if use_mode:
            return params[:, : self.num_params]
        dist = self.get_dist(params)
        return dist.rsample()
