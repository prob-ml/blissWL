import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """Simple two-layer MLP encoder for convergence maps.

    Takes flattened convergence maps and outputs variational parameters.
    """

    def __init__(
        self,
        num_bins: int,
        map_slen: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        input_dim = num_bins * map_slen * map_slen
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    """ResNet encoder for convergence maps.

    TODO: Implement ResNet architecture.
    """

    def __init__(
        self,
        num_bins: int,
        map_slen: int,
        output_dim: int,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
