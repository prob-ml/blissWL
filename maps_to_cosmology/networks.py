import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResidualBlock(nn.Module):
    """Basic block for a ResNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(identity)  # identity shortcut
        out += identity
        return F.silu(out)


class ResNet(nn.Module):
    """ResNet encoder for convergence maps."""

    def __init__(
        self, num_bins: int, map_slen: int, output_dim: int, base_channels: int = 32
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(num_bins, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
        )

        self.layer1 = ResidualBlock(base_channels, base_channels)
        self.layer2 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.layer3 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(base_channels * 4, output_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)
