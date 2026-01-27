import torch
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet-18 encoder for convergence maps.
    
    """

    def __init__(
        self,
        num_bins: int,
        map_slen: int,
        output_dim: int,
    ):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_bins, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, output_dim)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride=stride))
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)          # [B, 512, 1, 1]
        x = torch.flatten(x, 1)      # [B, 512]
        x = self.fc(x)               # [B, output_dim]
        return x




'''
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
'''
