from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ResNetMini(nn.Module):
    def __init__(self, num_classes: int = 1000, width: int = 2, blocks: int = 4):
        super().__init__()

        norm_layer = nn.BatchNorm2d

        self.width = width
        n_channels = [16, 16 * width, 32 * width, 64 * width]
        self.conv1 = nn.Conv2d(
            3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(n_channels[0], n_channels[1], blocks)
        self.layer2 = self._make_layer(n_channels[1], n_channels[2], blocks, stride=2)
        self.layer3 = self._make_layer(n_channels[2], n_channels[3], blocks, stride=2)
        self.bn = norm_layer(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(n_channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int = 1
    ):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride), norm_layer(out_channels)
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        out = self.linear(out)
        return out


def get_model(name: str) -> nn.Module:
    if name == "resnet20":
        return ResNetMini(num_classes=10, width=1, blocks=3)
    elif name == "wrn_28_2":
        return ResNetMini(num_classes=10, width=2, blocks=4)
    elif name == "wrn_28_10":
        return ResNetMini(num_classes=10, width=10, blocks=4)
    elif name == "mlp500":
        return MLP(28 * 28, 500, 10)
    elif name == "linear":
        return LinearModel(28 * 28, 10)
    elif name in models.__dict__:
        fn = models.__dict__[name]
    else:
        raise RuntimeError("Unknown model name {}".format(name))

    return fn(num_classes=10)
