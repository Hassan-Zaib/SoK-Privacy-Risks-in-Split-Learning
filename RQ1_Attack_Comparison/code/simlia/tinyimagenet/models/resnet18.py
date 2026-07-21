from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class CifarResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class ResNetHead(nn.Module):
    def __init__(self, avgpool: nn.Module, fc: nn.Module):
        super().__init__()
        self.avgpool = avgpool
        self.fc = fc

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


@dataclass(frozen=True)
class CutDefinition:
    client_blocks: tuple[str, ...]
    server_blocks: tuple[str, ...]


CUT_DEFINITIONS = {
    "c1": CutDefinition(("stem", "layer1"), ("layer2", "layer3", "layer4", "head")),
    "c2": CutDefinition(("stem", "layer1", "layer2"), ("layer3", "layer4", "head")),
    "c3": CutDefinition(("stem", "layer1", "layer2", "layer3"), ("layer4", "head")),
    "c4": CutDefinition(("stem", "layer1", "layer2", "layer3", "layer4"), ("head",)),
}


def _block_map(model: CifarResNet18) -> dict[str, nn.Module]:
    return {
        "stem": nn.Sequential(model.conv1, model.bn1, model.relu),
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
        "head": ResNetHead(model.avgpool, model.fc),
    }


def build_split_resnet18(cut: str, num_classes: int = 10) -> tuple[nn.Module, nn.Module]:
    cut = cut.lower()
    if cut not in CUT_DEFINITIONS:
        raise ValueError(f"Unknown cut {cut}")
    model = CifarResNet18(num_classes=num_classes)
    blocks = _block_map(model)
    definition = CUT_DEFINITIONS[cut]
    client = nn.Sequential(*[blocks[name] for name in definition.client_blocks])
    server = nn.Sequential(*[blocks[name] for name in definition.server_blocks])
    return client, server
