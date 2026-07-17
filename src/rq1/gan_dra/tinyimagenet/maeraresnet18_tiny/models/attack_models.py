from __future__ import annotations

from torch import nn


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FeatureCritic(nn.Module):
    def __init__(self, in_channels: int, spatial: int):
        super().__init__()
        hidden = min(max(in_channels, 128), 512)
        layers = [
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        while spatial > 4:
            layers += [
                nn.Conv2d(hidden, min(hidden * 2, 512), 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            hidden = min(hidden * 2, 512)
            spatial //= 2
        layers += [nn.Flatten(), nn.Linear(hidden * spatial * spatial, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


class InversionDecoder(nn.Module):
    def __init__(self, in_channels: int, spatial: int, target_size: int = 64):
        super().__init__()
        hidden = min(max(in_channels, 128), 512)
        layers = [
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        ]
        while spatial < target_size:
            out_channels = max(hidden // 2, 64)
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(hidden, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            hidden = out_channels
            spatial *= 2
        layers += [
            nn.Conv2d(hidden, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
