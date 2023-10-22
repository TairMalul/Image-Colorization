import torch
from torch import nn as nn


class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride,
                      bias=False, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels, affine=True),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, bias=False, padding=1,
                      padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CnnBlock(in_channels, feature, stride=2 if feature != 512 else 1))
            in_channels = feature
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, 1, 4, 1, 1, padding_mode='reflect')
        ))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)

