import torch
import torch.nn as nn


class CGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, up=False, use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode='reflect',bias=False)
            if up is False
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
               nn.InstanceNorm2d(out_channels, affine=True),
               #nn.BatchNorm2d(out_channels)
            nn.LeakyReLU(0.2) if up is False else nn.ReLU(),
        )
        self.useDropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.useDropout is True else x


class Generator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        self.down1 = CGANBlock(features, features * 2, 2, False, False)
        self.down2 = CGANBlock(features * 2, features * 4, 2, False, False)
        self.down3 = CGANBlock(features * 4, features * 8, 2, False, False)
        self.down4 = CGANBlock(features * 8, features * 8, 2, False, False)
        self.down5 = CGANBlock(features * 8, features * 8, 2, False, False)
        self.down6 = CGANBlock(features * 8, features * 8, 2, False, False)
        self.bottle_neck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU()
        )
        self.up1 = CGANBlock(features * 8, features * 8, 2, True, True)
        self.up2 = CGANBlock(features * 8 * 2, features * 8, 2, True, True)
        self.up3 = CGANBlock(features * 8 * 2, features * 8, 2, True, True)
        self.up4 = CGANBlock(features * 8 * 2, features * 8, 2, True, False)
        self.up5 = CGANBlock(features * 8 * 2, features * 4, 2, True, False)
        self.up6 = CGANBlock(features * 4 * 2, features * 2, 2, True, False)
        self.up7 = CGANBlock(features * 2 * 2, features, 2, True, False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, 2, kernel_size=4, stride=2, padding=1),
            # nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottle_neck = self.bottle_neck(d7)
        u1 = self.up1(bottle_neck)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))




