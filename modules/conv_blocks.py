import typing

from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.identity = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, stride, stride
        )

    def forward(self, x):
        return self.block(x) + self.identity(x)


class SingleConvResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.identity = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, stride, stride
        )

    def forward(self, x):
        return self.block(x) + self.identity(x)


class ConvBnReluBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DWConvBnReluBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False, groups=min(in_channels, out_channels)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 7, stride, 3, bias=False, groups=in_channels),
            nn.BatchNorm2d(in_channels),

            nn.Conv2d(in_channels, in_channels * 4, 1),

            nn.LeakyReLU(),

            nn.Conv2d(in_channels * 4, out_channels, 1)
        )
        self.identity = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, stride, stride
        )

    def forward(self, x):
        return self.block(x) + self.identity(x)

