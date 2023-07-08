import torch
import torchsummary
from torch import nn

from modules.conv_blocks import ConvBnReluBlock


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            ConvBnReluBlock(64, 64, 2, negative_slope=0.2),

            ConvBnReluBlock(64, 128, 1, negative_slope=0.2),
            ConvBnReluBlock(128, 128, 2, negative_slope=0.2),

            ConvBnReluBlock(128, 256, 1, negative_slope=0.2),
            ConvBnReluBlock(256, 256, 2, negative_slope=0.2),

            ConvBnReluBlock(256, 512, 1, negative_slope=0.2),
            ConvBnReluBlock(512, 512, 1, negative_slope=0.2),
            nn.Conv2d(512, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    torchsummary.summary(Discriminator(), (3, 256, 256), device="cpu")

