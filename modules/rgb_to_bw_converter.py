import cv2
import torch
from torch import nn
import kornia


class RGBToBWConverter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("coefficients", torch.FloatTensor([0.11, 0.59, 0.3]))

    def forward(self, x):
        return kornia.color.bgr_to_grayscale(x)

