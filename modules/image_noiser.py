import random

import torch


from torch import nn
from tqdm import trange


class ImageNoiser(nn.Module):
    def __init__(self, variance_min, variance_max):
        super().__init__()
        self.variance_min = variance_min
        self.variance_max = variance_max

    def forward(self, x):
        noise = torch.randn_like(x)
        noised_data = x + (self.variance_min + random.random() * (self.variance_max - self.variance_min)) * noise
        return torch.clip(noised_data, 0, 1)

