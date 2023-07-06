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





if __name__ == '__main__':
    import cv2

    image = cv2.imread(r"H:\ExtractedDatasets\Coco\train2017\000000000025.jpg")
    image = cv2.resize(image, (384, 384)) / 255

    noiser = ImageNoiser(0.01, 0.5)
    image = torch.from_numpy(image).float()
    for i in trange(1000):
        noised_image = noiser(image)
        cv2.imshow("Noised image", noised_image.numpy())
        cv2.waitKey()