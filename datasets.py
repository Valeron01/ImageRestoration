import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, images_paths, augmentations):
        assert len(images_paths) != 0
        self.images_paths = images_paths
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.images_paths[item])
        image = np.divide(image, 255.0, dtype=np.float32)
        if self.augmentations is not None:
            image = self.augmentations(image=image)["image"]

        image = torch.from_numpy(image).permute(2, 0, 1)
        return image
