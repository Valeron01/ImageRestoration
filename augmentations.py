import albumentations as al
import numpy as np


class BWChecker(al.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if np.allclose(img[:, :, 0], img[:, :, 1]):
            return img * 0
        return img
