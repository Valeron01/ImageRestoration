import glob
import os

import cv2
import albumentations as al

from datasets import ImagesDataset


def build_dataset(images_folder_path):
    paths = glob.glob(os.path.join(images_folder_path, "*.*"))
    paths.sort()

    augmentations = al.Sequential([
        al.RandomResizedCrop(256, 256)
    ])

    dataset = ImagesDataset(paths, augmentations)

    return dataset

