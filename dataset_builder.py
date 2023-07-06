import os

import cv2
import albumentations as al


def build_dataset(images_folder_path):
    paths = os.listdir(images_folder_path)
    paths.sort()



