import sys

import cv2
import torch

import dataset_builder
from modules.image_noiser import ImageNoiser
from modules.rgb_to_bw_converter import RGBToBWConverter


def main():
    dataset = dataset_builder.build_dataset(sys.argv[1])
    image_noiser = RGBToBWConverter()
    # image_noiser = ImageNoiser(0.01, 0.5)
    for image in dataset:
        noised_image = image_noiser(image)
        noised_image = torch.tile(noised_image, [3, 1, 1]) if noised_image.shape[0] == 1 else noised_image
        concatenated_image = torch.cat([noised_image, image], dim=2).permute(1, 2, 0).numpy()
        cv2.imshow("Dataset sample", concatenated_image)
        cv2.waitKey()


if __name__ == '__main__':
    main()
