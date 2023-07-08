import sys

import cv2
import torch

import dataset_builder
from lit_modules.colorizator_gan import ColorizatorGAN
from lit_modules.dngan import DNGAN
from lit_modules.simple_colorizator import SimpleColorizator
from lit_modules.simple_denoiser import SimpleDenoiser
from modules.image_noiser import ImageNoiser
from modules.rgb_to_bw_converter import RGBToBWConverter


def main():
    model = SimpleColorizator.load_from_checkpoint(
        "" # Here path to a checkpoint
    ).eval()

    dataset = dataset_builder.build_dataset(sys.argv[1])
    # image_noiser = RGBToBWConverter(0.01, 0.5)
    image_noiser = RGBToBWConverter()
    for image in dataset:
        noised_image = image_noiser(image)
        with torch.no_grad():
            denoised_image = model(noised_image[None])[0]

        noised_image = torch.tile(noised_image, [3, 1, 1]) if noised_image.shape[0] == 1 else noised_image
        concatenated_image = torch.cat([
            noised_image, denoised_image * 0.5 + 0.5, image
        ], dim=2).permute(1, 2, 0).numpy()
        cv2.imshow("Dataset sample", concatenated_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
