import argparse
import os
import sys

import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers
from torch.utils.data import DataLoader

import dataset_builder
from lit_modules.simple_denoiser import SimpleDenoiser
from modules.conv_blocks import ConvBnReluBlock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", required=True)
    parser.add_argument("--val_folder", required=False, default=None)
    parser.add_argument("--batch_size", required=False, default=16)
    parser.add_argument("--num_workers", required=False, default=4)
    parser.add_argument("--lightning_folder", required=False, default="./lighting")
    parser.add_argument("--checkpoints_folder", required=False, default="./lighting/checkpoints")
    parser.add_argument("--max_epochs", required=False, default=100)
    args = parser.parse_args()

    train_dataset = dataset_builder.build_dataset(args.train_folder)
    val_dataset = dataset_builder.build_dataset(args.val_folder) if args.val_folder else None

    # train_dataset = torch.utils.data.Subset(train_dataset, range(15_000))
    val_dataset = torch.utils.data.Subset(val_dataset, range(1_000))

    train_loader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers
    ) if val_dataset else None

    model = SimpleDenoiser([64, 128, 256, 512, 1024], block=ConvBnReluBlock, variance_min=0.01, variance_max=0.5)

    logger = pl.loggers.TensorBoardLogger(args.lightning_folder)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(args.checkpoints_folder, f"run_{logger.version}"), monitor="val_loss", verbose=True, save_last=True
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=20
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
