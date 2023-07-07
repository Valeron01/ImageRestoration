import pytorch_lightning as pl
import torch.optim
from torch import nn

from modules.image_noiser import ImageNoiser
from modules.rgb_to_bw_converter import RGBToBWConverter
from modules.unet import UNet


class SimpleColorizator(pl.LightningModule):
    def __init__(
            self,
            n_features_list,
            block,
    ):
        super().__init__()
        self.model = UNet(n_features_list, block, in_channels=1)
        self.noiser = RGBToBWConverter()

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)

        return optimizer

    def forward(self, x):
        return self.model(x)

    def model_step(self, x):
        noised_images = self.noiser(x)

        restored = self.model(noised_images)
        loss = nn.functional.mse_loss(restored, x * 2 - 1)

        return loss

    def training_step(self, batch, *args):
        loss = self.model_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, *args):
        loss = self.model_step(batch)
        self.log("val_loss", loss)
