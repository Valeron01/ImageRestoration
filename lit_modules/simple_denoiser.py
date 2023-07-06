import pytorch_lightning as pl
import torch.optim
from torch import nn

from modules.image_noiser import ImageNoiser
from modules.unet import UNet


class SimpleDenoiser(pl.LightningModule):
    def __init__(
            self,
            n_features_list,
            block,
            variance_min, variance_max
    ):
        super().__init__()
        self.model = UNet(n_features_list, block)
        self.noiser = ImageNoiser(variance_min, variance_max)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), 1e-3)

        return optimizer

    def model_step(self, x):
        noised_images = self.noiser(x)

        restored = self.model(noised_images)
        loss = nn.functional.l1_loss(restored, x)

        return loss

    def training_step(self, batch, *args):
        loss = self.model_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, *args):
        loss = self.model_step(batch)
        self.log("val_loss", loss)
