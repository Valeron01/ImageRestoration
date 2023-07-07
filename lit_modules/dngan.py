import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from modules.discriminator import Discriminator
from modules.image_noiser import ImageNoiser
from modules.unet import UNet
from modules.vgg_loss import VGGLoss


class DNGAN(pl.LightningModule):
    def __init__(
            self,
            n_features_list,
            block,
            variance_min, variance_max,
            generator_state_dict=None
    ):
        super().__init__()
        self.automatic_optimization = False

        self.generator = UNet(n_features_list, block)
        if generator_state_dict:
            self.generator.load_state_dict(generator_state_dict)
            print("Loaded generator state dict")

        self.discriminator = Discriminator()

        self.noiser = ImageNoiser(variance_min, variance_max)
        self.perceptual_loss = VGGLoss()

        self.save_hyperparameters(ignore=["generator_state_dict"])

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
        return [opt_g, opt_d], []

    def training_step(self, images):
        optimizer_g, optimizer_d = self.optimizers()

        noised_images = self.noiser(images)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        denoised_images = self.generator(noised_images) * 0.5 + 0.5

        valid = torch.ones(noised_images.size(0), 1)
        valid = valid.type_as(noised_images)

        gen_adv_loss = binary_cross_entropy_with_logits(self.discriminator(denoised_images), valid)
        gen_content_loss = self.perceptual_loss(denoised_images, images)
        g_loss = gen_content_loss + gen_adv_loss * 1e-3

        self.log("g_loss", g_loss, prog_bar=True)
        self.log("gen_adv_loss", gen_adv_loss, prog_bar=True)
        self.log("gen_content_loss", gen_content_loss, prog_bar=True)

        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        optimizer_g.step()

        self.untoggle_optimizer(optimizer_g)

        # And now the training step of the discriminator

        self.toggle_optimizer(optimizer_d)

        real_loss = binary_cross_entropy_with_logits(self.discriminator(images), valid)

        fake = torch.zeros(images.size(0), 1)
        fake = fake.type_as(images)

        noised_images = self.noiser(images)
        with torch.no_grad():
            denoised_images = self(noised_images) * 0.5 + 0.5
        fake_loss = binary_cross_entropy_with_logits(self.discriminator(denoised_images), fake)

        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)

        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()

        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, *args):
        noised_images = self.noiser(batch)
        restored = self.generator(noised_images)
        loss = nn.functional.mse_loss(restored, batch * 2 - 1)
        self.log("val_loss", loss)
