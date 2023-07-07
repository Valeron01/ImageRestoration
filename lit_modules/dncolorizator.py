import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from modules.discriminator import Discriminator
from modules.image_noiser import ImageNoiser
from modules.rgb_to_bw_converter import RGBToBWConverter
from modules.unet import UNet
from modules.vgg_loss import VGGLoss


class DNColorizator(pl.LightningModule):
    def __init__(
            self,
            n_features_list,
            block,
            generator_state_dict=None
    ):
        super().__init__()
        self.automatic_optimization = False
        self.generator = UNet(n_features_list, block, in_channels=1)

        if generator_state_dict:
            self.generator.load_state_dict(generator_state_dict)
            print("Loaded generator state dict")

        self.discriminator = Discriminator()

        self.decolorizer = RGBToBWConverter()
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

        bw_images = self.decolorizer(images)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        colorized_images = self.generator(bw_images) * 0.5 + 0.5

        disc_denoised = self.discriminator(colorized_images)
        ones = torch.ones_like(disc_denoised)
        gen_adv_loss = binary_cross_entropy_with_logits(disc_denoised, ones)

        gen_content_loss = self.perceptual_loss(colorized_images, images)
        g_loss = gen_content_loss + gen_adv_loss * 5e-2

        self.log("g_loss", g_loss, prog_bar=True)
        self.log("gen_adv_loss", gen_adv_loss, prog_bar=True)
        self.log("gen_content_loss", gen_content_loss, prog_bar=True)

        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        optimizer_g.step()

        self.untoggle_optimizer(optimizer_g)

        # And now the training step of the discriminator

        self.toggle_optimizer(optimizer_d)

        real_loss = binary_cross_entropy_with_logits(self.discriminator(images), ones)

        disc_denoised = self.discriminator(colorized_images.detach())
        fake_loss = binary_cross_entropy_with_logits(disc_denoised, torch.zeros_like(disc_denoised))

        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)

        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()

        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, *args):
        noised_images = self.decolorizer(batch)
        restored = self.generator(noised_images)
        loss = nn.functional.mse_loss(restored, batch * 2 - 1)
        self.log("val_loss", loss)
