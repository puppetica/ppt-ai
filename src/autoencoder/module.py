import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import Decoder
from models.discriminator import Discriminator
from models.encoder import Encoder


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        in_ch: int,
        res_block_ch: list[int],
        num_attn_blocks: int,
        adv_weight: float = 0.001,
    ):
        super().__init__()
        self.automatic_optimization = False  # required for manual multi-opt

        self.save_hyperparameters()
        self.lr = lr
        self.encoder = Encoder(in_ch, res_block_ch, num_attn_blocks)
        self.decoder = Decoder(in_ch, res_block_ch, num_attn_blocks)
        self.discriminator = Discriminator(in_ch=in_ch, base_ch=64)

        self.l1 = nn.L1Loss()
        self.adv_weight = adv_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def adversarial_loss(self, pred: torch.Tensor, target_is_real: bool = True):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return F.binary_cross_entropy_with_logits(pred, target)

    def training_step(self, batch, batch_idx):
        real = batch
        fake = self.forward(real)

        gen_opt, desc_opt = self.optimizers()

        # Train Discriminator
        pred_real = self.discriminator(real)
        pred_fake = self.discriminator(fake.detach())
        loss_real = self.adversarial_loss(pred_real, True)
        loss_fake = self.adversarial_loss(pred_fake, False)
        d_loss = (loss_real + loss_fake) / 2
        # Optimize Discriminator
        desc_opt.zero_grad()
        self.manual_backward(d_loss)
        desc_opt.step()
        self.log("d_loss", d_loss, prog_bar=True)

        # Train Generator
        pred_fake = self.discriminator(fake)
        l1_loss = self.l1(fake, real)
        adv_loss = self.adversarial_loss(pred_fake, True)
        g_loss = l1_loss + self.adv_weight * adv_loss
        # Optimize Generator
        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        gen_opt.step()
        self.log("g_loss", g_loss, prog_bar=True)
        self.log("g_l1", l1_loss, prog_bar=True)
        self.log("g_adv", adv_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        recon = self.forward(batch)
        l1_loss = self.l1(recon, batch)
        self.log("val_l1", l1_loss, prog_bar=True)

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999),
        )
        desc_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
        )
        return [gen_opt, desc_opt], []
