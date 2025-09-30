import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder.models.decoder import Decoder
from autoencoder.models.discriminator import Discriminator
from autoencoder.models.encoder import Encoder
from common.logging.data_logger import DataLogger


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
        self.logger: DataLogger
        self.automatic_optimization = False  # required for manual multi-opt

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
        optimizers = self.optimizers()
        assert isinstance(optimizers, list)
        gen_opt = optimizers[0]
        desc_opt = optimizers[1]

        with torch.amp.autocast_mode.autocast("cuda", dtype=torch.float16):
            real = batch
            fake = self.forward(real)

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

        with torch.amp.autocast_mode.autocast("cuda", dtype=torch.float16):
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

        # save previews
        if batch_idx < 8:
            gt = batch[0]
            pred = recon[0]
            grid = torch.cat([gt, pred], dim=-1)  # side by side
            self.logger.log_image(grid, name=f"preview_{batch_idx:03d}.png", epoch=self.current_epoch)

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
