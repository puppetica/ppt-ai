import pytorch_lightning as pl
import torch
from torch import nn

from common.logging.data_logger import DataLogger


class VioModule(pl.LightningModule):
    def __init__(self, lr: float, data_logger: DataLogger):
        super().__init__()
        self.lr = lr
        self.data_logger = data_logger

        # simple model
        self.model = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 3))

        self.loss_fn = nn.MSELoss()  # or any loss suitable for your task

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
