import time

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset

from autoencoder.enums import DataSplit


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_cfg, batch_size: int, num_workers: int):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.gpu_transfer_start = time.time()  # For profiler

    def on_before_batch_transfer(self, batch, dataloader_idx=0):
        torch.cuda.synchronize()
        self.gpu_transfer_start = time.perf_counter()
        return super().on_before_batch_transfer(batch, dataloader_idx)

    def setup(self, stage=None):
        self.train_set: Dataset = instantiate(self.dataset_cfg, data_split=DataSplit.TRAIN)
        self.val_set: Dataset = instantiate(self.dataset_cfg, data_split=DataSplit.VAL)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)
