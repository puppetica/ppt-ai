import time

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader

from autoencoder.data.datasets.img_loader import ImgLoader
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
        def create_dataset(cfg, split: DataSplit):
            dataset_type: str = cfg.type
            kwargs = cfg.model_dump(exclude={"type"})
            if dataset_type == "img_loader":
                return ImgLoader(**kwargs, data_split=split)
            else:
                raise ValueError(f"Unkown dataset type: {dataset_type}")

        train_sets = [create_dataset(cfg, DataSplit.TRAIN) for cfg in self.dataset_cfg]
        val_sets = [create_dataset(cfg, DataSplit.VAL) for cfg in self.dataset_cfg]

        self.train_set = ConcatDataset(train_sets)
        self.val_set = ConcatDataset(val_sets)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
