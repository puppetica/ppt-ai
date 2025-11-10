import time

import pytorch_lightning as pl
import torch
from torch.utils.data import ChainDataset, DataLoader

from vio.configs.config import McapStreamerCfg
from vio.data.batch_data import Batch
from vio.data.mcap_dataset import McapDataset
from vio.enums import DataSplit


def collate_fn(batch) -> Batch:
    # NOTE: Collate happens already in the streamers to ensure max 1 sequence per batch
    # TODO: Do pin memory here?
    return batch


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_cfg: list[McapStreamerCfg], batch_size: int, num_workers: int):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.gpu_transfer_start = time.time()  # for profiler

    def on_before_batch_transfer(self, batch, dataloader_idx=0):
        torch.cuda.synchronize()
        self.gpu_transfer_start = time.perf_counter()
        return super().on_before_batch_transfer(batch, dataloader_idx)

    def transfer_batch_to_device(self, batch: Batch, device, dataloader_idx):
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

        # move nested IMU lists
        imu = batch["imu"]
        for i in range(len(imu)):
            imu[i] = [x.to(device) for x in imu[i]]

        return batch

    def setup(self, stage=None):
        train_sets = [
            McapDataset(**cfg.model_dump(), data_split=DataSplit.TRAIN, max_batch_size=self.batch_size)
            for cfg in self.dataset_cfg
        ]
        val_sets = [
            McapDataset(**cfg.model_dump(), data_split=DataSplit.VAL, max_batch_size=self.batch_size)
            for cfg in self.dataset_cfg
        ]

        self.train_set = ChainDataset(train_sets)
        self.val_set = ChainDataset(val_sets)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=1 if self.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=1 if self.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        # Reuse the val set as long as we are starved for data
        return DataLoader(
            self.val_set,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=1 if self.num_workers > 0 else None,
            collate_fn=collate_fn,
        )
