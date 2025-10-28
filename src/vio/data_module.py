import time

import pytorch_lightning as pl
import torch
from torch.utils.data import ChainDataset, DataLoader

from vio.batch_data import Batch, CameraBatch
from vio.configs.config import McapStreamerCfg
from vio.enums import DataSplit
from vio.mcap_streamer import McapStreamer


def collate_fn(batch: list[Batch]) -> Batch:
    seq_name = [b["seq_name"][0] for b in batch]

    # These already have batch dim=1, so cat instead of stack
    imgs = torch.cat([b["cam_front"]["img"] for b in batch], dim=0)
    imgs_tm1 = torch.cat([b["cam_front"]["img_tm1"] for b in batch], dim=0)
    intr = torch.cat([b["cam_front"]["intr"] for b in batch], dim=0)
    extr = torch.cat([b["cam_front"]["extr"] for b in batch], dim=0)
    ts_sec = torch.cat([b["cam_front"]["ts_sec"] for b in batch], dim=0)
    cam_front = CameraBatch(
        img=imgs,
        img_tm1=imgs_tm1,
        intr=intr,
        extr=extr,
        ts_sec=ts_sec,
    )
    # imu already correct shape: just flatten the outer list
    imu = [b["imu"][0] for b in batch]

    gt_ego_motion = torch.cat([b["gt_ego_motion"] for b in batch], dim=0)
    gt_ego_motion_valid = torch.cat([b["gt_ego_motion_valid"] for b in batch], dim=0)

    return Batch(
        seq_name=seq_name,
        cam_front=cam_front,
        imu=imu,
        gt_ego_motion=gt_ego_motion,
        gt_ego_motion_valid=gt_ego_motion_valid,
    )


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_cfg: list[McapStreamerCfg], batch_size: int, num_workers: int):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.gpu_transfer_start = time.time()  # For profiler

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
        train_sets = [McapStreamer(**cfg.model_dump(), data_split=DataSplit.TRAIN) for cfg in self.dataset_cfg]
        val_sets = [McapStreamer(**cfg.model_dump(), data_split=DataSplit.VAL) for cfg in self.dataset_cfg]

        self.train_set = ChainDataset(train_sets)
        self.val_set = ChainDataset(val_sets)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
        )
