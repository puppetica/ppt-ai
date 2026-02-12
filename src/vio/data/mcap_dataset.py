import math
import os
import random

import torch
from torch.utils.data import IterableDataset, get_worker_info

from vio.data.augmentation import gen_aug_set
from vio.data.batch_data import Batch, CameraBatch
from vio.data.mcap_streamer import McapStreamer
from vio.enums import DataSplit


def collate_batch_list(batch: list[Batch]) -> Batch:
    seq_name = [b["seq_name"][0] for b in batch]
    input_mcap_path = [b["input_mcap_path"][0] for b in batch]
    ts_ns = [b["ts_ns"][0] for b in batch]

    # These already have batch dim=1, so cat instead of stack
    imgs = torch.cat([b["cam_front"]["img"] for b in batch], dim=0)
    imgs_tm1 = torch.cat([b["cam_front"]["img_tm1"] for b in batch], dim=0)
    intr = torch.cat([b["cam_front"]["intr"] for b in batch], dim=0)
    extr = torch.cat([b["cam_front"]["extr"] for b in batch], dim=0)
    timediff_s = torch.cat([b["cam_front"]["timediff_s"] for b in batch], dim=0)
    cam_front = CameraBatch(
        img=imgs,
        img_tm1=imgs_tm1,
        intr=intr,
        extr=extr,
        timediff_s=timediff_s,
    )
    # imu already correct shape: just flatten the outer list
    imu = [b["imu"][0] for b in batch]
    imu_dt = torch.cat([b["imu_dt"] for b in batch], dim=0)

    gt_ego_motion = torch.cat([b["gt_ego_motion"] for b in batch], dim=0)
    gt_ego_motion_valid = torch.cat([b["gt_ego_motion_valid"] for b in batch], dim=0)

    return Batch(
        seq_name=seq_name,
        input_mcap_path=input_mcap_path,
        ts_ns=ts_ns,
        cam_front=cam_front,
        imu=imu,
        imu_dt=imu_dt,
        gt_ego_motion=gt_ego_motion,
        gt_ego_motion_valid=gt_ego_motion_valid,
    )


class McapDataset(IterableDataset):
    def __init__(
        self,
        target_height: int,
        target_width: int,
        crop_top: int,
        crop_bottom: int,
        scale: float,
        root_dir: str,
        data_split: DataSplit,
        max_batch_size: int,
    ):
        assert max_batch_size > 0
        self.data_split = data_split
        self.max_batch_size = max_batch_size

        split_dir = "train" if data_split == DataSplit.TRAIN else "val"
        split_path = os.path.join(root_dir, split_dir)
        self.mcap_files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.lower().endswith(".mcap")]

        self.mcap_streamer_queue = []
        self.streamer_args = (target_height, target_width, crop_top, crop_bottom, scale)

    def __iter__(self):
        mcap_files_set = self.mcap_files.copy()

        # Support multi processing -> split mcaps per process
        worker_info = get_worker_info()
        if worker_info is not None:
            n = len(self.mcap_files)
            per_worker = int(math.ceil(n / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, n)
            mcap_files_set = self.mcap_files[start:end]

        if self.data_split == DataSplit.TRAIN:
            random.shuffle(mcap_files_set)  # shuffle file order once per epoch

        if len(mcap_files_set) == 0:
            return

        seq_idx = 0
        while True:
            if len(self.mcap_streamer_queue) < self.max_batch_size and seq_idx < len(mcap_files_set):
                streamer = McapStreamer(
                    *self.streamer_args, mcap_path=mcap_files_set[seq_idx], do_aug=self.data_split == DataSplit.TRAIN
                )
                self.mcap_streamer_queue.append(iter(streamer))
                seq_idx += 1
                continue

            batch = []
            for streamer in list(self.mcap_streamer_queue):  # iterate over a copy since we may remove elements
                try:
                    frame = next(streamer)
                    batch.append(frame)
                except StopIteration:
                    self.mcap_streamer_queue.remove(streamer)

            if batch:
                yield collate_batch_list(batch)

            if len(self.mcap_streamer_queue) == 0:
                break
