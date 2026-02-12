# ruff: noqa: F821
from typing import TypedDict

import torch


class CameraBatch(TypedDict):
    # [channels, height, width]
    # Shape: [B, C, H, W]
    img: torch.Tensor
    # t-1 [channels, height, width]
    # Shape: [B, C, H, W]
    img_tm1: torch.Tensor
    # [fx, fy, cx, cy]
    # Shape: [B, 4]
    intr: torch.Tensor
    # sensor -> ego [x, y, z, q_x, q_y, q_z, q_w]
    # Shape: [B, 7]
    extr: torch.Tensor
    # timediff to t-1 in seconds
    # Shape: [B, 1]
    timediff_s: torch.Tensor


class Batch(TypedDict):
    # ----------------------------------
    # Meta Data
    # ----------------------------------
    # Name of sequence for each batch element
    seq_name: list[str]
    # Path to the mcap data used for model input
    input_mcap_path: list[str]
    # Master timestamp in ns
    ts_ns: list[int]

    # ----------------------------------
    # Sensor Input
    # ----------------------------------
    cam_front: CameraBatch
    # N-number of IMU readings [a_x, a_y, a_z, accel_x, accel_y, accel_z, timediff_s (to master ts_ns)]
    # Shape: [B, N] list of [7,] torch tensors
    imu: list[list[torch.Tensor]]
    # How far did we move in meters from IMU list (this is kind of rough, but enough for scaling)
    # Shape: [B, 1]
    imu_dt: torch.Tensor

    # ----------------------------------
    # Ground Truth / Labels
    # ----------------------------------
    # t-1 -> t transformation in ego [x, y, z, q_x, q_y, q_z, q_w]
    # Shape: [B, 7]
    gt_ego_motion: torch.Tensor
    # valid flag for the ego motion
    # Shape: [B, 1]
    gt_ego_motion_valid: torch.Tensor
