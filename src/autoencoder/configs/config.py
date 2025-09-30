from typing import Any

from pydantic import BaseModel


class McapLoaderCfg(BaseModel):
    type: str
    target_height: int
    target_width: int
    crop_top: int
    crop_bottom: int
    scale: float
    root_dir: str
    topics: list[str]


class ImgLoaderCfg(BaseModel):
    type: str
    target_height: int
    target_width: int
    crop_top: int
    crop_bottom: int
    scale: float
    root_dir: str


class ModelCfg(BaseModel):
    lr: float
    in_ch: int
    res_block_ch: list[int]
    num_attn_blocks: int
    adv_weight: float


class TrainCfg(BaseModel):
    name: str
    max_epochs: int
    accelerator: str
    devices: int
    batch_size: int
    num_workers: int
    run_profiler: bool
    datasets: list[ImgLoaderCfg | McapLoaderCfg]
    model: ModelCfg


class PredictCfg(BaseModel):
    datasets: dict[str, Any]
    accelerator: str
    devices: int
    batch_size: int
    num_workers: int
