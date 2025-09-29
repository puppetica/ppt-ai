from typing import Any

from pydantic import BaseModel


class ImgDataset(BaseModel):
    name: str
    crop_height: int
    crop_width: int
    scale: float
    root_dir: str


class ModelCfg(BaseModel):
    lr: float
    in_ch: int
    res_block_ch: list[int]
    num_attn_blocks: int
    adv_weight: float


class TrainCfg(BaseModel):
    datasets: list[ImgDataset]
    model: ModelCfg
    max_epochs: int
    accelerator: str
    devices: int
    batch_size: int
    num_workers: int
    run_profiler: bool


class PredictCfg(BaseModel):
    datasets: dict[str, Any]
    accelerator: str
    devices: int
    batch_size: int
    num_workers: int
