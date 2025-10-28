from pydantic import BaseModel


class McapStreamerCfg(BaseModel):
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


class TrainCfg(BaseModel):
    name: str
    max_epochs: int
    accelerator: str
    devices: int
    batch_size: int
    num_workers: int
    run_profiler: bool
    datasets: list[McapStreamerCfg]
    model: ModelCfg
