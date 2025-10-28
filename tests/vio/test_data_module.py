from typing import cast

import pytest
import torch
from omegaconf import OmegaConf

from vio.config.config import TrainCfg
from vio.data_module import DataModule


@pytest.fixture(scope="module")
def train_cfg():
    cfg_path = "src/vio/config/train.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    return TrainCfg(**cfg_dict)  # type: ignore[arg-type]


@pytest.mark.parametrize("split", ["train"])  # , "val", "test"])
def test_lightning_datamodule(train_cfg: TrainCfg, split):
    dm = DataModule(dataset_cfg=train_cfg.datasets, batch_size=train_cfg.batch_size, num_workers=0)

    dm.setup(split)
    loader = getattr(dm, f"{split}_dataloader")()
    batch = next(iter(loader))
    assert batch is not None
