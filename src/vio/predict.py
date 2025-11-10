import logging
from datetime import datetime

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from common.logging.data_logger import DataLogger
from vio.configs.config import PredictCfg, TrainCfg
from vio.data.data_module import DataModule
from vio.module import VioModule

logger = logging.getLogger("vio.predict")


@hydra.main(config_path="configs", config_name="predict", version_base=None)
def main(cfg_dict: DictConfig):
    torch.set_float32_matmul_precision("high")

    cfg = PredictCfg.model_validate(cfg_dict)
    name = cfg.name

    train_cfg_path = f"{cfg.path}/{cfg.train_cfg}"
    train_cfg_yaml = OmegaConf.load(train_cfg_path)
    train_cfg = TrainCfg.model_validate(train_cfg_yaml)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"predict_{name}_{timestamp}"
    data_logger = DataLogger(name=run_name)
    data_logger.log_cfg(OmegaConf.to_yaml(cfg_dict))

    data_module = DataModule(
        dataset_cfg=train_cfg.datasets,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = VioModule.load_from_checkpoint(
        f"{cfg.path}/{cfg.checkpoint}",
        data_logger=data_logger,
        cfg=train_cfg.model,
    )
    model.eval()

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        logger=data_logger,
        enable_progress_bar=True,
        limit_predict_batches=cfg.limit_batches if cfg.limit_batches != -1 else None,
    )

    trainer.predict(model, datamodule=data_module)


if __name__ == "__main__":
    main()
