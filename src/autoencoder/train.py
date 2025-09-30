import logging
from datetime import datetime

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.callback import Callback

from autoencoder.configs.config import TrainCfg
from autoencoder.data.data_module import DataModule
from autoencoder.module import AutoEncoder
from common.callbacks.profiler import ProfilerCallback
from common.logging.data_logger import DataLogger

logger = logging.getLogger("autoencoder.train")


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg_dict: DictConfig):
    # Get config and generate name for the run
    cfg = TrainCfg.model_validate(cfg_dict)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = f"{cfg.name}_{timestamp}"

    data_logger = DataLogger(name=name)
    data_logger.log_cfg(OmegaConf.to_yaml(cfg_dict))

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=data_logger.save_dir,
            filename=f"{name}_latest",
            save_last=True,
            save_top_k=0,
            every_n_epochs=1,
        )
    ]
    if cfg.run_profiler:
        callbacks.append(ProfilerCallback())

    data_module = DataModule(
        cfg.datasets,
        batch_size=cfg.batch_size,
        num_workers=cfg.batch_size,
    )
    model = AutoEncoder(
        lr=cfg.model.lr,
        in_ch=cfg.model.in_ch,
        res_block_ch=cfg.model.res_block_ch,
        num_attn_blocks=cfg.model.num_attn_blocks,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        callbacks=callbacks,
        enable_progress_bar=False if cfg.run_profiler else True,
        logger=data_logger,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
