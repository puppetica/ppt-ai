import os
from typing import Any

import hydra
import pytorch_lightning as pl
import torchvision.utils as vutils
from omegaconf import DictConfig, OmegaConf

from autoencoder.configs.config import PredictCfg
from autoencoder.data.data_module import DataModule
from autoencoder.module import AutoEncoder


class PostProcess(pl.Callback):
    def __init__(self, cfg: PredictCfg):
        self.cfg = cfg

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AutoEncoder,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        batch = batch.cpu()
        outputs = outputs.cpu()

        for i in range(batch.size(0)):
            inp_img = batch[i]
            out_img = outputs[i]

            vutils.save_image(inp_img, f"outputs/input_{i}.png")
            vutils.save_image(out_img, f"outputs/recon_{i}.png")


@hydra.main(config_path="configs", config_name="predict", version_base=None)
def main(cfg_dict: DictConfig):
    # Cast DictConfig → Pydantic for typing & validation
    cfg = PredictCfg.model_validate(cfg_dict)
    print("\n" + OmegaConf.to_yaml(cfg_dict) + "\n")

    ckpt_path = "checkpoints/last.ckpt"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    data_module = DataModule(
        cfg.datasets,
        batch_size=cfg.batch_size,
        num_workers=cfg.batch_size,
    )
    model = AutoEncoder.load_from_checkpoint(ckpt_path)
    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        callbacks=[PostProcess(cfg)],
    )
    trainer.predict(model, data_module, return_predictions=False)


if __name__ == "__main__":
    main()
