import os

import torch
import torchvision.utils as vutils
from pytorch_lightning.loggers import Logger


class DataLogger(Logger):
    def __init__(self, name: str, save_dir: str = ".logger"):
        super().__init__()
        self._save_dir = os.path.join(save_dir, name)
        self._name = name
        os.makedirs(self._save_dir, exist_ok=False)
        self.metrics_file = open(os.path.join(self._save_dir, "metrics.txt"), "a")

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return "1.0"

    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step):
        line = f"[Step {step}] {metrics}\n"
        self.metrics_file.write(line)
        self.metrics_file.flush()

    def log_image(self, image, name: str, epoch: int):
        if isinstance(image, torch.Tensor):
            grid = vutils.make_grid(image)
            os.makedirs(f"{self.save_dir}/{epoch}", exist_ok=True)
            path = f"{self.save_dir}/{epoch}/{name}"
            vutils.save_image(grid, path)
        else:
            # handle other formats (PIL, etc.)
            pass

    def log_cfg(self, cfg: str):
        path = f"{self.save_dir}/cfg.yaml"
        with open(path, "w") as f:
            f.write(cfg)
        # logger.info(f"Start Training: {name}")
        # logger.info("\n" + OmegaConf.to_yaml(cfg_dict) + "\n")

    def finalize(self, status):
        print(f"Finished with status: {status}")
