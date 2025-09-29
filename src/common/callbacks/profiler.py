import logging
import time

import torch
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger("profiler")


class ProfilerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_opt = self.get_synced_now()
        self.data_loading_start = self.get_synced_now()
        # summs for avg measurements over epochs
        self.sum_data_loading_time = 0
        self.sum_gpu_transfer_time = 0
        self.sum_gpu_opt_time = 0
        self.sum_full_batch_time = 0

    @staticmethod
    def get_ram_memory() -> float:
        """
        Get memory usage for the container.
        """
        mem_usage_bytes = -1
        try:
            # Find memory limit for container in "/sys/fs/cgroup/memory.max"
            with open("/sys/fs/cgroup/memory.current") as f:
                mem_usage_bytes = int(f.read().strip())
        except FileNotFoundError:
            logger.warning("Unable to find docker memory info")
            return -1
        return mem_usage_bytes / 1024**2

    @staticmethod
    def get_gpu_memory() -> float:
        torch.cuda.synchronize()
        gpu_mem_byte = torch.cuda.memory_allocated()
        return gpu_mem_byte / 1024**2  # Convert to MB

    @staticmethod
    def get_synced_now() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    def batch_start_info(self, pl_module, counter: int, prefix: str = ""):
        if counter == 0:
            return
        if not hasattr(pl_module.trainer.datamodule, "gpu_transfer_start"):
            msg = "Datamodule has no gpu_transfer_start member. You must implement it in on_before_batch_transfer()."
            logger.error(msg)
            raise NotImplementedError(msg)

        data_loading_time = pl_module.trainer.datamodule.gpu_transfer_start - self.data_loading_start
        self.sum_data_loading_time += data_loading_time
        avg = self.sum_data_loading_time / float(counter)
        logger.info(f"{prefix} Data loading: {data_loading_time:.2f}s, Avg: {avg:.2f}s")
        transfer_time = self.get_synced_now() - pl_module.trainer.datamodule.gpu_transfer_start
        self.sum_gpu_transfer_time += transfer_time
        avg = self.sum_gpu_transfer_time / float(counter)
        logger.info(f"{prefix} Data GPU transfer: {transfer_time:.2f}s, Avg: {avg:.2f}s")
        self.start_opt = self.get_synced_now()

    def batch_end_info(self, counter: int, prefix: str = ""):
        if counter == 0:
            return
        gpu_opt_time = self.get_synced_now() - self.start_opt
        self.sum_gpu_opt_time += gpu_opt_time
        avg = self.sum_gpu_opt_time / float(counter)
        logger.info(f"{prefix} Batch training (GPU): {gpu_opt_time:.2f}s, Avg: {avg:.2f}s")
        full_batch_time = self.get_synced_now() - self.data_loading_start
        self.sum_full_batch_time += full_batch_time
        avg = self.sum_full_batch_time / float(counter)
        logger.info(f"  {prefix} Full Batch Runtime: {full_batch_time:.2f}s, Avg: {avg:.2f}s")
        logger.info(f"  {prefix} RAM: {self.get_ram_memory():.0f} MB, GPU Memory: {self.get_gpu_memory():.0f} MB")
        logger.info("- - - - - - - - - ")
        self.data_loading_start = self.get_synced_now()

    def reset_sums(self):
        self.sum_data_loading_time = 0
        self.sum_gpu_transfer_time = 0
        self.sum_gpu_opt_time = 0
        self.sum_full_batch_time = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx < 1:
            return
        self.batch_start_info(pl_module, batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 1:
            self.data_loading_start = self.get_synced_now()
            return
        self.batch_end_info(batch_idx)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            return
        self.batch_start_info(pl_module, batch_idx, "[VAL]")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self.data_loading_start = self.get_synced_now()
            return
        self.batch_end_info(batch_idx, "[VAL]")

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            return
        self.batch_start_info(pl_module, batch_idx, "[PREDICT]")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self.data_loading_start = self.get_synced_now()
            return
        self.batch_end_info(batch_idx, "[PREDICT]")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset_sums()

    def on_train_epoch_start(self, trainer, pl_module):
        self.reset_sums()

    def on_predict_epoch_start(self, trainer, pl_module):
        self.reset_sums()
