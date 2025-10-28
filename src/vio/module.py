import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from common.logging.data_logger import DataLogger
from vio.batch_data import Batch
from vio.configs.config import ModelCfg
from vio.loss import photometric_loss
from vio.model.cam_encoder import CamEncoder
from vio.model.depth_decoder import DepthDecoder
from vio.model.imu_encoder import ImuEncoder
from vio.model.pose_estimator import PoseEstimator


class VioModule(pl.LightningModule):
    def __init__(self, data_logger: DataLogger, cfg: ModelCfg):
        super().__init__()
        self.lr = cfg.lr
        self.data_logger = data_logger

        self.cam_encoder = CamEncoder(cfg.cam.in_ch, cfg.cam.res_block_ch, cfg.cam.num_attn_blocks)
        self.depth_decoder = DepthDecoder(cfg.depth.res_block_ch, cfg.depth.num_attn_blocks)
        self.imu_encoder = ImuEncoder(cfg.imu.token_dim, cfg.imu.hidden, cfg.imu.num_token)
        self.pose_estimator = PoseEstimator(
            cam_in_dim=cfg.cam.res_block_ch[-1],
            token_dim=cfg.imu.token_dim,
        )

    def forward(self, batch: Batch):
        img_t = batch["cam_front"]["img"]  # [B,3,H,W]
        img_tm1 = batch["cam_front"]["img_tm1"]  # [B,3,H,W]

        cam_feat = self.cam_encoder(img_t, img_tm1)  # [B,C,Hb,Wb]
        depth = self.depth_decoder(cam_feat)  # [B,1,Hd,Wd]

        imu_list = batch["imu"]
        imu_tokens = self.imu_encoder(imu_list)  # [B,K,D]

        intr = batch["cam_front"]["intr"]  # [B,4]
        extr = batch["cam_front"]["extr"]  # [B,7]
        ts = batch["cam_front"]["ts_sec"]  # [B,2]
        delta_t = ts[:, 0] - ts[:, 1]  # [B]

        pose = self.pose_estimator(
            cam_feat,
            imu_tokens,
            intr,
            extr,
            delta_t,
        )  # [B,7]

        return {"pose": pose, "depth": depth}

    def training_step(self, batch: Batch, batch_idx):
        out = self.forward(batch)
        photo_loss, smooth_loss, _ = photometric_loss(batch, out["depth"], out["pose"])
        loss = photo_loss + 1e-3 * smooth_loss
        self.log_dict(
            {"train_photo": photo_loss, "train_smooth": smooth_loss, "train_loss": loss},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx):
        out = self.forward(batch)
        depth = out["depth"]
        photo_loss, smooth_loss, recon = photometric_loss(batch, depth, out["pose"])
        loss = photo_loss + 1e-3 * smooth_loss
        self.log_dict(
            {"val_photo": photo_loss, "val_smooth": smooth_loss, "val_loss": loss},
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        if batch_idx in [0, 400, 1000, 2000, 4000] and hasattr(self.logger, "log_image"):
            img_t = batch["cam_front"]["img"][0]  # [3,H,W]
            _, H, W = img_t.shape
            recon_t = recon[0]  # [3,H,W]
            depth_viz = depth[0] / (depth[0].max() + 1e-6)
            depth_viz = F.interpolate(depth_viz.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)[0]

            preview = torch.cat([img_t, recon_t, depth_viz.repeat(3, 1, 1)], dim=-1)
            self.data_logger.log_image(
                preview,
                name=f"preview_e{self.current_epoch:03d}_b{batch_idx:04d}.png",
                epoch=self.current_epoch,
            )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
