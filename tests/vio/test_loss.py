import math

import cv2
import numpy as np
import pytest
import torch
from torchvision.io import read_image

from src.vio.loss import vio_loss


class DummyBatch(dict):
    pass


@pytest.fixture
def make_batch():
    def _make():
        # Load real images
        img0 = read_image("assets/test_cam_front_0.png").float() / 255.0  # [3,H,W]
        img1 = read_image("assets/test_cam_front_1.png").float() / 255.0
        batch = DummyBatch()
        batch["cam_front"] = {
            "img": img0[:3].unsqueeze(0),  # [1,3,H,W]
            "img_tm1": img1[:3].unsqueeze(0),  # [1,3,H,W]
            "intr": torch.tensor([[141.31375, 141.31375, 320.20143, 242.32710]], dtype=torch.float32),
            "extr": torch.tensor(
                [[0.010168, 0.0208, -0.0007491, 0.48848, -0.5049, 0.5107, -0.4954]], dtype=torch.float32
            ),
            "timediff_s": torch.tensor([[0.033]]),
        }
        batch["imu_dt"] = torch.tensor([0.275])

        return batch

    return _make


def depth_to_logdepth_raw(depth, min_depth=0.1, max_depth=80.0):
    depth = depth.clamp(min_depth, max_depth)

    log_min = math.log(min_depth)
    log_max = math.log(max_depth)

    # log depth
    log_depth = depth.log()

    # normalize to [0,1]
    d_norm = (log_depth - log_min) / (log_max - log_min)

    # to [-1,1]
    t = d_norm * 2.0 - 1.0

    # numerically safe clamp
    eps = 1e-6
    t = t.clamp(-1 + eps, 1 - eps)

    # invert tanh
    d_raw = 0.5 * torch.log((1 + t) / (1 - t))
    return d_raw


def test_vio_loss_runs(make_batch):
    batch = make_batch()

    _, _, H, W = batch["cam_front"]["img"].shape
    H2, W2 = H // 2, W // 2

    min_depth = 0.1
    max_depth = 80.0

    # bottom = near, top = far  ← this is flipped now
    v = torch.linspace(max_depth, min_depth, H2).view(1, 1, H2, 1)
    depth_map = v.repeat(1, 1, 1, W2)

    depth_map = depth_to_logdepth_raw(depth_map)

    pred_pose = torch.tensor([[0.217283, -0.077740, 0.073410, -0.003269, 0.001573, 0.005826, 0.999718]])
    photometric, smooth_loss, reg_pose_loss, scale_loss, recon_img = vio_loss(batch, depth_map, pred_pose)

    tensor = recon_img[0]  # [3,H,W]
    img = tensor.permute(1, 2, 0).cpu().numpy()  # HWC RGB
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("recon.png", img_bgr)

    assert photometric.ndim == 0
    assert smooth_loss.ndim == 0
    assert recon_img.dtype == torch.float32
