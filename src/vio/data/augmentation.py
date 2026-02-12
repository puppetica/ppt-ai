import random
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AugmentorCfg:
    past_frame: int = 1
    brightness: float = 1.0
    contrast: float = 1.0
    blur: float = 0.0
    noise: float = 0.0
    sharpe: float = 0.0


def gen_aug_set(
    past_frame_range=(1, 4),
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    blur_range=(0.0, 3.0),
    noise_range=(0.0, 0.05),
    sharpen_range=(0.0, 1.0),
):
    return AugmentorCfg(
        past_frame=random.randint(*past_frame_range),
        brightness=random.uniform(*brightness_range),
        contrast=random.uniform(*contrast_range),
        blur=random.uniform(*blur_range),
        noise=random.uniform(*noise_range),
        sharpe=random.uniform(*sharpen_range),
    )


def apply_img_augmentations(img, cfg: AugmentorCfg):
    out = img.astype(np.float32)

    # brightness (scale pixel values)
    out = out * cfg.brightness

    # contrast (simple linear scaling around mean)
    mean = out.mean()
    out = (out - mean) * cfg.contrast + mean

    # blur
    if cfg.blur > 0:
        k = int(max(1, round(cfg.blur)))
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), 0)

    # noise
    if cfg.noise > 0:
        sigma = cfg.noise * 255
        noise = np.random.randn(*out.shape) * sigma
        out = out + noise

    # sharpen
    if cfg.sharpe > 0:
        alpha = cfg.sharpe
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharp = cv2.filter2D(out, -1, kernel)
        out = (1 - alpha) * out + alpha * sharp

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
