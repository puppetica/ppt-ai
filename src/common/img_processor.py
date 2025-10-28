import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F


def crop_top_bottom_with_K(crop_top: int, crop_bottom: int):
    def f(img: torch.Tensor, K: torch.Tensor):
        img = img[:, crop_top : img.shape[1] - crop_bottom, :]
        K[1, 2] -= crop_top
        return img, K

    return f


def random_or_center_crop_params(h_in: int, w_in: int, target_h: int, target_w: int, random_crop: bool):
    if random_crop:
        if h_in == target_h and w_in == target_w:
            return 0, 0, target_h, target_w
        top = random.randint(0, h_in - target_h)
        left = random.randint(0, w_in - target_w)
    else:
        top = int(round((h_in - target_h) / 2.0))
        left = int(round((w_in - target_w) / 2.0))
    return top, left, target_h, target_w


def fixed_crop_with_K(top, left, h, w):
    def f(img: torch.Tensor, K: torch.Tensor):
        img = F.crop(img, top, left, h, w)
        K[0, 2] -= left
        K[1, 2] -= top
        return img, K

    return f


def resize_with_K(scale):
    def f(img: torch.Tensor, K: torch.Tensor):
        in_h, in_w = img.shape[-2:]
        out_h, out_w = int(in_h // scale), int(in_w // scale)
        img = F.resize(img, [out_h, out_w])
        K[0] *= out_w / in_w
        K[1] *= out_h / in_h
        return img, K

    return f


class ComposeWithK:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            img, K = t(img, K)
        return img, K


class ImgProcessorFactory:
    def __init__(
        self,
        crop_top: int,
        crop_bottom: int,
        scale: float,
        target_h: int,
        target_w: int,
    ):
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.scale = scale
        self.target_h = target_h
        self.target_w = target_w

    def create(self, img_h: int, img_w: int, random_aug: bool = False) -> ComposeWithK:
        i, j, h, w = random_or_center_crop_params(
            int(img_h // self.scale),
            int(img_w // self.scale),
            self.target_h,
            self.target_w,
            random_aug,
        )
        transforms_list = [
            crop_top_bottom_with_K(self.crop_top, self.crop_bottom),
            resize_with_K(self.scale),
            fixed_crop_with_K(i, j, h, w),
            lambda img, K: (transforms.ConvertImageDtype(torch.float32)(img), K),
        ]
        return ComposeWithK(transforms_list)
