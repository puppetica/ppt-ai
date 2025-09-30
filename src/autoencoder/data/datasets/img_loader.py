import os
import random

import torch
from torch.utils.data import IterableDataset
from torchvision import io, transforms

from autoencoder.enums import DataSplit


class ImgLoader(IterableDataset):
    def __init__(
        self,
        target_height: int,
        target_width: int,
        crop_bottom: int,
        crop_top: int,
        scale: float,
        root_dir: str,
        data_split: DataSplit,
    ):
        self.data_split = data_split
        img_root_path = os.path.join(root_dir, "train" if data_split == DataSplit.TRAIN else "val")
        self.files = [
            os.path.join(img_root_path, f)
            for f in os.listdir(img_root_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Crop function directly on tensors
        def crop_top_bottom_tensor(tensor):
            return tensor[:, crop_top : tensor.shape[1] - crop_bottom, :]

        if data_split == DataSplit.TRAIN:
            crop = transforms.RandomCrop((target_height, target_width))
        else:
            crop = transforms.CenterCrop((target_height, target_width))

        self.transform = transforms.Compose(
            [
                transforms.Lambda(crop_top_bottom_tensor),
                crop,
                transforms.Resize((int(target_height // scale), int(target_width // scale))),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

    def count_data(self) -> int:
        return len(self.files)

    def __iter__(self):
        files = self.files.copy()
        if self.data_split == DataSplit.TRAIN:
            random.shuffle(files)
        files = files[:10]
        for path in files:
            img = io.read_image(path)
            img = self.transform(img)
            yield img
