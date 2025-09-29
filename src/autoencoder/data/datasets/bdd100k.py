import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from autoencoder.enums import DataSplit


class Bdd100k(Dataset):
    def __init__(
        self,
        crop_height: int,
        crop_width: int,
        scale: float,
        root_dir: str,
        data_split: DataSplit,
    ):
        self.data_split = data_split
        img_root_path = os.path.join(root_dir, "val")
        if data_split == DataSplit.TRAIN:
            img_root_path = os.path.join(root_dir, "train")
        self.files = [os.path.join(img_root_path, f) for f in os.listdir(img_root_path) if f.endswith(".jpg")]

        self.transform = transforms.Compose(
            [
                transforms.CenterCrop((crop_height, crop_width)),
                transforms.Resize((int(crop_height // scale), int(crop_width // scale))),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
