import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from autoencoder.enums import DataSplit


class Comma10k(Dataset):
    def __init__(
        self,
        crop_height: int,
        crop_width: int,
        scale: float,
        root_dir: str,
        data_split: DataSplit,
    ):
        self.data_split = data_split
        all_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".png")]
        # Use all files ending with "9" as val dataset
        if self.data_split == DataSplit.TRAIN:
            self.files = [f for f in all_files if os.path.basename(f)[-5] != "9"]
        else:
            self.files = [f for f in all_files if os.path.basename(f)[-5] == "9"]

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
