import io
import os

import tfrecord
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.datasets.waymo_protos import dataset_pb2 as open_dataset
from enums import DataSplit


class WaymoPerc(Dataset):
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


import matplotlib.pyplot as plt

# lightweight TFRecord reader
# Import your compiled proto (dataset_pb2.py)


def read_images_from_tfrecord(tfrecord_path, show=True):
    """
    Read images from a Waymo TFRecord file.
    """
    for record in tfrecord.tfrecord_loader(tfrecord_path):
        frame = open_dataset.Frame()
        frame.ParseFromString(record)

        print(f"Frame has {len(frame.images)} camera images")

        for i, image in enumerate(frame.images):
            # Decode JPEG-encoded bytes
            img = Image.open(io.BytesIO(image.image))

            if show:
                plt.imshow(img)
                cam_name = open_dataset.CameraName.Name.Name(image.name)
                plt.title(f"Camera {cam_name}")
                plt.axis("off")
                plt.show()

        # Just read the first frame for demo
        break


if __name__ == "__main__":
    tfrecord_path = "path/to/segment-xxxxx.tfrecord"
    read_images_from_tfrecord(tfrecord_path)
