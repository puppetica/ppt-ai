import os

import torch
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from torch.utils.data import Dataset
from torchvision import io, transforms

from autoencoder.enums import DataSplit


class McapImgLoader(Dataset):
    def __init__(
        self,
        target_height: int,
        target_width: int,
        crop_bottom: int,
        crop_top: int,
        scale: float,
        root_dir: str,
        data_split: DataSplit,
        topics: list[str],
    ):
        self.data_split = data_split

        # pick train or val
        split_dir = "train" if data_split == DataSplit.TRAIN else "val"
        self.mcap_files = [
            os.path.join(root_dir, split_dir, f)
            for f in os.listdir(os.path.join(root_dir, split_dir))
            if f.lower().endswith(".mcap")
        ]

        self.topics = topics
        self.samples = []  # (mcap_file, topic, log_time, image_bytes)

        # Index images from MCAPs
        for mcap_file in self.mcap_files:
            with open(mcap_file, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                for schema, channel, message, proto_msg in reader.iter_decoded_messages(topics=self.topics):
                    self.samples.append((mcap_file, channel.topic, message.log_time, proto_msg.data))

        def crop_top_bottom_tensor(tensor: torch.Tensor):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, topic, ts, img_bytes = self.samples[idx]

        img = io.decode_image(torch.frombuffer(img_bytes, dtype=torch.uint8), mode=io.ImageReadMode.RGB)

        # Apply transforms
        img = self.transform(img)

        return img
