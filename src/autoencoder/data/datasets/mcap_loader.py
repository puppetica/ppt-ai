import os
import random

import torch
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from torch.utils.data import IterableDataset
from torchvision import io, transforms

from autoencoder.enums import DataSplit


class McapImgLoader(IterableDataset):
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
        buffer_size: int = 1000,
    ):
        self.data_split = data_split
        self.buffer_size = buffer_size
        self.topics = topics

        # pick train or val
        split_dir = "train" if data_split == DataSplit.TRAIN else "val"
        split_path = os.path.join(root_dir, split_dir)
        self.mcap_files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.lower().endswith(".mcap")]
        # Count all the channels for __len__()
        self.num_samples = 0
        for mcap_file in self.mcap_files:
            with open(mcap_file, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                summary = reader.get_summary()
                assert summary is not None and summary.statistics is not None
                for cid, channel in summary.channels.items():
                    if channel.topic in topics:
                        self.num_samples += summary.statistics.channel_message_counts.get(cid, 0)

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
        return self.num_samples

    def __iter__(self):
        buffer = []

        for mcap_file in self.mcap_files:
            with open(mcap_file, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                for schema, channel, message, proto_msg in reader.iter_decoded_messages(topics=self.topics):
                    img_bytes = proto_msg.data
                    img = io.decode_image(
                        torch.frombuffer(img_bytes, dtype=torch.uint8),
                        mode=io.ImageReadMode.RGB,
                    )

                    img = self.transform(img)

                    buffer.append(img)
                    if len(buffer) >= self.buffer_size:
                        idx = random.randrange(len(buffer)) if self.data_split == DataSplit.TRAIN else 0
                        yield buffer.pop(idx)

        # Drain remaining buffer
        while buffer:
            idx = random.randrange(len(buffer)) if self.data_split == DataSplit.TRAIN else 0
            yield buffer.pop()
