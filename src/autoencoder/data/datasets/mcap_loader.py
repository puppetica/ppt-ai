import math
import os
import random

import torch
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from torch.utils.data import IterableDataset, get_worker_info
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
    ):
        self.data_split = data_split
        self.topics = topics

        # pick train or val
        split_dir = "train" if data_split == DataSplit.TRAIN else "val"
        split_path = os.path.join(root_dir, split_dir)
        self.mcap_files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.lower().endswith(".mcap")]

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

    def count_data(self) -> int:
        num_samples = 0
        for mcap_file in self.mcap_files:
            with open(mcap_file, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                summary = reader.get_summary()
                assert summary is not None and summary.statistics is not None
                for cid, channel in summary.channels.items():
                    if channel.topic in self.topics:
                        self.num_samples += summary.statistics.channel_message_counts.get(cid, 0)
        return num_samples

    def __iter__(self):
        mcap_files = self.mcap_files.copy()

        # Support multi processing
        worker_info = get_worker_info()
        if worker_info is not None:
            n = len(self.mcap_files)
            per_worker = int(math.ceil(n / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, n)
            mcap_files = self.mcap_files[start:end]

        if self.data_split == DataSplit.TRAIN:
            random.shuffle(mcap_files)  # shuffle file order once per epoch

        frame_idx = 0
        for mcap_file in mcap_files:
            with open(mcap_file, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                for schema, channel, message, proto_msg in reader.iter_decoded_messages(topics=self.topics):
                    if frame_idx % 5 != 0:  # downsample from 10Hz to 2Hz
                        frame_idx += 1
                        continue

                    img_bytes = proto_msg.data
                    img = io.decode_image(
                        torch.tensor(bytearray(img_bytes), dtype=torch.uint8),
                        mode=io.ImageReadMode.RGB,
                    )
                    img = self.transform(img)
                    yield img
                    frame_idx += 1
