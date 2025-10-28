import math
import os
import random
from typing import Any

import av
import torch
from torch.utils.data import IterableDataset, get_worker_info

from common.frame_buffer import FrameBuffer
from common.img_processor import ImgProcessorFactory
from common.mcap_merge import merged_messages
from vio.batch_data import Batch, CameraBatch
from vio.enums import DataSplit


class McapStreamer(IterableDataset):
    def __init__(
        self,
        target_height: int,
        target_width: int,
        crop_top: int,
        crop_bottom: int,
        scale: float,
        root_dir: str,
        data_split: DataSplit,
    ):
        self.data_split = data_split

        split_dir = "train" if data_split == DataSplit.TRAIN else "val"
        split_path = os.path.join(root_dir, split_dir)
        self.mcap_files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.lower().endswith(".mcap")]

        self.img_processor_factory = ImgProcessorFactory(crop_top, crop_bottom, scale, target_height, target_width)
        self.img_processor = None
        self.img_decoder: dict[str, Any] = {}
        self.buffers: dict[str, FrameBuffer] = {}

        self.topics = [
            # Sensor input
            "/cam/front0/image",
            "/cam/front0/calibration",
            "/cam/front0/transform",
            "/imu",
            # Gt/labels
            "/transform/global",
        ]

        # Define conditions for a synced frame
        self.master_topic = "/cam/front0/image"
        self.min_master_size = 2  # minimum buffer size of the master topic
        self.sync_topics = ["/imu"]

    def _decode_img(self, msg):
        img_bytes = msg.proto_msg.data
        if msg.topic not in self.img_decoder:
            self.img_decoder[msg.topic] = av.CodecContext.create("h264", "r")
        frames = self.img_decoder[msg.topic].decode(av.Packet(img_bytes))
        assert len(frames) == 1
        img = frames[0].to_ndarray(format="rgb24")  # HxWx3 numpy
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # CxHxW
        return img

    def _ns_to_s_norm(self, ts_ns: int):
        if self.norm_ts_ns is None:
            raise ValueError("Timestamp of first frame not set!")
        normalized_ts_ns = ts_ns - self.norm_ts_ns
        normalized_ts_s = normalized_ts_ns / 1e9
        return normalized_ts_s

    def _gen_frame(self, seq_name: str):
        # Front img t
        ts_ns, msg = self.buffers["/cam/front0/image"].get(0)
        img = self._decode_img(msg)
        _, msg = self.buffers["/cam/front0/calibration"].get_by_ts(ts_ns)
        K = torch.tensor(msg.proto_msg.K).reshape(3, 3)
        if self.img_processor is None:
            self.img_processor = self.img_processor_factory.create(img.shape[1], img.shape[2])
        img_new, K_new = self.img_processor(img, K)
        cam_intr = torch.tensor([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])
        _, msg = self.buffers["/cam/front0/transform"].get_by_ts(ts_ns)
        t = msg.proto_msg.translation
        r = msg.proto_msg.rotation
        cam_ext = torch.tensor([t.x, t.y, t.z, r.x, r.y, r.z, r.w])
        # Front img t-1
        ts_ns_tm1, msg_tm1 = self.buffers["/cam/front0/image"].get(1)
        img_tm1 = self._decode_img(msg_tm1)
        img_tm1, _ = self.img_processor(img_tm1, K)
        img_ts_sec = torch.tensor([self._ns_to_s_norm(ts_ns), self._ns_to_s_norm(ts_ns_tm1)])
        # IMU data between t and t-1
        imu_list: list[list[torch.Tensor]] = []
        imu_list.append([])
        for ts_ns, msg in self.buffers["/imu"]:
            if ts_ns < ts_ns_tm1:
                break
            av = msg.proto_msg.angular_velocity
            accel = msg.proto_msg.linear_acceleration
            ts_sec = self._ns_to_s_norm(ts_ns)
            imu_list[0].append(torch.tensor([av.x, av.y, av.z, accel.x, accel.y, accel.z, ts_sec]))

        return Batch(
            seq_name=[seq_name],
            cam_front=CameraBatch(
                img=img_new,
                img_tm1=img_tm1,
                intr=cam_intr,
                extr=cam_ext,
                ts_ns=img_ts_sec,
            ),
            imu=imu_list,
            gt_ego_motion=torch.tensor([0.0]),
            gt_ego_motion_valid=torch.tensor([0.0]),
        )

    def __iter__(self):
        mcap_files_set = self.mcap_files.copy()

        # Support multi processing
        worker_info = get_worker_info()
        if worker_info is not None:
            n = len(self.mcap_files)
            per_worker = int(math.ceil(n / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, n)
            mcap_files_set = self.mcap_files[start:end]

        if self.data_split == DataSplit.TRAIN:
            random.shuffle(mcap_files_set)  # shuffle file order once per epoch

        for mcap_path in mcap_files_set:
            self.img_processor = None
            self.norm_ts_ns = None
            got_sync_topic = {t: False for t in self.sync_topics}
            self.img_decoder = {}
            frame_ts_ns: int | None = None
            for ts, msg in merged_messages([mcap_path], self.topics):
                # Take timestamp of very first frame for normalization (avoid large timestamp numbers)
                if self.norm_ts_ns is None:
                    self.norm_ts_ns = ts
                # Save msg to buffer
                if msg.topic not in self.buffers:
                    self.buffers[msg.topic] = FrameBuffer(msg.topic)
                self.buffers[msg.topic].add((ts, msg))
                # Record if we got a sync message
                if msg.topic in self.sync_topics:
                    got_sync_topic[msg.topic] = True
                # Check if we got enough to generate a frame_ts after the current timestamp, set the frame_ts_ns
                if (
                    msg.topic == self.master_topic
                    and all(got_sync_topic.values())
                    and len(self.buffers[self.master_topic]) >= self.min_master_size
                ):
                    frame_ts_ns = ts
                # Check if we got all needed data and are finished reading with all data of that timestamp
                if frame_ts_ns is not None and ts > frame_ts_ns:
                    yield self._gen_frame(mcap_path)
                    frame_ts_ns = None
