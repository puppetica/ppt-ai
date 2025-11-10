import os
from collections.abc import Iterable, Iterator
from typing import Any

import av
import torch

from common.frame_buffer import FrameBuffer
from common.img_processor import ImgProcessorFactory
from common.mcap_merge import merged_messages
from vio.data.batch_data import Batch, CameraBatch


class McapStreamer(Iterable[Batch]):
    def __init__(
        self,
        target_height: int,
        target_width: int,
        crop_top: int,
        crop_bottom: int,
        scale: float,
        mcap_path: str,
    ):
        self.mcap_path = mcap_path
        self.seq_name = os.path.splitext(os.path.basename(self.mcap_path))[0]
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

    def _gen_frame(self) -> Batch:
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
        img_timediff_s = torch.tensor([(ts_ns - ts_ns_tm1) / 1e9])
        # IMU data between t and t-1
        imu_list: list[torch.Tensor] = []
        for imug_ts_ns, msg in self.buffers["/imu"]:
            if ts_ns < ts_ns_tm1:
                break
            av = msg.proto_msg.angular_velocity
            accel = msg.proto_msg.linear_acceleration
            imu_timediff_s = (ts_ns - imug_ts_ns) / 1e9
            imu_list.append(torch.tensor([av.x, av.y, av.z, accel.x, accel.y, accel.z, imu_timediff_s]))

        pos = torch.zeros(3)
        vel = torch.zeros(3)
        for i in range(len(imu_list) - 1):
            s0 = imu_list[i]
            s1 = imu_list[i + 1]
            dt = s1[-1] - s0[-1]  # seconds
            a = s0[3:6]
            vel = vel + a * dt
            pos = pos + vel * dt

        # Translation magnitude in meters
        imu_dt = pos.norm().unsqueeze(0)  # shape [1]

        return Batch(
            seq_name=[self.seq_name],
            input_mcap_path=[self.mcap_path],
            ts_ns=[ts_ns],
            cam_front=CameraBatch(
                img=img_new.unsqueeze(0),
                img_tm1=img_tm1.unsqueeze(0),
                intr=cam_intr.unsqueeze(0),
                extr=cam_ext.unsqueeze(0),
                timediff_s=img_timediff_s.unsqueeze(0),
            ),
            imu=[imu_list],
            imu_dt=imu_dt.unsqueeze(0),
            gt_ego_motion=torch.tensor([0.0]),
            gt_ego_motion_valid=torch.tensor([0.0]),
        )

    def __iter__(self) -> Iterator[Batch]:
        got_sync_topic = {t: False for t in self.sync_topics}
        frame_ts_ns: int | None = None
        for ts, msg in merged_messages([self.mcap_path], self.topics):
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
                yield self._gen_frame()
                frame_ts_ns = None
                got_sync_topic[msg.topic] = False
