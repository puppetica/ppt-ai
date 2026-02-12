import os
from collections.abc import Iterable, Iterator
from typing import Any

import av
import torch

from common.frame_buffer import FrameBuffer
from common.img_processor import ImgProcessorFactory
from common.mcap_merge import merged_messages
from vio.data.augmentation import (  # TODO: Combine augmentation with the imgprocessfactory
    apply_img_augmentations,
    gen_aug_set,
)
from vio.data.batch_data import Batch, CameraBatch


class ImuMotionFilter:
    def __init__(self, deadband=0.05, alpha=0.1, total_alpha=0.1, device="cpu", dtype=torch.float32):
        self.deadband = deadband
        self.alpha = alpha  # smoothing for per-sample mag
        self.total_alpha = total_alpha  # smoothing for final total
        self.ma = None  # per-sample EMA
        self.total_ma = None  # EMA over total between windows
        self.device = device
        self.dtype = dtype

    def __call__(self, imu_list: list[torch.Tensor]) -> torch.Tensor:
        """Calculate distance travelled in meters based on an imu window list

        Args:
            imu_list (list[torch.Tensor]): list of torch tensors which include [wx, wy, wz, ax, ay, az, t]

        Returns:
            torch.Tensor: Resulting travel distance after 2 integrations
        """
        total = torch.zeros(1, dtype=self.dtype, device=self.device)

        # per-sample smoothing
        for i in range(len(imu_list) - 1):
            s0 = imu_list[i]
            s1 = imu_list[i + 1]

            dt = s1[-1] - s0[-1]
            a = s0[3:6]

            mag = torch.clamp(a.norm() - self.deadband, min=0.0)

            if self.ma is None:
                self.ma = mag.clone()
            else:
                self.ma = self.alpha * mag + (1 - self.alpha) * self.ma

            total = total + self.ma * dt

        # smooth final window result
        if self.total_ma is None:
            self.total_ma = total.clone()
        else:
            self.total_ma = self.total_alpha * total + (1 - self.total_alpha) * self.total_ma

        return self.total_ma


class McapStreamer(Iterable[Batch]):
    def __init__(
        self,
        target_height: int,
        target_width: int,
        crop_top: int,
        crop_bottom: int,
        scale: float,
        mcap_path: str,
        do_aug: bool,
    ):
        self.do_aug = do_aug
        self.mcap_path = mcap_path
        self.seq_name = os.path.splitext(os.path.basename(self.mcap_path))[0]
        self.img_processor_factory = ImgProcessorFactory(crop_top, crop_bottom, scale, target_height, target_width)
        self.aug_cfg = gen_aug_set()

        self.img_processor = None
        self.img_decoder: dict[str, Any] = {}
        self.buffers: dict[str, FrameBuffer] = {}
        self.imu_filter = ImuMotionFilter()

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
        self.min_master_size = (
            self.aug_cfg.past_frame + 1 if self.do_aug else 2
        )  # minimum buffer size of the master topic
        self.sync_topics = ["/imu"]

    def _decode_img(self, msg):
        img_bytes = msg.proto_msg.data
        if msg.topic not in self.img_decoder:
            self.img_decoder[msg.topic] = av.CodecContext.create("h264", "r")
        frames = self.img_decoder[msg.topic].decode(av.Packet(img_bytes))
        assert len(frames) == 1
        img = frames[0].to_ndarray(format="rgb24")  # HxWx3 numpy
        img = apply_img_augmentations(img, self.aug_cfg)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # CxHxW
        return img

    def _gen_frame(self) -> Batch:
        # Front img t
        ts_ns, msg = self.buffers["/cam/front0/image"].get(0)
        img = self._decode_img(msg)
        _, msg = self.buffers["/cam/front0/calibration"].get_by_ts(ts_ns)
        K = torch.tensor(msg.proto_msg.K).reshape(3, 3)
        if self.img_processor is None:
            self.img_processor = self.img_processor_factory.create(img.shape[1], img.shape[2], self.do_aug)
        img_new, K_new = self.img_processor(img, K)
        cam_intr = torch.tensor([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])
        _, msg = self.buffers["/cam/front0/transform"].get_by_ts(ts_ns)
        t = msg.proto_msg.translation
        r = msg.proto_msg.rotation
        cam_ext = torch.tensor([t.x, t.y, t.z, r.x, r.y, r.z, r.w])
        # Front img t-1
        past_img_idx = 1 if not self.do_aug else self.aug_cfg.past_frame
        ts_ns_tm1, msg_tm1 = self.buffers["/cam/front0/image"].get(past_img_idx)
        img_tm1 = self._decode_img(msg_tm1)
        img_tm1, _ = self.img_processor(img_tm1, K)
        img_timediff_s = torch.tensor([(ts_ns - ts_ns_tm1) / 1e9])
        # IMU data between t and t-1
        imu_list: list[torch.Tensor] = []
        for imu_ts_ns, msg in self.buffers["/imu"]:
            if imu_ts_ns < ts_ns_tm1:
                break
            elif imu_ts_ns > ts_ns:
                continue
            av = msg.proto_msg.angular_velocity
            accel = msg.proto_msg.linear_acceleration
            imu_timediff_s = (ts_ns - imu_ts_ns) / 1e9
            imu_list.append(torch.tensor([av.x, av.y, av.z, accel.x, accel.y, accel.z, imu_timediff_s]))

        imu_dt = self.imu_filter(imu_list)

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
            imu_dt=imu_dt,
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
                got_sync_topic = {t: False for t in self.sync_topics}
