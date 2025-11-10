import numpy as np

from interface.python.foxglove.RawImage_pb2 import RawImage


def vis_depth_img(ts_ns: int, depth: np.ndarray) -> RawImage:
    max_d = 255.0  # meters
    img_u8 = 255.0 * (1.0 - depth / max_d)
    img_u8 = np.clip(img_u8, 0.0, 255.0).astype(np.uint8, copy=False)
    img_u8 = np.ascontiguousarray(img_u8)

    _, H, W = img_u8.shape

    msg = RawImage()
    msg.frame_id = "/cam/front0"
    msg.timestamp.FromNanoseconds(ts_ns)
    msg.width = W
    msg.height = H
    msg.encoding = "mono8"
    msg.step = W
    msg.data = img_u8.tobytes()

    return msg
