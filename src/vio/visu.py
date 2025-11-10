import numpy as np
from scipy.spatial.transform import Rotation as R

from interface.python.foxglove.FrameTransform_pb2 import FrameTransform
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


def vis_rel_transform(ts_ns: int, rel_pose: np.ndarray, last_pose: FrameTransform | None) -> FrameTransform:
    """Create a FrameTransform from a rel pose and the previous transform

    Args:
        ts_ns (int): Current timestamp in ns
        rel_pose (np.ndarray): Relative pose as [x, y, z, qx, qy, qz, qw]
        last_pose (FrameTransform): Last known pose as a FrameTransform

    Returns:
        FrameTransform: _description_
    """
    msg = FrameTransform()
    msg.timestamp.FromNanoseconds(ts_ns)
    msg.parent_frame_id = "global_pred"
    msg.child_frame_id = "ego_pred"

    if last_pose is None:
        last_pose = FrameTransform()
        last_pose.rotation.x = 0
        last_pose.rotation.y = 0
        last_pose.rotation.z = 0
        last_pose.rotation.w = 1
        last_pose.translation.x = 0
        last_pose.translation.y = 0
        last_pose.translation.z = 0

    # last orientation
    q_last = R.from_quat(
        [
            last_pose.rotation.x,
            last_pose.rotation.y,
            last_pose.rotation.z,
            last_pose.rotation.w,
        ]
    )

    # relative orientation
    qx, qy, qz, qw = rel_pose[3:7]
    q_rel = R.from_quat([qx, qy, qz, qw])

    # compose
    q_new = q_last * q_rel
    qx_new, qy_new, qz_new, qw_new = q_new.as_quat()

    # new translation
    t_rel = rel_pose[:3]
    last_t = np.array([last_pose.translation.x, last_pose.translation.y, last_pose.translation.z])
    t_new = last_t + q_last.apply(t_rel)

    msg.rotation.x = qx_new
    msg.rotation.y = qy_new
    msg.rotation.z = qz_new
    msg.rotation.w = qw_new
    msg.translation.x = t_new[0]
    msg.translation.y = t_new[1]
    msg.translation.z = t_new[2]

    return msg
