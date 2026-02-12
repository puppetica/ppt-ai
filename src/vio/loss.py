import math

import torch
import torch.nn.functional as F

from vio.data.batch_data import Batch


# ----------------------------------------
# ---------- Matrix Utilities ------------
# ----------------------------------------
def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + 1e-8)


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    # q: [B,4] (qx,qy,qz,qw) -> R: [B,3,3]
    x, y, z, w = q.unbind(-1)
    B = q.shape[0]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).view(B, 3, 3)
    return R


def _se3_from_tq(t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # t: [B,3], q: [B,4] (qx,qy,qz,qw) -> [B,4,4]
    qn = _normalize_quat(q)
    R = _quat_to_rot(qn)
    T = torch.eye(4, device=t.device, dtype=t.dtype).unsqueeze(0).repeat(t.size(0), 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def _invert_se3(T: torch.Tensor) -> torch.Tensor:
    # NOTE: Faster closed inverse for a 4x4 transformation matrix
    # [B,4,4]
    R = T[:, :3, :3]
    t = T[:, :3, 3:4]
    RT = R.transpose(1, 2)
    Tinvt = -RT @ t
    Tout = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.size(0), 1, 1)
    Tout[:, :3, :3] = RT
    Tout[:, :3, 3] = Tinvt.squeeze(-1)
    return Tout


def _make_K(intr: torch.Tensor) -> torch.Tensor:
    # intr: [B,4] (fx,fy,cx,cy) -> K: [B,3,3]
    fx, fy, cx, cy = intr.unbind(-1)
    B = intr.shape[0]
    K = torch.zeros(B, 3, 3, device=intr.device, dtype=intr.dtype)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0
    return K


def _transform_points(T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # T: [B,4,4], X: [B,3,H,W] -> [B,3,H,W]
    B, _, H, W = X.shape
    Xh = torch.cat([X.view(B, 3, -1), torch.ones(B, 1, H * W, device=X.device, dtype=X.dtype)], dim=1)  # [B,4,HW]
    Yh = T @ Xh  # [B,4,HW]
    Y = Yh[:, :3, :].view(B, 3, H, W)
    return Y


def quat_to_euler_rad(q):
    # q: [B,4] = (x,y,z,w)
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    # roll (x axis)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    # pitch (y axis)
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    # yaw (z axis)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return roll + pitch + yaw


# ----------------------------------------
# -------- Warpping Utilities ------------
# ----------------------------------------
def _meshgrid_xy(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype), torch.arange(W, device=device, dtype=dtype), indexing="ij"
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,3,H,W]
    return pix


def _backproject(depth: torch.Tensor, Kinv: torch.Tensor) -> torch.Tensor:
    # depth: [B,1,H,W], Kinv: [B,3,3] -> X_cam: [B,3,H,W]
    B, _, H, W = depth.shape
    pix = _meshgrid_xy(B, H, W, depth.device, depth.dtype)  # [B,3,H,W]
    pix_flat = pix.view(B, 3, -1)  # [B,3,HW]
    rays = (Kinv @ pix_flat).view(B, 3, H, W)  # [B,3,H,W]
    X = rays * depth  # [B,3,H,W]
    return X


def _project(K: torch.Tensor, X: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
    # K: [B,3,3], X: [B,3,H,W] in camera frame -> grid: [B,H,W,2] in [-1,1]
    B = X.shape[0]
    Xz = X[:, 2:3, :, :].clamp(min=1e-6)
    x = X[:, 0:1, :, :] / Xz
    y = X[:, 1 : 1 + 1, :, :] / Xz
    u = K[:, 0, 0].view(B, 1, 1, 1) * x + K[:, 0, 2].view(B, 1, 1, 1)
    v = K[:, 1, 1].view(B, 1, 1, 1) * y + K[:, 1, 2].view(B, 1, 1, 1)
    u_norm = 2.0 * (u / (W - 1)) - 1.0
    v_norm = 2.0 * (v / (H - 1)) - 1.0
    grid = torch.stack([u_norm.squeeze(1), v_norm.squeeze(1)], dim=-1)  # [B,H,W,2]
    valid = (Xz > 0).squeeze(1)  # [B,H,W]
    return grid, valid


# ----------------------------------------
# ------------- Loss Calc ----------------
# ----------------------------------------
def _ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x,y: [B,3,H,W] in [0,1]
    C1, C2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    ssim = ssim_n / (ssim_d + 1e-8)
    return torch.clamp((1 - ssim) / 2, 0, 1)  # [0,1]


def logdepth_to_depth(d_raw, min_depth=0.1, max_depth=80.0):
    # map raw -> [log(min), log(max)] via tanh
    d_norm = 0.5 * (torch.tanh(d_raw) + 1.0)

    log_min = math.log(min_depth)
    log_max = math.log(max_depth)

    log_depth = log_min + d_norm * (log_max - log_min)
    depth = torch.exp(log_depth)
    return depth


def depth_smooth_loss(depth: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """Edege aware smoothness loss, make sure that depth is smooth, but not around image edges

    Args:
        depth (torch.Tensor): depth map
        img (torch.Tensor): image (same size)

    Returns:
        torch.Tensor: loss
    """
    dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

    img_dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), dim=1, keepdim=True)
    img_dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), dim=1, keepdim=True)

    dx = dx * torch.exp(-img_dx)
    dy = dy * torch.exp(-img_dy)

    return dx.mean() + dy.mean()


def pose_regularizer(
    pred_pose: torch.Tensor, dt: torch.Tensor, v_max: float = 180.0, w_max: float = 20.0
) -> torch.Tensor:
    """Regularize very large poses that could cause that the model finds a shortcut to mask out the full warpped image

    Args:
        pred_pose (torch.Tensor): predicted pose by the model [B, 7]
        dt (torch.Tensor): time diff in seconds [B, 1]
        v_max (float, optional): max velocity. Defaults to 180.0.
        w_max (float, optional): max radial velocity. Defaults to 20.0.

    Returns:
        torch.Tensor: loss
    """
    # --- translation ---
    dp = pred_pose[:, :3]  # delta translation
    trans_dist = torch.norm(dp, dim=-1)  # m per step

    trans_limit = v_max * dt  # per-step allowed m
    trans_excess = F.relu(trans_dist - trans_limit)
    trans_loss = (trans_excess**2).mean()

    # --- rotation ---
    dq = pred_pose[:, 3:7]
    dq = dq / (dq.norm(dim=-1, keepdim=True) + 1e-8)
    qw = dq[..., 3].clamp(-1.0, 1.0)
    dtheta = 2.0 * torch.acos(qw)

    rot_limit = w_max * dt  # allowed rad per step
    rot_excess = F.relu(dtheta - rot_limit)
    rot_loss = (rot_excess**2).mean()

    return (1e-2 * trans_loss + 1e-1 * rot_loss).mean()


def vio_loss(
    batch: Batch, pred_depth: torch.Tensor, pred_pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate photometric loss from input and output

    Args:
        batch (Batch): Batch for this frame
        pred_depth (torch.Tensor): Depth pred for this frame from the model: [B, 1, H, W] with depth in meter
        pred_pose (torch.Tensor): Pose pred or this frame from the model: [B, 7] as  [x, y, z, qx, qy, qz, qw]

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Return multiple losses:
            Photometric loss: how similar are the original and warpped image on a feature level
            Reconstruction loss: how similar are the original and warpped image on pixel level
            reconstructed image: The resulting reconstructed image for validation/visu
    """
    img_t = batch["cam_front"]["img"]  # [B,3,H,W]
    img_tm1 = batch["cam_front"]["img_tm1"]  # [B,3,H,W]
    intr = batch["cam_front"]["intr"]  # [B,4]
    extr = batch["cam_front"]["extr"]  # [B,7]
    imu_dt = batch["imu_dt"]  # [B, 1]

    # Convert Depth prediction to proper depth values
    B, _, H, W = img_t.shape
    if pred_depth.shape[-2:] != (H, W):
        resized_pred_depth = F.interpolate(pred_depth, size=(H, W), mode="bilinear", align_corners=True)
    else:
        resized_pred_depth = pred_depth
    depth_t = logdepth_to_depth(resized_pred_depth, 0.1, 255.0)

    # Warp image based on predictions
    K = _make_K(intr)
    K_inv = torch.inverse(K)

    T = _se3_from_tq(extr[:, :3], extr[:, 3:7])
    T_inv = _invert_se3(T)

    T_ego_tm1_to_t = _se3_from_tq(pred_pose[:, :3], pred_pose[:, 3:7])
    T_ego_t_to_tm1 = _invert_se3(T_ego_tm1_to_t)
    T_cam_t_to_tm1 = T_inv @ T_ego_t_to_tm1 @ T  # cam_t -> ego_t -> ego_t -> ego_tm1 -> ego_tm1 -> cam_tm1

    X_cam_t = _backproject(depth_t, K_inv)
    X_cam_tm1 = _transform_points(T_cam_t_to_tm1, X_cam_t)

    grid, valid = _project(K, X_cam_tm1, H, W)
    gx, gy = grid[..., 0], grid[..., 1]
    inb = (gx > -1) & (gx < 1) & (gy > -1) & (gy < 1)
    mask = (valid & inb).float().unsqueeze(1)
    recon_tm1 = F.grid_sample(img_t, grid, mode="bilinear", padding_mode="zeros")

    # Calc image loss
    ssim = _ssim(img_tm1, recon_tm1)
    l1 = torch.abs(img_tm1 - recon_tm1).mean(dim=1, keepdim=True)
    err_per_pixel = 0.85 * ssim.mean(dim=1, keepdim=True) + 0.15 * l1
    photo_loss = (err_per_pixel * mask).sum() / (mask.sum() + 1e-6)

    # Calc depth smoothness loss (keep very small)
    smooth_loss = depth_smooth_loss(depth_t, img_t)

    # Pose reg loss to avoid shortcut of just providing huge rotations and translations to get mask to 0
    pose_reg = pose_regularizer(pred_pose, batch["cam_front"]["timediff_s"])

    # Scale loss based on imu measurement
    trans_pred = pred_pose[:, :3].norm(dim=1)
    scale_loss = 0.001 * F.l1_loss(torch.log(trans_pred + 1e-6), torch.log(imu_dt + 1e-6))

    return (
        photo_loss,
        smooth_loss,
        pose_reg,
        scale_loss,
        recon_tm1,
    )
