import torch
import torch.nn.functional as F

# -------------------- geometry utils --------------------


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


def _meshgrid_xy(B: int, H: int, W: int, device, dtype):
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


def _transform_points(T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # T: [B,4,4], X: [B,3,H,W] -> [B,3,H,W]
    B, _, H, W = X.shape
    Xh = torch.cat([X.view(B, 3, -1), torch.ones(B, 1, H * W, device=X.device, dtype=X.dtype)], dim=1)  # [B,4,HW]
    Yh = T @ Xh  # [B,4,HW]
    Y = Yh[:, :3, :].view(B, 3, H, W)
    return Y


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


# -------------------- photometric + smoothness --------------------


def _ssim(x, y):
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


def _edge_aware_smoothness(depth: torch.Tensor, img_t: torch.Tensor):
    # depth: [B,1,H,W], img_t: [B,3,H,W] in [0,1]
    dx_depth = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    dy_depth = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
    dx_img = torch.mean(torch.abs(img_t[:, :, :, 1:] - img_t[:, :, :, :-1]), dim=1, keepdim=True)
    dy_img = torch.mean(torch.abs(img_t[:, :, 1:, :] - img_t[:, :, :-1, :]), dim=1, keepdim=True)
    wx = torch.exp(-dx_img)
    wy = torch.exp(-dy_img)
    smooth = (dx_depth * wx).mean() + (dy_depth * wy).mean()
    return smooth


# -------------------- main loss wiring --------------------


def photometric_loss(batch, pred_depth, pred_pose):
    img_t = batch["cam_front"]["img"]  # [B,3,H,W]
    img_tm1 = batch["cam_front"]["img_tm1"]  # [B,3,H,W]
    intr = batch["cam_front"]["intr"]  # [B,4]
    extr = batch["cam_front"]["extr"]  # [B,7]
    imu_dt = batch["imu_dt"]  # [B]  # <<< ADDED: integrated metric translation (meters)

    B, _, H, W = img_t.shape

    if pred_depth.shape[-2:] != (H, W):
        depth_t = F.interpolate(pred_depth, size=(H, W), mode="bilinear", align_corners=False)
    else:
        depth_t = pred_depth

    K = _make_K(intr)
    Kinv = torch.inverse(K)

    T_ce = _se3_from_tq(extr[:, :3], extr[:, 3:7])
    T_ec = _invert_se3(T_ce)

    T_ego_tm1_to_t = _se3_from_tq(pred_pose[:, :3], pred_pose[:, 3:7])
    T_cam_tm1_to_t = T_ec @ T_ego_tm1_to_t @ T_ce

    X_cam_t = _backproject(depth_t, Kinv)
    X_cam_tm1 = _transform_points(T_cam_tm1_to_t, X_cam_t)
    grid, valid = _project(K, X_cam_tm1, H, W)

    # <<< ADDED: in-bounds mask
    gx, gy = grid[..., 0], grid[..., 1]
    inb = (gx > -1) & (gx < 1) & (gy > -1) & (gy < 1)
    mask = (valid & inb).float().unsqueeze(1)

    recon = F.grid_sample(img_tm1, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def photo_err(src, tgt):
        ssim = _ssim(src, tgt)
        l1 = torch.abs(src - tgt).mean(dim=1, keepdim=True)
        return 0.85 * ssim.mean(dim=1, keepdim=True) + 0.15 * l1

    err = photo_err(img_t, recon)  # t ← (t-1)

    # <<< ADDED: identity auto-masking
    err_id = photo_err(img_t, img_tm1)
    keep = (err < err_id).float()
    mask = mask * keep

    photo_loss = (err * mask).sum() / (mask.sum() + 1e-6)

    # <<< ADDED: inverse-depth smoothness (scale-normalized)
    inv = 1.0 / (depth_t + 1e-6)
    inv = inv / (inv.mean(dim=[2, 3], keepdim=True) + 1e-6)
    dx = torch.abs(inv[:, :, :, 1:] - inv[:, :, :, :-1])
    dy = torch.abs(inv[:, :, 1:, :] - inv[:, :, :-1, :])
    dx_img = torch.mean(torch.abs(img_t[:, :, :, 1:] - img_t[:, :, :, :-1]), dim=1, keepdim=True)
    dy_img = torch.mean(torch.abs(img_t[:, :, 1:, :] - img_t[:, :, :-1, :]), dim=1, keepdim=True)
    wx = torch.exp(-dx_img)
    wy = torch.exp(-dy_img)
    smooth_loss = (dx * wx).mean() + (dy * wy).mean()

    # <<< ADDED: IMU scale alignment
    trans_norm_pred = pred_pose[:, :3].norm(dim=1)
    s_pred = trans_norm_pred.mean()
    s_imu = imu_dt.mean()
    scale_loss = F.l1_loss(s_pred, s_imu)

    return photo_loss, smooth_loss, scale_loss, recon
