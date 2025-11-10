import torch
from torch import nn


class PoseEstimator(nn.Module):
    def __init__(
        self,
        token_dim: int = 256,  # must match IMU token_dim
        cam_in_dim: int = 384,  # CamEncoder output channels (last stage)
        nhead: int = 8,
        depth: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # project pooled camera feature to token_dim
        self.cam_pool = nn.AdaptiveAvgPool2d(1)
        self.cam_proj = nn.Linear(cam_in_dim, token_dim)

        # calibration embedding: [fx, fy, cx, cy] + extr(7) + Δt(1) = 12
        self.calib_proj = nn.Sequential(
            nn.Linear(12, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # type embeddings
        self.cls_type = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.cam_type = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.imu_type = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        # transformer encoder over tokens [CLS, CAM, IMU...]
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=int(token_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth, enable_nested_tensor=False)

        # pose head from CLS token
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 7),  # 3 trans + 4 quat
        )

    def forward(
        self,
        cam_feat: torch.Tensor,  # [B, C, H, W] from CamEncoder
        imu_tokens: torch.Tensor,  # [B, K, token_dim] from ImuEncoder
        intr: torch.Tensor,  # [B, 4]  (fx, fy, cx, cy)
        extr: torch.Tensor,  # [B, 7]  (x,y,z,qx,qy,qz,qw) sensor→ego
        delta_t: torch.Tensor,  # [B]     seconds between t-1 and t
    ) -> torch.Tensor:
        B = cam_feat.size(0)

        # camera token
        cam_vec = self.cam_pool(cam_feat).flatten(1)  # [B, C]
        cam_tok = self.cam_proj(cam_vec).unsqueeze(1)  # [B,1,D]

        # imu tokens
        imu_tok = imu_tokens + self.imu_type.expand(B, imu_tokens.size(1), -1)  # [B,K,D]

        # CLS token
        cls_tok = torch.zeros(B, 1, cam_tok.size(-1), device=cam_tok.device)
        cls_tok = cls_tok + self.cls_type

        # calibration embedding added to CAM and CLS (conditions pose on rig + timing)
        calib = torch.cat([intr, extr, delta_t.unsqueeze(-1)], dim=1)  # [B,12]
        calib_emb = self.calib_proj(calib).unsqueeze(1)  # [B,1,D]

        cam_tok = cam_tok + self.cam_type + calib_emb
        cls_tok = cls_tok + calib_emb

        # fuse sequence
        tokens = torch.cat([cls_tok, cam_tok, imu_tok], dim=1)  # [B, 1+1+K, D]
        fused = self.encoder(tokens)  # [B, L, D]

        # regress from CLS
        pose = self.head(fused[:, 0])  # [B,7]

        # normalize quaternion
        t = pose[:, :3]
        q = pose[:, 3:]
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.cat([t, q], dim=1)  # [B,7]
