from typing import List

import torch
from torch import nn


class ImuEncoder(nn.Module):
    """
    Input per sample: list of tensors shaped [7] = [ax, ay, az, gx, gy, gz, ts_ns]
    Output: [B, num_tokens, token_dim]
    """

    def __init__(
        self,
        token_dim: int = 256,
        hidden: int = 256,
        num_tokens: int = 3,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_proj = nn.Sequential(
            nn.Linear(7, hidden),  # 6 IMU + Δt_step will be formed inline
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.gru_cell = nn.GRUCell(hidden, hidden)

        # Temporal projection: K learnable queries attend over the hidden sequence
        self.query = nn.Parameter(torch.randn(num_tokens, hidden) * 0.02)
        self.scale = hidden**-0.5
        self.post = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, token_dim))

    @staticmethod
    def _build_features(sample_list: List[torch.Tensor], device, dtype):
        """
        Returns [N, 7] = 6 IMU + Δt_step (seconds)
        """
        if len(sample_list) == 0:
            return torch.zeros(1, 7, device=device, dtype=dtype)

        s = torch.stack(sample_list, dim=0).to(device=device, dtype=dtype)  # [N,7]
        ts = s[:, -1]
        dt = torch.zeros_like(ts)
        if s.shape[0] > 1:
            dt[1:] = ts[1:] - ts[:-1]
        imu6 = s[:, :6]
        return torch.cat([imu6, dt.unsqueeze(-1)], dim=-1)  # [N,7]

    def _encode_one(self, imu_list: List[torch.Tensor], device, dtype):
        # build features
        feats = self._build_features(imu_list, device, dtype)  # [N, 7]
        x = self.in_proj(feats)  # [N, H]

        # GRUCell unroll
        H = x.size(-1)
        h = torch.zeros(H, device=device, dtype=dtype)
        hs = []
        for t in range(x.size(0)):
            h = self.gru_cell(x[t], h)
            hs.append(h.clone())
        Hseq = torch.stack(hs, dim=0)  # [N, H]

        # temporal projection to K tokens via learned queries
        # scores: [K, N] = QK^T
        scores = (self.query @ Hseq.t()) * self.scale
        weights = scores.softmax(dim=-1)  # [K, N]
        tokens = weights @ Hseq  # [K, H]
        tokens = self.post(tokens)  # [K, token_dim]
        return tokens

    def forward(self, imu_lists: List[List[torch.Tensor]]) -> torch.Tensor:
        B = len(imu_lists)
        # pick device/dtype robustly
        device = imu_lists[0][0].device if len(imu_lists[0]) > 0 else torch.device("cpu")
        dtype = torch.float32

        outs = []
        for b in range(B):
            outs.append(self._encode_one(imu_lists[b], device, dtype))
        return torch.stack(outs, dim=0)  # [B, num_token, token_dim]
