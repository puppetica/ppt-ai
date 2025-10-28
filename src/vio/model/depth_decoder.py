import torch
import torch.nn.functional as F
from torch import nn

from common.layers import ResBlock, SpatialMHABlock, get_2d_sincos_pos_embed


class DepthDecoder(nn.Module):
    def __init__(
        self,
        ch: list[int] = [64, 128, 256, 384],
        num_attn_blocks: int = 4,
    ):
        super().__init__()
        assert len(ch) >= 2, "Channel ladder must have at least two stages."

        # cache for positional encoding
        self._cached_pe, self._cached_key = None, None

        # bottleneck attention (dimension = encoder's last channel)
        self.attn = nn.Sequential(*[SpatialMHABlock(dim=ch[-1], num_heads=6) for _ in range(num_attn_blocks)])

        # upsampling stages: for i: ch[i+1] -> ch[i]
        ups = []
        for i in range(len(ch) - 2, -1, -1):
            in_c, out_c = ch[i + 1], ch[i]
            ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    ResBlock(out_c),
                    ResBlock(out_c),
                )
            )
        self.ups = nn.ModuleList(ups)

        # final depth head (single map)
        self.head = nn.Conv2d(ch[0], 1, 3, padding=1)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        key = (C, H, W, x.device, x.dtype)
        if self._cached_key != key:
            self._cached_pe = get_2d_sincos_pos_embed(C, H, W, x.device).to(x.dtype)
            self._cached_key = key
        return x + self._cached_pe  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, ch[-1], H_b, W_b] from CamEncoder
        x = self._pos_embed(x)
        x = self.attn(x)

        for stage in self.ups:
            x = stage(x)

        # positive metric depth
        depth = F.softplus(self.head(x)) + 1e-6  # [B,1,H_out,W_out]
        return depth
