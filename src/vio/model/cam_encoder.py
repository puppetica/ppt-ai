import torch
from torch import nn

from common.layers import ResBlock, SpatialMHABlock, get_2d_sincos_pos_embed


class CamEncoder(nn.Module):
    def __init__(
        self,
        in_ch=3,
        res_block_ch: list[int] = [64, 64, 128, 256, 384],
        num_attn_blocks: int = 4,
    ):
        super().__init__()

        # We input images t and t-1
        self.in_ch = in_ch * 2

        self._cached_pe = None
        self._cached_key = None

        # ResNet Blocks
        self.res_layers = nn.ModuleList()
        self.res_layers.append(nn.Conv2d(self.in_ch, res_block_ch[0], 5, stride=2, padding=1, bias=False))
        self.res_layers.append(nn.ReLU(inplace=True))

        for i in range(len(res_block_ch) - 1):
            curr_ch = res_block_ch[i]
            next_ch = res_block_ch[i + 1]
            self.res_layers.append(
                nn.Sequential(
                    ResBlock(curr_ch),
                    ResBlock(curr_ch),
                    nn.Conv2d(curr_ch, next_ch, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        dpr = torch.linspace(0, 0.1, num_attn_blocks).tolist()
        self.attn = nn.Sequential(
            *[SpatialMHABlock(dim=res_block_ch[-1], num_heads=6, drop_path=dpr[i]) for _ in range(num_attn_blocks)]
        )

    def _pos_embed(self, x):
        B, C, H, W = x.shape
        key = (C, H, W, x.device, x.dtype)
        if self._cached_key != key:
            self._cached_pe = get_2d_sincos_pos_embed(C, H, W, x.device).to(x.dtype)
            self._cached_key = key
        return x + self._cached_pe

    def forward(self, img_t, img_tm1):
        # concat along channel: [B,3,H,W] + [B,3,H,W] → [B,6,H,W]
        x = torch.cat([img_tm1, img_t], dim=1)
        for layer in self.res_layers:
            x = layer(x)
        x = self._pos_embed(x)
        x = self.attn(x)
        return x  # [B, C_out, H_out, W_out]
