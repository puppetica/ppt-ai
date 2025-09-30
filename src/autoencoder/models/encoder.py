from torch import linspace, nn

from common.layers import ResBlock, SpatialMHABlock, get_2d_sincos_pos_embed


class Encoder(nn.Module):
    def __init__(
        self,
        in_ch=3,
        res_block_ch: list[int] = [32, 64, 64, 128, 128, 256],
        num_attn_blocks: int = 4,
    ):
        super().__init__()

        # Cached positional encoding
        self._cached_pe = None
        self._cached_key = None

        # ResNet Blocks
        self.res_layers = nn.ModuleList()

        self.res_layers.append(nn.Conv2d(in_ch, res_block_ch[0], 5, stride=2, padding=1, bias=False))
        self.res_layers.append(nn.ReLU(inplace=True))

        for i in range(len(res_block_ch) - 1):
            curr_ch = res_block_ch[i]
            next_ch = res_block_ch[i + 1]
            self.res_layers.append(
                nn.Sequential(
                    ResBlock(curr_ch),
                    ResBlock(curr_ch),
                    # Downsample
                    nn.Conv2d(curr_ch, next_ch, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        # Bottleneck attention
        dpr = linspace(0, 0.1, num_attn_blocks).tolist()  # from 0 → 0.1
        self.attn = nn.Sequential(
            *[SpatialMHABlock(dim=384, num_heads=6, drop_path=dpr[i]) for _ in range(num_attn_blocks)]
        )

    def _pos_embed(self, x):
        B, C, H, W = x.shape
        key = (C, H, W, x.device, x.dtype)
        if self._cached_key != key:
            self._cached_pe = get_2d_sincos_pos_embed(C, H, W, x.device).to(x.dtype)
            self._cached_key = key
        return x + self._cached_pe

    def forward(self, x):
        for layer in self.res_layers:
            x = layer(x)
        x = self._pos_embed(x)
        x = self.attn(x)
        return x
