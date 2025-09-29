from torch import nn

from common.layers import ResBlock, SpatialMHABlock, get_2d_sincos_pos_embed


class Decoder(nn.Module):
    def __init__(
        self,
        out_ch=3,
        res_block_ch: list[int] = [32, 64, 64, 128, 128, 256],
        num_attn_blocks: int = 4,
    ):
        super().__init__()

        # Cached positional encoding
        self._cached_pe = None
        self._cached_key = None

        # Bottleneck attention (same as encoder)
        self.attn = nn.Sequential(*[SpatialMHABlock(dim=res_block_ch[-1], num_heads=6) for _ in range(num_attn_blocks)])

        # ResNet + Upsample stack (reverse order of encoder)
        self.res_layers = nn.ModuleList()
        for i in reversed(range(len(res_block_ch) - 1)):
            curr_ch = res_block_ch[i + 1]  # higher channels
            next_ch = res_block_ch[i]  # lower channels
            stage = nn.Sequential(
                nn.ConvTranspose2d(curr_ch, next_ch, 4, stride=2, padding=1, bias=False),  # upsample
                nn.ReLU(inplace=True),
                ResBlock(next_ch),
                ResBlock(next_ch),
            )
            self.res_layers.append(stage)

        # Final projection to output channels (e.g. 3 for RGB)
        self.out_conv = nn.Conv2d(res_block_ch[0], out_ch, 3, padding=1)

    def _pos_embed(self, x):
        B, C, H, W = x.shape
        key = (C, H, W, x.device, x.dtype)
        if self._cached_key != key:
            self._cached_pe = get_2d_sincos_pos_embed(C, H, W, x.device).to(x.dtype)
            self._cached_key = key
        return x + self._cached_pe

    def forward(self, x):
        # Bottleneck attention with positional encoding
        x = self._pos_embed(x)
        x = self.attn(x)

        # Reverse ResNet + Upsample stages
        for layer in self.res_layers:
            x = layer(x)

        # Final projection
        x = self.out_conv(x)
        return x
