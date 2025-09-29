import math

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic Depth per sample (residual path)."""

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob: float = 0.0, training: bool = False):
        """Drop paths (Stochastic Depth) per sample.
        Args:
            x: input tensor
            drop_prob: probability of dropping paths
            training: apply only during training
        """
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        # Work with broadcast along non-batch dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.act(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatialMHABlock(nn.Module):
    """
    Transformer block that operates over (H*W) tokens with multi-head attention.
    Input/Output: (B, C, H, W)
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, attn_dropout=0.1, proj_dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=proj_dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B,N,C)
        h = self.norm1(tokens)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x_seq = tokens + self.drop_path(attn_out)
        h = self.norm2(x_seq)
        x_seq = x_seq + self.drop_path(self.mlp(h))
        # Back to image
        x = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


def get_2d_sincos_pos_embed(c, h, w, device):
    """
    Returns (1, C, H, W) 2D sin/cos positional embedding.
    C must be even and divisible by 2.
    """
    assert c % 4 == 0, "Channel dim for pos embed must be /4"
    # split channels across H and W
    c_h = c // 2
    c_w = c - c_h

    def _pe(length, dim):
        pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # (length, dim)

    pe_h = _pe(h, c_h)  # (H, c_h)
    pe_w = _pe(w, c_w)  # (W, c_w)

    pe = pe_h[:, None, :].repeat(1, w, 1)  # (H,1,c_h)  # (H,W,c_h)
    pew = pe_w[None, :, :].repeat(h, 1, 1)  # (1,W,c_w)  # (H,W,c_w)
    pe2d = torch.cat([pe, pew], dim=-1)  # (H,W,C)
    pe2d = pe2d.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    return pe2d
