"""Frequency-resolved enrollment pooling into K tokens."""

from __future__ import annotations
import math
import torch
from torch import nn
from .layers import FramewiseLayerNorm4D


class GatedAttentiveEnrollmentPooling(nn.Module):
    """Enrollment-only speaker pooling with lightweight local TF enhancement.

    This keeps the static-prefix contract of EnrollmentPooling:
    input  (B, C, T_enr, F) -> output (B, C, K, F).
    """

    def __init__(self, channels, num_tokens=4, att_dim=32):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.att_dim = int(att_dim)
        self.pre_norm = FramewiseLayerNorm4D(channels)
        self.local_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            groups=channels,
            bias=False,
        )
        self.local_pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(
            channels, channels, kernel_size=(3, 3), padding=(1, 1), groups=channels
        )
        self.enhance_norm = FramewiseLayerNorm4D(channels)
        self.enhance_act = nn.PReLU()
        self.key_proj = nn.Conv2d(channels, self.att_dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.quality = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                groups=channels,
                bias=False,
            ),
            nn.PReLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.query = nn.Parameter(
            torch.randn(self.num_tokens, self.att_dim) * self.att_dim ** (-0.5)
        )
        self.token_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.token_norm = FramewiseLayerNorm4D(channels)
        self.token_act = nn.PReLU()

    def forward(self, x):
        h = self.pre_norm(x)
        local = self.local_pw(self.local_dw(h))
        gate = torch.sigmoid(self.gate(h))
        enhanced = self.enhance_act(self.enhance_norm(x + local * gate))
        key = self.key_proj(enhanced).mean(dim=3).transpose(1, 2)
        value = self.value_proj(enhanced)
        quality = self.quality(enhanced).mean(dim=3).squeeze(1)
        query = self.query.to(device=x.device, dtype=x.dtype)
        logits = torch.einsum("ka,bta->bkt", query, key)
        logits = logits / math.sqrt(max(1, self.att_dim))
        logits = logits + quality.unsqueeze(1)
        attn = torch.softmax(logits, dim=-1)
        tokens = torch.einsum("bkt,bctf->bckf", attn, value)
        tokens = self.token_proj(tokens)
        tokens = self.token_norm(tokens)
        return self.token_act(tokens)
