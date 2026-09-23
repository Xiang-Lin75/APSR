"""Shared mixture/enrollment encoder."""

from __future__ import annotations
import torch
from torch import nn
from .layers import ConvBlock


class SpeechEncoder(nn.Module):
    """Shared frame-local speech encoder. Input (B,9,T,F); output (B,C,T,F) and skips."""

    def __init__(
        self,
        hidden_channels: int = 64,
        freq_downsample_layers: int = 1,
        encoder_freq_stride: int = 1,
    ) -> None:
        super().__init__()
        if int(freq_downsample_layers) != 1:
            raise ValueError("APSR requires freq_downsample_layers=1")
        if int(encoder_freq_stride) != 1:
            raise ValueError("APSR requires encoder_freq_stride=1")
        channels = int(hidden_channels)
        if channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.en_convs = nn.ModuleList(
            [
                ConvBlock(
                    3 * 3,
                    channels,
                    (1, 5),
                    stride=(1, 1),
                    padding=(0, 2),
                    use_deconv=False,
                    is_last=False,
                )
            ]
        )

    def forward(self, x: torch.Tensor):
        encoder_outputs = []
        for layer in self.en_convs:
            x = layer(x)
            encoder_outputs.append(x)
        return (x, encoder_outputs)
