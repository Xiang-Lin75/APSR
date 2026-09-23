# ConvBlock and grouped GRUs adapted from GTCRN (Rong Xiaobin, MIT).
# LA_2D follows TIGER gated local/global fusion with 2-D causal normalization.
# Retained notices and source references: NOTICE and docs/THIRD_PARTY_NOTICES.md.

"""Normalization, grouped GRUs and convolutional building blocks."""

from __future__ import annotations
import torch
from torch import nn


class cLN4D(nn.Module):
    def __init__(self, channels, eps=1e-08):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._streaming = False
        self._s_sum = None
        self._s_pow_sum = None
        self._s_count = 0

    def _reset_streaming(self):
        self._s_sum = None
        self._s_pow_sum = None
        self._s_count = 0

    def forward(self, x):
        b, c, t, f = x.shape
        step_sum = x.sum(dim=(1, 3))
        step_pow_sum = x.pow(2).sum(dim=(1, 3))
        if self._streaming:
            if self._s_sum is None:
                self._s_sum = torch.zeros(b, 1, device=x.device, dtype=x.dtype)
                self._s_pow_sum = torch.zeros(b, 1, device=x.device, dtype=x.dtype)
            cum_sum = torch.cumsum(step_sum, dim=1) + self._s_sum
            cum_pow_sum = torch.cumsum(step_pow_sum, dim=1) + self._s_pow_sum
            self._s_sum = cum_sum[:, -1:].clone()
            self._s_pow_sum = cum_pow_sum[:, -1:].clone()
            base_count = self._s_count
            entry_cnt = (
                torch.arange(
                    base_count + 1, base_count + t + 1, device=x.device, dtype=x.dtype
                )
                * c
                * f
            )
            self._s_count = base_count + t
            entry_cnt = entry_cnt.unsqueeze(0)
        else:
            cum_sum = torch.cumsum(step_sum, dim=1)
            cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)
            entry_cnt = torch.arange(1, t + 1, device=x.device, dtype=x.dtype) * c * f
            entry_cnt = entry_cnt.unsqueeze(0)
        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()
        cum_mean = cum_mean.unsqueeze(1).unsqueeze(3)
        cum_std = cum_std.unsqueeze(1).unsqueeze(3)
        x_hat = (x - cum_mean) / cum_std
        return x_hat * self.gain + self.bias


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        use_deconv=False,
        is_last=False,
    ):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups
        )
        self.bn = nn.Identity() if is_last else cLN4D(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LA_2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.local_norm = cLN4D(channels)
        self.gate_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate_norm = cLN4D(channels)
        self.gate_act = nn.Sigmoid()
        self.global_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.global_norm = cLN4D(channels)

    def forward(self, x_local, x_global):
        local_feat = self.local_norm(self.local_proj(x_local))
        gate = self.gate_act(self.gate_norm(self.gate_proj(x_global)))
        global_feat = self.global_norm(self.global_proj(x_global))
        return local_feat * gate + global_feat


class LinearHeadConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        use_deconv=False,
    ):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups
        )

    def forward(self, x):
        return self.conv(x)


class LearnableSigmoid(nn.Module):
    def __init__(self, n_channels, n_freqs):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(n_channels, 1, n_freqs))

    def forward(self, x):
        return torch.sigmoid(self.slope * x)


class FramewiseLayerNorm4D(nn.Module):
    """Layer-normalize each time frame over channel-frequency bins."""

    def __init__(self, channels, eps=1e-08):
        super().__init__()
        self.eps = float(eps)
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=(1, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(1, 3), keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.gain + self.bias


class GRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.rnn2 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x, h=None):
        if h == None:
            if self.bidirectional:
                h = torch.zeros(2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(1, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = (h1.contiguous(), h2.contiguous())
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return (y, h)


class StreamingGRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.rnn2 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self._streaming = False
        self._s_h1 = None
        self._s_h2 = None

    def _reset_streaming(self):
        self._s_h1 = None
        self._s_h2 = None

    def forward(self, x, h=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        if h is not None:
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            h1, h2 = (h1.contiguous(), h2.contiguous())
        else:
            h1, h2 = (None, None)
        if self._streaming:
            if h1 is None and self._s_h1 is not None:
                h1 = self._s_h1
            if h2 is None and self._s_h2 is not None:
                h2 = self._s_h2
        y1, new_h1 = self.rnn1(x1.contiguous(), h1)
        y2, new_h2 = self.rnn2(x2.contiguous(), h2)
        if self._streaming:
            self._s_h1 = new_h1
            self._s_h2 = new_h2
        y = torch.cat([y1, y2], dim=-1)
        new_h = torch.cat([new_h1, new_h2], dim=-1)
        return (y, new_h)
