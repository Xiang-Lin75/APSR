# ERB and SFE adapted from GTCRN, Copyright (c) 2024 Rong Xiaobin.
# MIT License: see licenses/GTCRN-MIT.txt and NOTICE.

"""Fixed ERB analysis/synthesis and local frequency unfolding."""

from __future__ import annotations
import numpy as np
import torch
from torch import nn


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(
            erb_subband_1, erb_subband_2, nfft, high_lim, fs
        )
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    def erb2hz(self, erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def erb_filter_banks(
        self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000
    ):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)
        erb_filters[0, bins[0] : bins[1]] = (
            bins[1] - np.arange(bins[0], bins[1]) + 1e-12
        ) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i] : bins[i + 1]] = (
                np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12
            ) / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1] : bins[i + 2]] = (
                bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12
            ) / (bins[i + 2] - bins[i + 1] + 1e-12)
        erb_filters[-1, bins[-2] : bins[-1] + 1] = (
            1 - erb_filters[-2, bins[-2] : bins[-1] + 1]
        )
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        x_low = x[..., : self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1 :])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        x_erb_low = x_erb[..., : self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1 :])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, (kernel_size - 1) // 2),
        )

    def forward(self, x):
        return self.unfold(x).reshape(
            x.shape[0], x.shape[1] * self.kernel_size, x.shape[2], x.shape[3]
        )
