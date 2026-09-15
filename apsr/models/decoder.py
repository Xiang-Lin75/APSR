"""Gated skip fusion, spectral decoder and target/sink mask head."""

from __future__ import annotations
import torch
from torch import nn
from .layers import LA_2D, LinearHeadConv, cLN4D, LearnableSigmoid


class SpeechDecoder(nn.Module):
    """Decoder with a gated encoder skip and a linear spectral projection."""

    def __init__(
        self,
        hidden_channels: int = 64,
        freq_downsample_layers: int = 1,
        encoder_freq_stride: int = 1,
    ) -> None:
        super().__init__()
        if int(freq_downsample_layers) != 1:
            raise ValueError("APSR-P requires freq_downsample_layers=1")
        if int(encoder_freq_stride) != 1:
            raise ValueError("APSR-P requires encoder_freq_stride=1")
        channels = int(hidden_channels)
        if channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.de_convs = nn.ModuleList(
            [
                LinearHeadConv(
                    channels,
                    channels,
                    (1, 5),
                    stride=(1, 1),
                    padding=(0, 2),
                    use_deconv=False,
                )
            ]
        )
        self.skip_fuse = nn.ModuleList([LA_2D(channels)])

    def forward(self, x: torch.Tensor, encoder_outputs):
        if len(encoder_outputs) != 1:
            raise ValueError(
                f"APSR-P decoder requires exactly one encoder skip, got {len(encoder_outputs)}"
            )
        x = self.skip_fuse[0](x_local=encoder_outputs[0], x_global=x)
        return self.de_convs[0](x)


class ComplexMaskHead(nn.Module):
    """Gated target/sink complex masks with an affine mask-sum constraint."""

    def __init__(
        self,
        hidden_channels,
        num_sources=1,
        apply_constraint=True,
        nfft=512,
        noise_sink=True,
    ):
        super().__init__()
        self.num_sources = int(num_sources)
        self.apply_constraint = bool(apply_constraint)
        self.noise_sink = bool(noise_sink)
        self.group_k = self.num_sources + 1 if self.noise_sink else self.num_sources
        self.mask_pre = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.PReLU(),
            cLN4D(hidden_channels),
        )
        self.mask_out = nn.Conv2d(hidden_channels, 4 * self.group_k, 1)
        self.gate_act = LearnableSigmoid(
            n_channels=2 * self.group_k, n_freqs=nfft // 2 + 1
        )

    def constrain_mask_sum(self, mask_real, mask_imag):
        k = self.group_k
        real_sum = mask_real.sum(dim=1, keepdim=True)
        imag_sum = mask_imag.sum(dim=1, keepdim=True)
        mask_real = mask_real - (real_sum - 1) / k
        mask_imag = mask_imag - imag_sum / k
        return (mask_real, mask_imag)

    def _apply_mask_to_spec(self, mask_real, mask_imag, spec):
        outputs = []
        for src_idx in range(mask_real.shape[1]):
            m_real = mask_real[:, src_idx]
            m_imag = mask_imag[:, src_idx]
            s_real = spec[:, 0] * m_real - spec[:, 1] * m_imag
            s_imag = spec[:, 1] * m_real + spec[:, 0] * m_imag
            outputs.append(torch.stack([s_real, s_imag], dim=1))
        return torch.stack(outputs, dim=1)

    def forward(self, feat, spec, enr_tokens=None, routing_prompt=None):
        del enr_tokens, routing_prompt
        b, _, t, f = feat.shape
        mask_output = self.mask_out(self.mask_pre(feat))
        mask_output = mask_output.reshape(b, 2, 2, self.group_k, t, f)
        complex_mask = mask_output[:, 0]
        gate_logits = mask_output[:, 1].reshape(b, 2 * self.group_k, t, f)
        gates = self.gate_act(gate_logits).reshape(b, 2, self.group_k, t, f)
        gated_mask = complex_mask * gates
        grouped_real = gated_mask[:, 0]
        grouped_imag = gated_mask[:, 1]
        if self.apply_constraint:
            grouped_real, grouped_imag = self.constrain_mask_sum(
                grouped_real, grouped_imag
            )
        grouped_specs = self._apply_mask_to_spec(grouped_real, grouped_imag, spec)
        speech_specs = grouped_specs[:, : self.num_sources]
        aux = {
            "grouped_specs": grouped_specs,
            "grouped_mask_real": grouped_real,
            "grouped_mask_imag": grouped_imag,
            "latent_mask_real": None,
            "latent_mask_imag": None,
            "assign": None,
            "activity_logit": None,
            "activity_prob": None,
            "direct_mask_head": True,
            "prompt_routing_conditioning": False,
        }
        return (speech_specs, aux)
