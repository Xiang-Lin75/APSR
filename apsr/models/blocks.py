# Grouped dual-path processing follows GTCRN (MIT). Per-head attention
# projections and head recombination follow ESPnet TF-GridNet (Apache-2.0).
# APSR changes normalization, causal/prefix-boundary handling and recurrence.
# Retained notices: NOTICE, licenses/GTCRN-MIT.txt, licenses/ESPnet-Apache-2.0.txt.

"""APSR separator block; memory is the sole enrollment-to-mixture route."""

from __future__ import annotations
import math
import torch
from torch import nn
from .layers import cLN4D, GRNN, StreamingGRNN


class MemoryOnlyCausalAttention(nn.Module):
    """Causal attention with separate prefix/mixture cumulative normalization and masked cross-boundary keys."""

    def __init__(self, emb_dim, n_freqs, n_head=2, approx_qk_dim=128):
        super().__init__()
        self.n_head = n_head
        self.emb_dim = emb_dim
        proj_dim = math.ceil(approx_qk_dim / n_freqs)
        if emb_dim % n_head != 0:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be divisible by n_head ({n_head})"
            )
        for ii in range(n_head):
            self.add_module(
                f"attn_conv_Q_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, proj_dim, 1), nn.PReLU(), cLN4D(proj_dim)
                ),
            )
            self.add_module(
                f"attn_conv_K_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, proj_dim, 1), nn.PReLU(), cLN4D(proj_dim)
                ),
            )
            self.add_module(
                f"attn_conv_V_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    nn.PReLU(),
                    cLN4D(emb_dim // n_head),
                ),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(nn.Conv2d(emb_dim, emb_dim, 1), nn.PReLU(), cLN4D(emb_dim)),
        )
        self._streaming = False
        self._s_mix_k_cache = None
        self._s_mix_v_cache = None
        self._stream_max_mix_frames = None
        self._stream_attention_mode = "tfacm_like_local"
        self._stream_prefix_tokens = 0

    def __getitem__(self, key):
        return getattr(self, key)

    def _forward_offline(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, f = x.shape
        prefix_tokens = int(getattr(self, "direct_prefix_tokens", 0))
        if not 0 < prefix_tokens < t:
            raise ValueError(
                f"direct-prefix attention control requires 0 < prefix tokens < sequence length, got prefix={prefix_tokens}, length={t}"
            )
        all_q = []
        all_k = []
        all_v = []
        for head_index in range(self.n_head):
            projected = []
            for name in ("Q", "K", "V"):
                module = self[f"attn_conv_{name}_{head_index}"]
                prefix_projection = module(x[:, :, :prefix_tokens])
                mixture_projection = module(x[:, :, prefix_tokens:])
                projected.append(
                    torch.cat([prefix_projection, mixture_projection], dim=2)
                )
            q_head, k_head, v_head = projected
            all_q.append(q_head)
            all_k.append(k_head)
            all_v.append(v_head)
        q = torch.cat(all_q, dim=0).transpose(1, 2).flatten(start_dim=2)
        k = torch.cat(all_k, dim=0).transpose(1, 2).flatten(start_dim=2)
        v = torch.cat(all_v, dim=0).transpose(1, 2)
        old_shape = v.shape
        v = v.flatten(start_dim=2)
        attn_dim = q.shape[-1]
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        mask[prefix_tokens:, :prefix_tokens] = False
        scores = torch.matmul(q, k.transpose(1, 2)) / attn_dim**0.5
        scores = scores.masked_fill(~mask, float("-inf"))
        attended = torch.matmul(torch.softmax(scores, dim=-1), v)
        attended = attended.reshape(old_shape).transpose(1, 2)
        value_channels = attended.shape[1]
        attended = (
            attended.view(self.n_head, b, value_channels, t, f)
            .transpose(0, 1)
            .contiguous()
            .view(b, self.n_head * value_channels, t, f)
        )
        prefix_attended = self["attn_concat_proj"](attended[:, :, :prefix_tokens])
        mixture_attended = self["attn_concat_proj"](attended[:, :, prefix_tokens:])
        attended = torch.cat([prefix_attended, mixture_attended], dim=2)
        return attended + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._streaming:
            raise RuntimeError("APSR blocks are not verified for streaming")
        return self._forward_offline(x)


class SeparatorBlock(nn.Module):
    """Frequency GRU, independent temporal GRUs, anchored memory, then causal attention. Input/output (B,C,K+1+T,F)."""

    def __init__(self, input_size, width, hidden_size, n_head=2, approx_qk_dim=128):
        super().__init__()
        self.width = width
        self.hidden_size = hidden_size
        self.intra_rnn = GRNN(
            input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True
        )
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-08)
        self.inter_rnn = StreamingGRNN(
            input_size=hidden_size, hidden_size=hidden_size, bidirectional=False
        )
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-08)
        self.car_attn = MemoryOnlyCausalAttention(
            hidden_size, n_freqs=width, n_head=n_head, approx_qk_dim=approx_qk_dim
        )
        self.gcfn = nn.Identity()

    def forward(self, x: torch.Tensor, h=None):
        if h is not None:
            raise ValueError("APSR blocks require fresh block state")
        b, c, t, f = x.shape
        prefix_tokens = int(getattr(self, "direct_prefix_tokens", 0))
        if not 0 < prefix_tokens < t:
            raise ValueError(
                f"direct-prefix block control requires 0 < prefix tokens < sequence length, got prefix={prefix_tokens}, length={t}"
            )
        x_tfc = x.permute(0, 2, 3, 1)
        intra_in = x_tfc.reshape(b * t, f, c)
        intra_mix, _ = self.intra_rnn(intra_in)
        intra_mix = self.intra_fc(intra_mix)
        intra_x = self.intra_ln(intra_mix.reshape(b, t, self.width, self.hidden_size))
        intra_out = x_tfc + intra_x
        inter_in = intra_out.permute(0, 2, 1, 3).reshape(
            b * self.width, t, self.hidden_size
        )
        prefix_inter, _prefix_hidden = self.inter_rnn(
            inter_in[:, :prefix_tokens], h=None
        )
        mixture_inter, mixture_hidden = self.inter_rnn(
            inter_in[:, prefix_tokens:], h=None
        )
        inter_x = torch.cat([prefix_inter, mixture_inter], dim=1)
        inter_x = self.inter_fc(inter_x).reshape(b, self.width, t, self.hidden_size)
        inter_x = inter_x.permute(0, 2, 1, 3)
        inter_x = self.inter_ln(inter_x)
        inter_out = intra_out + inter_x
        inter_out = self.target_memory.forward_block(inter_out)
        inter_bctf = inter_out.permute(0, 3, 1, 2).contiguous()
        attn_out = self.car_attn(inter_bctf)
        return (self.gcfn(attn_out), mixture_hidden)
