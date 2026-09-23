"""Enrollment-anchored causal memory for APSR."""

from __future__ import annotations
import math
import torch
from torch import nn
from typing import Dict, Optional, Tuple


class TargetAnchoredTimeConstantMemory(nn.Module):
    """Low-rank, target-anchored memory with an interpretable update rate.

    Tensor contracts use channel-last separator features:

    - enrollment prefix: ``(B, K, F, C)``
    - mixture sequence: ``(B, T, F, C)``
    - anchor / memory: ``(B, F, D)``

    ``forward_block`` expects the temporal block sequence to be laid out as
    ``[K enrollment tokens, one separator token, T mixture frames]``.  Prefix
    frames are never modified by the slow-memory read path.
    """

    def __init__(
        self,
        channels: int,
        memory_dim: int = 16,
        num_enroll_tokens: int = 8,
        hop_seconds: float = 0.008,
        tau_min_seconds: float = 0.25,
        tau_max_seconds: float = 4.0,
        tau_init_seconds: float = 2.0,
        write_bias: float = -2.0,
        beta_init: float = 0.1,
        beta_max: float = 0.5,
        zero_init_output: bool = True,
        scan_chunk_frames: int = 64,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.memory_dim = int(memory_dim)
        self.num_enroll_tokens = int(num_enroll_tokens)
        self.prefix_tokens = self.num_enroll_tokens + 1
        self.hop_seconds = float(hop_seconds)
        self.tau_min_seconds = float(tau_min_seconds)
        self.tau_max_seconds = float(tau_max_seconds)
        self.tau_init_seconds = float(tau_init_seconds)
        self.beta_max = float(beta_max)
        self.zero_init_output = bool(zero_init_output)
        self.scan_chunk_frames = int(scan_chunk_frames)
        self.anchor_mode = "enrollment"
        if self.channels <= 0 or self.memory_dim <= 0:
            raise ValueError("channels and memory_dim must be positive")
        if self.num_enroll_tokens <= 0:
            raise ValueError("num_enroll_tokens must be positive")
        if self.hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive")
        if not 0 < self.tau_min_seconds < self.tau_max_seconds:
            raise ValueError("tau bounds must satisfy 0 < tau_min < tau_max")
        if not self.tau_min_seconds <= self.tau_init_seconds <= self.tau_max_seconds:
            raise ValueError("tau_init_seconds must lie inside the tau bounds")
        if not 0 < beta_init < self.beta_max:
            raise ValueError("beta_init must satisfy 0 < beta_init < beta_max")
        if self.scan_chunk_frames <= 0:
            raise ValueError("scan_chunk_frames must be positive")
        d = self.memory_dim
        self.anchor_proj = nn.Linear(self.channels, d)
        self.fast_proj = nn.Linear(self.channels, d)
        self.anchor_norm = nn.LayerNorm(d)
        self.fast_norm = nn.LayerNorm(d)
        match_dim = 4 * d
        self.reliability_proj = nn.Linear(match_dim, 1)
        self.tau_proj = nn.Linear(match_dim, 1)
        self.candidate_proj = nn.Linear(2 * d, d)
        self.read_proj = nn.Linear(3 * d, d)
        self.output_proj = nn.Linear(d, self.channels)
        beta_fraction = beta_init / self.beta_max
        self.beta_logit = nn.Parameter(
            torch.tensor(
                math.log(beta_fraction / (1.0 - beta_fraction)), dtype=torch.float32
            )
        )
        self._last_summary: Dict[str, torch.Tensor] = {}
        self.reset_parameters(write_bias=write_bias)

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_max * torch.sigmoid(self.beta_logit)

    def reset_parameters(self, write_bias: float = -2.0) -> None:
        for layer in (
            self.anchor_proj,
            self.fast_proj,
            self.reliability_proj,
            self.candidate_proj,
            self.read_proj,
        ):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.constant_(self.reliability_proj.bias, float(write_bias))
        nn.init.zeros_(self.tau_proj.weight)
        tau_fraction = (self.tau_init_seconds - self.tau_min_seconds) / (
            self.tau_max_seconds - self.tau_min_seconds
        )
        tau_fraction = min(max(tau_fraction, 1e-06), 1.0 - 1e-06)
        nn.init.constant_(
            self.tau_proj.bias, math.log(tau_fraction / (1.0 - tau_fraction))
        )
        if self.zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
        else:
            nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.ones_(self.anchor_norm.weight)
        nn.init.zeros_(self.anchor_norm.bias)
        nn.init.ones_(self.fast_norm.weight)
        nn.init.zeros_(self.fast_norm.bias)

    def make_anchor(self, enrollment_prefix: torch.Tensor) -> torch.Tensor:
        """Create ``(B, F, D)`` anchors from ``(B, K, F, C)`` tokens."""
        if enrollment_prefix.dim() != 4:
            raise ValueError(
                f"enrollment_prefix must have shape (B, K, F, C), got {tuple(enrollment_prefix.shape)}"
            )
        if enrollment_prefix.shape[1] != self.num_enroll_tokens:
            raise ValueError(
                f"expected {self.num_enroll_tokens} enrollment tokens, got {enrollment_prefix.shape[1]}"
            )
        if enrollment_prefix.shape[-1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {enrollment_prefix.shape[-1]}"
            )
        pooled = enrollment_prefix.mean(dim=1)
        return self.anchor_norm(self.anchor_proj(pooled))

    def init_state(self, anchor: torch.Tensor) -> torch.Tensor:
        if anchor.dim() != 3 or anchor.shape[-1] != self.memory_dim:
            raise ValueError(
                f"anchor must have shape (B, F, {self.memory_dim}), got {tuple(anchor.shape)}"
            )
        return anchor

    def _terms(
        self, mixture_features: torch.Tensor, anchor: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        if mixture_features.dim() != 4:
            raise ValueError(
                f"mixture_features must have shape (B, T, F, C), got {tuple(mixture_features.shape)}"
            )
        b, t, f, c = mixture_features.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")
        if anchor.shape != (b, f, self.memory_dim):
            raise ValueError(
                f"anchor must have shape {(b, f, self.memory_dim)}, got {tuple(anchor.shape)}"
            )
        u = self.fast_norm(self.fast_proj(mixture_features))
        a = anchor[:, None, :, :].expand(-1, t, -1, -1)
        match = torch.cat([u, a, u * a, torch.abs(u - a)], dim=-1)
        reliability = torch.sigmoid(self.reliability_proj(match))
        tau_fraction = torch.sigmoid(self.tau_proj(match))
        tau = (
            self.tau_min_seconds
            + (self.tau_max_seconds - self.tau_min_seconds) * tau_fraction
        )
        base_update = -torch.expm1(-self.hop_seconds / tau)
        update = reliability * base_update
        candidate = a + self.beta * torch.tanh(
            self.candidate_proj(torch.cat([u, a], dim=-1))
        )
        return (u, a, reliability, tau, update, candidate)

    def forward_step(
        self,
        mixture_frame: torch.Tensor,
        anchor: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Advance one mixture frame without using future observations."""
        if mixture_frame.dim() != 3:
            raise ValueError(
                f"mixture_frame must have shape (B, F, C), got {tuple(mixture_frame.shape)}"
            )
        if memory is None:
            memory = self.init_state(anchor)
        u, a, reliability, tau, update, candidate = self._terms(
            mixture_frame[:, None, :, :], anchor
        )
        u = u[:, 0]
        a = a[:, 0]
        reliability = reliability[:, 0]
        tau = tau[:, 0]
        update = update[:, 0]
        candidate = candidate[:, 0]
        new_memory = memory + update * (candidate - memory)
        read = torch.sigmoid(self.read_proj(torch.cat([u, a, new_memory], dim=-1)))
        output = mixture_frame + self.output_proj(read * new_memory)
        aux = {
            "fast_features": u,
            "reliability": reliability,
            "tau_seconds": tau,
            "update": update,
            "read": read,
        }
        return (output, new_memory, aux)

    def forward_sequence(
        self,
        mixture_features: torch.Tensor,
        anchor: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Causally scan an entire mixture sequence."""
        if memory is None:
            memory = self.init_state(anchor)
        u, a, reliability, tau, update, candidate = self._terms(
            mixture_features, anchor
        )
        if mixture_features.shape[1] > 0:
            scan_dtype = (
                torch.float32
                if mixture_features.dtype in (torch.float16, torch.bfloat16)
                else mixture_features.dtype
            )
            memory_scan = memory.to(dtype=scan_dtype)
            memory_chunks = []
            for start in range(0, mixture_features.shape[1], self.scan_chunk_frames):
                stop = min(start + self.scan_chunk_frames, mixture_features.shape[1])
                alpha = update[:, start:stop].to(dtype=scan_dtype)
                cand = candidate[:, start:stop].to(dtype=scan_dtype)
                retain = 1.0 - alpha
                retain_product = torch.cumprod(retain, dim=1)
                injected = alpha * cand
                normalized_injected = injected / retain_product.clamp_min(1e-12)
                chunk_memory = retain_product * (
                    memory_scan[:, None, :, :]
                    + torch.cumsum(normalized_injected, dim=1)
                )
                memory_scan = chunk_memory[:, -1]
                memory_chunks.append(chunk_memory.to(dtype=mixture_features.dtype))
            memory_sequence = torch.cat(memory_chunks, dim=1)
            memory = memory_scan.to(dtype=mixture_features.dtype)
        else:
            memory_sequence = mixture_features.new_empty(
                mixture_features.shape[0], 0, mixture_features.shape[2], self.memory_dim
            )
        read = torch.sigmoid(self.read_proj(torch.cat([u, a, memory_sequence], dim=-1)))
        output = mixture_features + self.output_proj(read * memory_sequence)
        aux = {
            "fast_features": u,
            "reliability": reliability,
            "tau_seconds": tau,
            "update": update,
            "read": read,
            "memory_sequence": memory_sequence,
        }
        return (output, memory, aux)

    def forward_block_with_context(
        self, temporal_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Update mixture memory while preserving the enrollment prefix.

        Return the output and computed states for inspection; forward_block
        uses this same path and returns only the output tensor.
        """
        if temporal_features.dim() != 4:
            raise ValueError(
                f"temporal_features must have shape (B, K+1+T, F, C), got {tuple(temporal_features.shape)}"
            )
        if temporal_features.shape[1] < self.prefix_tokens:
            raise ValueError(
                f"expected at least {self.prefix_tokens} temporal frames, got {temporal_features.shape[1]}"
            )
        enrollment_prefix = temporal_features[:, : self.num_enroll_tokens]
        fixed_prefix = temporal_features[:, : self.prefix_tokens]
        mixture_features = temporal_features[:, self.prefix_tokens :]
        anchor = self.make_anchor(enrollment_prefix)
        mixture_output, final_memory, aux = self.forward_sequence(
            mixture_features, anchor
        )
        with torch.no_grad():
            if aux["update"].numel():
                update_mean = aux["update"].mean()
                reliability_mean = aux["reliability"].mean()
                tau_mean = aux["tau_seconds"].mean()
                read_mean = aux["read"].mean()
            else:
                zero = temporal_features.new_zeros(())
                update_mean = reliability_mean = tau_mean = read_mean = zero
            self._last_summary = {
                "update_mean": update_mean.detach(),
                "reliability_mean": reliability_mean.detach(),
                "tau_seconds_mean": tau_mean.detach(),
                "read_mean": read_mean.detach(),
                "anchor_norm": anchor.norm(dim=-1).mean().detach(),
                "memory_norm": final_memory.norm(dim=-1).mean().detach(),
            }
        output = torch.cat([fixed_prefix, mixture_output], dim=1)
        context = {
            "anchor": anchor,
            "enrollment_prefix": enrollment_prefix,
            "fast_features": aux["fast_features"],
            "reliability": aux["reliability"],
            "memory_sequence": aux["memory_sequence"],
            "tau_seconds": aux["tau_seconds"],
            "update": aux["update"],
            "read": aux["read"],
        }
        return (output, context)

    def forward_block(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """Apply memory to mixture frames of a full prefix+mixture block."""
        output, _context = self.forward_block_with_context(temporal_features)
        return output

    def summary(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_summary)
