"""Shared refinement schedule and frame-local bounded corrections."""

from __future__ import annotations
import torch


class ProgressiveRefinement:
    @staticmethod
    def _split_block_output(
        block_output: torch.Tensor, prefix_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = block_output[:, :, :prefix_length, :]
        mixture = block_output[:, :, prefix_length + 1 :, :]
        return (prefix, mixture)

    def _run_block(
        self,
        block_index: int,
        prefix: torch.Tensor,
        separator: torch.Tensor,
        mixture: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_length = prefix.shape[2]
        block_input = torch.cat([prefix, separator, mixture], dim=2)
        block_output, _hidden = self.dp_blocks[block_index](block_input, h=None)
        return self._split_block_output(block_output, prefix_length)

    @staticmethod
    def _finite_float32(value: torch.Tensor) -> torch.Tensor:
        value32 = value.float()
        return torch.where(torch.isfinite(value32), value32, torch.zeros_like(value32))

    @classmethod
    def _frame_rms(cls, value: torch.Tensor, eps: float = 1e-08) -> torch.Tensor:
        """Stable float32 RMS over C,F only, preserving every time index."""
        if value.dim() != 4:
            raise ValueError(f"frame RMS expects (B,C,T,F), got {tuple(value.shape)}")
        eps = max(float(eps), 1e-08)
        finite = cls._finite_float32(value)
        scale = finite.abs().amax(dim=(1, 3), keepdim=True).clamp_min(eps)
        normalized = finite / scale
        rms = scale * normalized.square().mean(dim=(1, 3), keepdim=True).sqrt()
        return torch.nan_to_num(rms, nan=0.0, posinf=torch.finfo(torch.float32).max)

    def _anchor_preserving_input(
        self, anchor: torch.Tensor, previous: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the recurrent state into a frame-local radius around A.

        ``U = A + s_in * (H - A)`` where
        ``s_in = min(1, kappa * RMS_CF(A) / (RMS_CF(H-A) + eps))``.
        """
        if anchor.shape != previous.shape:
            raise ValueError("anchor and previous refinement state must match")
        anchor32 = self._finite_float32(anchor)
        previous32 = previous.float()
        previous32 = torch.where(torch.isfinite(previous32), previous32, anchor32)
        drift = previous32 - anchor32
        numerator = self.trust_kappa * self._frame_rms(anchor32)
        denominator = self._frame_rms(drift) + 1e-08
        scale = torch.clamp(numerator / denominator, min=0.0, max=1.0)
        scale = torch.nan_to_num(scale, nan=0.0, posinf=1.0, neginf=0.0)
        recurrent_input = anchor32 + scale * drift
        recurrent_input = torch.where(
            torch.isfinite(recurrent_input), recurrent_input, anchor32
        )
        return (recurrent_input.to(dtype=previous.dtype), scale)

    def _trust_region_update(
        self, previous: torch.Tensor, candidate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Admit a bounded frame-local fraction of a later B2 correction.

        ``H' = H + eta * s_out * (Z-H)`` where
        ``s_out = min(1, kappa * RMS_CF(H) / (RMS_CF(Z-H) + eps))``.
        """
        if previous.shape != candidate.shape:
            raise ValueError("previous and candidate refinement states must match")
        previous32 = self._finite_float32(previous)
        candidate32 = candidate.float()
        candidate32 = torch.where(torch.isfinite(candidate32), candidate32, previous32)
        correction = candidate32 - previous32
        numerator = self.trust_kappa * self._frame_rms(previous32)
        denominator = self._frame_rms(correction) + 1e-08
        trust = torch.clamp(numerator / denominator, min=0.0, max=1.0)
        trust = torch.nan_to_num(trust, nan=0.0, posinf=1.0, neginf=0.0)
        updated = previous32 + self.trust_eta * trust * correction
        updated = torch.where(torch.isfinite(updated), updated, previous32)
        return (updated.to(dtype=previous.dtype), trust)

    def _conditioned_blocks(self, mixture, prefix):
        """Run physical blocks 0,1,2,2,2,2,3. Each call starts fresh acoustic state.

        prefix: (B,K,F) with channels present as (B,C,K,F); mixture: (B,C,T,F).
        The enrollment state progresses across calls independently of mixture outputs.
        """
        separator = self.sep_token.to(
            device=mixture.device, dtype=mixture.dtype
        ).expand(mixture.shape[0], -1, -1, -1)
        prefix, mixture = self._run_block(0, prefix, separator, mixture)
        prefix, anchor = self._run_block(1, prefix, separator, mixture)
        prefix, state = self._run_block(2, prefix, separator, anchor)
        for _ in range(1, self.refinement_rounds):
            recurrent_input, _ = self._anchor_preserving_input(anchor, state)
            prefix, candidate = self._run_block(2, prefix, separator, recurrent_input)
            state, _ = self._trust_region_update(state, candidate)
        prefix, mixture = self._run_block(3, prefix, separator, state)
        return (mixture, prefix)
