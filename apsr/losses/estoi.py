"""Modern-PyTorch compatibility wrapper for differentiable (E)STOI.

``torch_stoi`` 0.2.3 owns an old torchaudio resampler.  This wrapper keeps the
upstream loss equations and differentiable RFFT implementation, but performs
differentiable resampling explicitly for current torchaudio releases.

This module is training-only.  Paper metrics must continue to use the
repository evaluator's independent ``pystoi`` implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_STOI_INTERNAL_SAMPLE_RATE = 10_000


def build_differentiable_estoi_loss(
    *,
    sample_rate: int,
    extended: bool,
    use_vad: bool,
) -> nn.Module:
    """Build negative (E)STOI with explicit current-torchaudio resampling."""

    try:
        import torchaudio.functional as audio_functional
        from torch_stoi import NegSTOILoss
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Differentiable eSTOI training requires torch_stoi and "
            "torchaudio. Install the architecture-specific requirements "
            "file requirements-train.txt in the active "
            "PyTorch environment before launching this configuration."
        ) from error

    input_sample_rate = int(sample_rate)
    if input_sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    upstream = NegSTOILoss(
        sample_rate=_STOI_INTERNAL_SAMPLE_RATE,
        use_vad=bool(use_vad),
        extended=bool(extended),
        do_resample=False,
    )

    class _ResampledNegESTOI(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.loss = upstream

        def forward(
            self,
            estimate: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            if estimate.shape != target.shape:
                raise ValueError("eSTOI estimate and target shapes must match")
            estimate32 = estimate.float()
            target32 = target.float()
            if input_sample_rate != _STOI_INTERNAL_SAMPLE_RATE:
                estimate32 = audio_functional.resample(
                    estimate32,
                    input_sample_rate,
                    _STOI_INTERNAL_SAMPLE_RATE,
                )
                target32 = audio_functional.resample(
                    target32,
                    input_sample_rate,
                    _STOI_INTERNAL_SAMPLE_RATE,
                )
            return self.loss(estimate32, target32)

    return _ResampledNegESTOI()


__all__ = ["build_differentiable_estoi_loss"]
