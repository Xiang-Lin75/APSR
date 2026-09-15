"""Final-output three-term objective used by the P training recipe."""

import torch
from torch import nn
from .operators import compute_si_snr, compute_normalized_mag_ri_loss
from .estoi import build_differentiable_estoi_loss


class TrainingObjective(nn.Module):
    def __init__(self, lambda_spectral=0.1, lambda_estoi=0.1):
        super().__init__()
        self.lambda_spectral = float(lambda_spectral)
        self.lambda_estoi = float(lambda_estoi)
        self.estoi = build_differentiable_estoi_loss(
            sample_rate=8000, extended=True, use_vad=True
        )

    def forward(self, model, estimate, target):
        si = -compute_si_snr(estimate[:, 0], target).mean()
        spectral = compute_normalized_mag_ri_loss(
            model._last_aux["target_spectrum_ri"],
            model.training_stft_ri(target),
            magnitude_weight=0.5,
            ri_weight=0.5,
            eps=1e-8,
        )
        intelligibility = self.estoi(estimate[:, 0].float(), target.float()).mean()
        loss = (
            si + self.lambda_spectral * spectral + self.lambda_estoi * intelligibility
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training objective")
        return loss, dict(
            negative_si_sdr=float(si.detach()),
            normalized_spectral=float(spectral.detach()),
            negative_estoi=float(intelligibility.detach()),
        )
