"""Canonical waveform SI-SDR and pre-iSTFT normalized spectral loss."""

import math
import random
import numpy as np
import torch


@torch.no_grad()
def clip_grad_norm_with_diagnostics(model, max_norm, *, error_if_nonfinite=True):
    """Clip gradients while distinguishing bad elements from norm overflow.

    PyTorch computes the standard aggregate norm in the gradient dtype.  A
    collection of individually finite fp32 gradients can therefore produce an
    infinite fp32 norm.  In that narrow case, recompute the norm in float64
    and apply the intended clipping scale.  Actual NaN/Inf gradient elements
    still fail closed, now with the responsible parameter names.
    """
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    parameters = [parameter for _name, parameter in named_parameters]
    try:
        return torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, error_if_nonfinite=error_if_nonfinite
        )
    except RuntimeError as original_error:
        message = str(original_error).lower()
        if (
            not error_if_nonfinite
            or "total norm" not in message
            or "non-finite" not in message
        ):
            raise
        invalid = []
        finite_norms = []
        reference = None
        for name, parameter in named_parameters:
            gradient = parameter.grad.detach()
            reference = gradient if reference is None else reference
            finite = torch.isfinite(gradient)
            if not bool(finite.all()):
                nan_count = int(torch.isnan(gradient).sum().item())
                posinf_count = int(torch.isposinf(gradient).sum().item())
                neginf_count = int(torch.isneginf(gradient).sum().item())
                finite_values = gradient[finite]
                finite_max = (
                    float(finite_values.abs().max().item())
                    if finite_values.numel()
                    else float("nan")
                )
                invalid.append(
                    f"{name}(nan={nan_count},+inf={posinf_count},-inf={neginf_count},finite_max={finite_max:.3e})"
                )
                continue
            finite_norms.append(
                float(torch.linalg.vector_norm(gradient.double()).item())
            )
        if invalid:
            preview = "; ".join(invalid[:16])
            remainder = max(0, len(invalid) - 16)
            if remainder:
                preview += f"; ... and {remainder} more"
            raise FloatingPointError(
                "non-finite gradient elements detected in " + preview
            ) from original_error
        stable_norm = math.sqrt(math.fsum((value * value for value in finite_norms)))
        if not math.isfinite(stable_norm):
            raise FloatingPointError(
                "gradient elements are finite but the float64 aggregate norm is non-finite"
            ) from original_error
        clip_scale = min(1.0, float(max_norm) / (stable_norm + 1e-12))
        for _name, parameter in named_parameters:
            parameter.grad.mul_(clip_scale)
        if reference is None:
            return torch.tensor(0.0)
        return torch.tensor(stable_norm, device=reference.device, dtype=torch.float64)


def seed_dataloader_worker(worker_id):
    """Seed Python/NumPy from PyTorch's worker seed without global coupling."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def compute_si_snr(estimate, target, eps=1e-08):
    """Compute SI-SNR. Matches Asteroid/TIGER's SingleSrcNegSDR('sisdr') exactly."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    s_target_energy = (target**2).sum(dim=-1, keepdim=True) + eps
    s_target = dot * target / s_target_energy
    e_noise = estimate - s_target
    si_snr = 10 * torch.log10(
        (s_target**2).sum(dim=-1) / ((e_noise**2).sum(dim=-1) + eps) + eps
    )
    return si_snr


def compute_normalized_mag_ri_loss(
    estimate_ri, target_ri, *, magnitude_weight=0.5, ri_weight=0.5, eps=1e-08
):
    """Energy-normalized magnitude plus complex RI reconstruction loss.

    Both inputs must already be represented on the *same model analysis
    grid* as ``(B, 2, T, F)`` with real and imaginary components at channel
    indices zero and one.  Keeping STFT construction outside this function is
    deliberate: APSR-P supplies its pre-iSTFT target estimate and constructs the
    clean target with ``model.training_stft_ri`` so the loss cannot silently
    drift from the model's window, hop, centering, or frame-count convention.

    Normalization is performed per utterance using clean-target spectral
    energy before averaging the batch.  Thus quiet utterances do not vanish
    from the objective and loud utterances do not dominate it.
    """
    if not torch.is_tensor(estimate_ri) or not torch.is_tensor(target_ri):
        raise TypeError("estimate_ri and target_ri must be torch tensors")
    if estimate_ri.ndim != 4 or estimate_ri.shape[1] != 2:
        raise ValueError(
            f"estimate_ri must have shape (B,2,T,F), got {tuple(estimate_ri.shape)}"
        )
    if target_ri.ndim != 4 or target_ri.shape[1] != 2:
        raise ValueError(
            f"target_ri must have shape (B,2,T,F), got {tuple(target_ri.shape)}"
        )
    if estimate_ri.shape != target_ri.shape:
        raise ValueError(
            f"estimate_ri and target_ri must use the identical STFT grid: got {tuple(estimate_ri.shape)} and {tuple(target_ri.shape)}"
        )
    if estimate_ri.device != target_ri.device:
        raise ValueError(
            f"estimate_ri and target_ri must be on the same device: got {estimate_ri.device} and {target_ri.device}"
        )
    magnitude_weight = float(magnitude_weight)
    ri_weight = float(ri_weight)
    eps = float(eps)
    if magnitude_weight < 0.0 or ri_weight < 0.0:
        raise ValueError("magnitude_weight and ri_weight must be non-negative")
    if not math.isclose(magnitude_weight + ri_weight, 1.0, abs_tol=1e-08):
        raise ValueError("magnitude_weight + ri_weight must equal 1")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a finite positive value")
    estimate = estimate_ri.float()
    target = target_ri.float()
    reduce_dims = (1, 2, 3)
    target_energy = target.square().sum(dim=reduce_dims).clamp_min(eps)
    ri_error = (estimate - target).square().sum(dim=reduce_dims) / target_energy
    estimate_magnitude = torch.sqrt(
        estimate[:, 0].square() + estimate[:, 1].square() + eps
    )
    target_magnitude = torch.sqrt(target[:, 0].square() + target[:, 1].square() + eps)
    magnitude_error = (estimate_magnitude - target_magnitude).square().sum(
        dim=(1, 2)
    ) / target_energy
    return (ri_weight * ri_error + magnitude_weight * magnitude_error).mean()
