"""Float64 evaluation metrics; training uses its historical float32 loss."""

import warnings
import numpy as np

try:
    import mir_eval.separation
except ImportError:
    mir_eval = None
try:
    from pesq import pesq
except ImportError:
    pesq = None
try:
    from pystoi import stoi
except ImportError:
    stoi = None


def compute_si_sdr_np(
    estimate: np.ndarray, target: np.ndarray, eps: float = 1e-08
) -> float:
    estimate = estimate.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    dot = np.sum(estimate * target)
    target_energy = np.sum(target**2) + eps
    s_target = dot * target / target_energy
    e_noise = estimate - s_target
    return float(
        10.0 * np.log10((np.sum(s_target**2) + eps) / (np.sum(e_noise**2) + eps))
    )


def compute_projection_sdr_np(
    estimate: np.ndarray, target: np.ndarray, eps: float = 1e-08
) -> float:
    """Old projection SDR kept for auditing; this is close to SI-SDR."""
    estimate = estimate.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    alpha = np.sum(estimate * target) / (np.sum(target**2) + eps)
    s_target = alpha * target
    e_noise = estimate - s_target
    return float(
        10.0 * np.log10((np.sum(s_target**2) + eps) / (np.sum(e_noise**2) + eps))
    )


def compute_bss_eval_sdr_np(estimate: np.ndarray, target: np.ndarray) -> float:
    """Vincent et al. BSS Eval SDR for paper-style SDR reporting."""
    reference = target.astype(np.float64, copy=False)[None, :]
    estimated = estimate.astype(np.float64, copy=False)[None, :]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdr, _, _, _ = mir_eval.separation.bss_eval_sources(
                reference, estimated, compute_permutation=False
            )
        return float(sdr[0])
    except Exception:
        return float("nan")


def safe_pesq(sample_rate: int, target: np.ndarray, estimate: np.ndarray) -> float:
    if pesq is None:
        return float("nan")
    mode = "nb" if int(sample_rate) == 8000 else "wb"
    try:
        return float(pesq(int(sample_rate), target, estimate, mode))
    except Exception:
        return float("nan")


def safe_estoi(sample_rate: int, target: np.ndarray, estimate: np.ndarray) -> float:
    if stoi is None:
        return float("nan")
    try:
        return float(stoi(target, estimate, int(sample_rate), extended=True))
    except Exception:
        return float("nan")
