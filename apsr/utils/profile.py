"""MAC accounting for the exact APSR-P operators, including dense masked attention."""

from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict
import torch
from torch import nn


def _add(counter: Dict[str, int], key: str, value: int):
    counter[key] += int(value)


def _is_causal_attention_module(module):
    return type(module).__name__ == "MemoryOnlyCausalAttention"


def estimate_tse_macs(
    model: nn.Module,
    sample_rate: int = 8000,
    mix_seconds: float = 4.0,
    enroll_seconds: float = 2.0,
    device=None,
    dtype=torch.float32,
) -> Dict[str, Any]:
    """Neural MAC estimate. Excludes FFT, normalization, activations, softmax and elementwise work."""
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")
    if float(mix_seconds) <= 0.0 or float(enroll_seconds) <= 0.0:
        raise ValueError("mix_seconds and enroll_seconds must be positive")
    target_model = model.module if hasattr(model, "module") else model
    if device is None:
        try:
            device = next(target_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    mix_len = max(1, int(round(float(sample_rate) * float(mix_seconds))))
    enroll_len = max(1, int(round(float(sample_rate) * float(enroll_seconds))))
    counters: Dict[str, int] = defaultdict(int)
    handles = []

    def conv_hook(module, inputs, output):
        x = inputs[0]
        y = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            return
        if isinstance(module, nn.ConvTranspose2d):
            batch, in_channels, in_h, in_w = x.shape
            kh, kw = module.kernel_size
            macs = (
                batch
                * in_channels
                * in_h
                * in_w
                * (module.out_channels // module.groups)
                * kh
                * kw
            )
        else:
            batch, out_channels, out_h, out_w = y.shape
            kh, kw = module.kernel_size
            kernel_mul = module.in_channels // module.groups * kh * kw
            macs = batch * out_channels * out_h * out_w * kernel_mul
        _add(counters, module.__class__.__name__, macs)

    def linear_hook(module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        instances = x.numel() // max(1, module.in_features)
        _add(counters, "Linear", instances * module.in_features * module.out_features)

    def gru_hook(module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        if module.batch_first:
            batch, steps = (x.shape[0], x.shape[1])
        else:
            steps, batch = (x.shape[0], x.shape[1])
        directions = 2 if module.bidirectional else 1
        hidden = module.hidden_size
        macs = 0
        for layer_idx in range(module.num_layers):
            layer_input = module.input_size if layer_idx == 0 else hidden * directions
            macs += (
                batch
                * steps
                * directions
                * 3
                * (layer_input * hidden + hidden * hidden)
            )
        _add(counters, "GRU", macs)

    def attention_hook(module, inputs, output):
        if not _is_causal_attention_module(module):
            return
        x = inputs[0]
        if not torch.is_tensor(x) or x.dim() != 4:
            return
        batch, channels, frames, freqs = x.shape
        heads = int(getattr(module, "n_head", 1))
        q0 = getattr(module, "attn_conv_Q_0", None)
        if q0 is None:
            return
        q_channels = int(q0[0].out_channels)
        qk_dim = q_channels * freqs
        v_dim = channels // heads * freqs
        # P materializes the full score matrix before causal/prefix masking.
        macs = batch * heads * frames * frames * (qk_dim + v_dim)
        _add(counters, "CausalAttentionMatMul", macs)

    def enrollment_pooling_hook(module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(x) or x.dim() != 4:
            return
        batch, channels, frames, freqs = x.shape
        tokens = int(getattr(module, "num_tokens", 1))
        att_dim = int(getattr(module, "att_dim", 1))
        qk_macs = batch * tokens * frames * att_dim
        av_macs = batch * tokens * frames * channels * freqs
        _add(counters, "EnrollmentTokenPooling", qk_macs + av_macs)

    for module in target_model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.GRU):
            handles.append(module.register_forward_hook(gru_hook))
        elif _is_causal_attention_module(module):
            handles.append(module.register_forward_hook(attention_hook))
        elif module.__class__.__name__ == "GatedAttentiveEnrollmentPooling":
            handles.append(module.register_forward_hook(enrollment_pooling_hook))
    was_training = target_model.training
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    target_model.eval()
    try:
        with torch.inference_mode():
            mix = torch.randn(1, mix_len, device=device, dtype=dtype)
            enrollment = torch.randn(1, enroll_len, device=device, dtype=dtype)
            target_model(mix, enrollment)
    finally:
        for handle in handles:
            handle.remove()
        target_model.train(was_training)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    total_macs = int(sum(counters.values()))
    return {
        "total_macs": total_macs,
        "macs_per_second": total_macs / max(float(mix_seconds), 1e-12),
        "mix_seconds": float(mix_seconds),
        "enroll_seconds": float(enroll_seconds),
        "by_op": dict(sorted(counters.items())),
    }
