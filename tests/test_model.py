"""Public regression tests for P's topology, conditioning boundary and recurrence."""

from pathlib import Path

import pytest
import torch

from apsr import APSRP
from apsr.models.memory import TargetAnchoredTimeConstantMemory
from apsr.models.blocks import MemoryOnlyCausalAttention

torch.set_num_threads(2)


def test_schema_and_parameter_counts():
    model = APSRP()
    assert len(model.state_dict()) == 398
    assert sum(p.numel() for p in model.parameters()) == 312986
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 306842
    state = model.state_dict()
    state.pop(next(iter(state)))
    with pytest.raises((ValueError, RuntimeError)):
        model.load_state_dict(state)


def test_refinement_schedule_and_gradient():
    model = APSRP()
    calls = []
    handles = [
        block.register_forward_hook(lambda m, a, o, i=i: calls.append(i))
        for i, block in enumerate(model.dp_blocks)
    ]
    estimate = model(torch.randn(2, 1024), torch.randn(1, 1536))
    assert estimate.shape == (2, 1, 1024)
    assert calls == [0, 1, 2, 2, 2, 2, 3]
    (
        estimate.square().mean() + model._last_aux["target_spectrum_ri"].square().mean()
    ).backward()
    assert model.dp_blocks[2].target_memory.output_proj.weight.grad is not None
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    for h in handles:
        h.remove()


def test_attention_isolates_prefix_and_future():
    torch.manual_seed(12)
    attention = MemoryOnlyCausalAttention(8, 5).eval()
    attention.direct_prefix_tokens = 3
    x = torch.randn(1, 8, 13, 5)
    changed = x.clone()
    changed[:, :, :3] += torch.randn_like(changed[:, :, :3]) * 5
    with torch.inference_mode():
        original = attention(x)
        different_prefix = attention(changed)
        changed = x.clone()
        changed[:, :, 9:] += torch.randn_like(changed[:, :, 9:]) * 5
        different_future = attention(changed)
    torch.testing.assert_close(
        original[:, :, 3:], different_prefix[:, :, 3:], atol=0, rtol=0
    )
    torch.testing.assert_close(
        original[:, :, :9], different_future[:, :, :9], atol=0, rtol=0
    )


def test_memory_sequence_matches_causal_steps():
    torch.manual_seed(2)
    memory = TargetAnchoredTimeConstantMemory(
        8, 4, num_enroll_tokens=3, zero_init_output=False
    )
    anchor = memory.make_anchor(torch.randn(2, 3, 5, 8))
    x = torch.randn(2, 19, 5, 8)
    sequence, final, _ = memory.forward_sequence(x, anchor)
    state = None
    outputs = []
    for frame in x.unbind(1):
        output, state, _ = memory.forward_step(frame, anchor, state)
        outputs.append(output)
    torch.testing.assert_close(sequence, torch.stack(outputs, 1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(final, state, rtol=1e-5, atol=1e-6)


def test_paper_initialization_is_untrained_and_loadable():
    root = Path(__file__).resolve().parents[1]
    initial = torch.load(
        root / "checkpoints/paper_initialization_seed43.pt", weights_only=True
    )
    model = APSRP()
    model.load_state_dict(initial["model"])
    assert initial["seed"] == 43
    assert "optimizer" not in initial
    for block in model.dp_blocks:
        assert torch.count_nonzero(block.target_memory.output_proj.weight) == 0
