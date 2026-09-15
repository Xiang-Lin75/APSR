"""Protocol and safe resume serialization checks independent of speech corpora."""

import csv
from pathlib import Path
import random

import numpy as np
import torch

from apsr.metrics.objective import compute_si_sdr_np
from apsr.training.trainer import capture_rng, restore_rng, atomic_save


def test_portable_trial_pairing():
    root = Path(__file__).resolve().parents[1]
    for path, count in [
        (root / "protocols/wsj0_2mix/train.csv", 40000),
        (root / "protocols/wsj0_2mix/validation.csv", 10000),
        (root / "evaluation/wsj0-2mix/trials.csv", 6000),
    ]:
        with path.open(encoding="utf-8") as h:
            rows = list(csv.DictReader(h))
        assert len(rows) == count and len({r["key"] for r in rows}) == count
        groups = {}
        for r in rows:
            groups.setdefault(r["mixture_id"], []).append(r["target_source"])
            for k in ["mixture_relpath", "target_relpath", "enrollment_relpath"]:
                assert not Path(r[k]).is_absolute() and ".." not in Path(r[k]).parts
        assert all(sorted(v) == ["s1", "s2"] for v in groups.values())


def test_si_sdri_uses_measured_baseline():
    rng = np.random.default_rng(43)
    target = rng.normal(size=4096)
    interferer = rng.normal(size=4096)
    mixture = target + 2 * interferer
    output = target + 0.2 * interferer
    baseline = compute_si_sdr_np(mixture, target)
    improvement = compute_si_sdr_np(output, target) - baseline
    assert baseline < -5 and 19 < improvement < 21


def test_rng_resume_round_trip(tmp_path):
    random.seed(43)
    np.random.seed(43)
    torch.manual_seed(43)
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(3))
    atomic_save({"rng": state}, tmp_path / "checkpoint.pt")
    loaded = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    restore_rng(loaded["rng"])
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])
