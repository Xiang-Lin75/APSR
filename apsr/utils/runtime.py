"""Portable runtime helpers shared by the public entry points."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if sr != 8000 or audio.ndim != 1 or len(audio) <= 128:
        raise ValueError(f"Expected mono 8-kHz audio longer than 128 samples: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Nonfinite audio: {path}")
    return torch.from_numpy(audio)


def get_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("model") not in ("APSR", "APSR-P") or cfg.get("sample_rate") != 8000:
        raise ValueError("This release supports the 8-kHz APSR main model")
    cfg["model"] = "APSR"
    return cfg
