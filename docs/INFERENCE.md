# Inference and Python API

The command-line interface is the simplest way to extract a speaker:

```bash
python scripts/download_checkpoint.py
python inference.py --mixture mixture.wav --enrollment enrollment.wav --checkpoint checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt --output outputs/target.wav --device auto
```

Inputs must be mono 8-kHz WAVs. The output is a float WAV with the mixture's length. `--device auto` chooses CUDA when available, otherwise CPU. No target recording, API key or hosted service is required.

The README uses the supplied short listening excerpts for a runnable example. The website estimates were computed from full utterances before cropping; inference on cropped mixture/enrollment inputs need not reproduce those precomputed excerpts exactly. Use the full paired test protocol for benchmark evaluation.

## Python API

Use the local Python interface when integrating APSR-P into another program, rather than invoking the command-line script:

```python
import torch
from apsr import APSRP
from apsr.utils.runtime import read_audio

model = APSRP.from_pretrained(
    "checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt", device="cpu"
)
mixture = read_audio("mixture.wav").unsqueeze(0)
enrollment = read_audio("enrollment.wav").unsqueeze(0)
with torch.inference_mode():
    target = model(mixture, enrollment)
```

Input shapes are `(batch, samples)` and `(batch, enrollment_samples)`; output is `(batch, 1, samples)`. The tensors must be on the model's device. Loading validates checkpoint compatibility strictly. For CUDA, move both the model and input tensors to CUDA.

## Runtime and verification

```bash
python benchmark.py --checkpoint checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt --device cuda --output outputs/runtime.json
python evaluation/verify_results.py
python -m pip install -e ".[test]"
python -m pytest
```

See [measurement scope](EFFICIENCY.md), [checkpoint details](../checkpoints/README.md) and [integration checks](SOURCE_INTEGRATION.md). The artifact verifier checks the frozen CSV without running audio inference.
