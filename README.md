# APSR-P

**Enrollment-Anchored Memory and Progressive Refinement for Causal Target Speaker Extraction**

[Audio demo](https://xiang-lin75.github.io/APSR-P/) · [Pretrained checkpoint](https://github.com/Xiang-Lin75/APSR-P/releases/tag/wsj0-2mix-r4-seed43-e102) · [Code map](docs/CODE_MAP.md) · [Training](docs/TRAINING.md) · [Evaluation](evaluation/README.md)

APSR-P extracts a target speaker from a mixture using a separate enrollment utterance. It combines frequency-resolved enrollment references, an enrollment-anchored causal memory interface, and shared-block progressive refinement in the time–frequency domain.

## Architecture

![APSR-P overview and separator-block details](docs/figures/apsr_overview.png)

The shared encoder processes mixture and enrollment independently. Gated pooling preserves frequency information in enrollment tokens. Separator blocks model frequency and time, read enrollment-conditioned memory, and apply causal self-attention. The decoder fuses the resulting features with the mixture encoder's skip features and estimates a complex target mask.

- **Enrollment-anchored memory:** a frequency-indexed reference guides candidate content, adaptation rate, and gated readout. A bounded candidate and convex update keep the latent memory close to its reference within each block call.
- **Progressive enrollment state:** enrollment tokens evolve through shared refinement calls, while each call maintains its own acoustic-time memory trajectory. Memory provides the enrollment-to-mixture conditioning route.
- **Shared refinement:** four calls to a shared block refine mixture features with bounded corrections and no additional parameter set per call.

<details>
<summary><strong>Memory detail</strong></summary>

<p align="center"><img src="docs/figures/apsr_memory.png" width="420" alt="Enrollment-anchored causal memory: anchor, candidate, gated recurrence, and residual readout"></p>

The reference is fixed over mixture time within a call. Candidate saturation bounds latent displacement; it does not by itself guarantee correct speaker selection. Target margin and target-confusion rate assess that behavior empirically.

</details>

Vector figures: [overview PDF](docs/figures/apsr_overview.pdf) · [memory PDF](docs/figures/apsr_memory.pdf).

Editable figures: [overview Draw.io](docs/figures/apsr_overview.drawio) · [memory Draw.io](docs/figures/apsr_memory.drawio).

Neural operations use current and past mixture frames. The released configuration uses a centered 32-ms STFT window with 16-ms analysis lookahead. End-to-end streaming latency is not established by the offline test below.

## WSJ0-2mix test

Validation-selected **epoch 102**, **R4**, **seed 43**; 8 kHz, minimum-length mixtures. The full test evaluates both targets of all **3,000 mixtures**, yielding **6,000 queries**.

| SI-SDRi ↑ (dB) | SDR ↑ (dB) | SDRi ↑ (dB) | PESQ ↑ | eSTOI ↑ (%) | TCR ↓ (%) |
|---:|---:|---:|---:|---:|---:|
| 13.81 | 14.32 | 14.17 | 3.10 | 87.82 | 1.95 |

306,842 trainable parameters; 312,986 registered parameters. TCR is 117/6,000. Each improvement subtracts the corresponding mixture score for that query. These are the frozen full-test results for one training seed. This code release contains the APSR-P main model and its WSJ0-2mix/WHAM! training recipes; experimental model variants are outside its scope.

[Exact summary](evaluation/wsj0-2mix/summary.json) · [Per-query metrics](evaluation/wsj0-2mix/metrics.csv) · [Portable trial list](evaluation/wsj0-2mix/trials.csv)

## Checkpoint and reproducibility

Download **`apsr_p_wsj0_2mix_r4_seed43_e102.pt`** from the [E102 release](https://github.com/Xiang-Lin75/APSR-P/releases/tag/wsj0-2mix-r4-seed43-e102). It contains inference parameters and buffers only. The release also provides the matching inference configuration, model card, test summary, tensor inventory, export verification, and SHA256 checksums.

The exported state was checked tensor by tensor against the evaluated E102 checkpoint. Strict loading and a full-length paired-query forward comparison passed with identical outputs in the same runtime. The conversion does not replace or rerun the 6,000-query benchmark.

The standalone main model is implemented in [`apsr/models/apsr.py`](apsr/models/apsr.py). Its components, training objective, paired-query evaluator, and data-preparation tools are included in this repository. The artifact verifier below needs only Python's standard library:

```bash
python evaluation/verify_results.py
```

It verifies checksums, query pairing, mixture-baseline subtraction, identity decisions, and all published metric means. See [checkpoint details](checkpoints/README.md) and [evaluation protocol](evaluation/README.md).

## Installation

Python 3.10 or later is required. Install matching PyTorch and torchaudio builds for your CPU/CUDA environment first; see the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The integration was tested with Python 3.13 and PyTorch/torchaudio 2.7.0+cu118. The historical full test used PyTorch 2.10.0+cu128. Numerical results can differ across backends.

Run these commands from this repository's root:

```bash
python -m pip install -r requirements.txt
python scripts/download_checkpoint.py
python inference.py --mixture mixture.wav --enrollment enrollment.wav --checkpoint checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt --output outputs/target.wav --device auto
```

Both inputs must be mono 8-kHz WAV files. The CLI writes a float WAV of the same length as the mixture. It does not need a clean target or silently resample audio.

Python API:

```python
import torch
from apsr import APSRP

model = APSRP.from_pretrained("checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt", device="cpu")
# mixture: (batch, samples); enrollment: (batch, enrollment_samples)
with torch.inference_mode():
    target = model(mixture, enrollment)  # (batch, 1, samples)
```

## Repository structure

```text
apsr/models/       Main model, encoder/decoder, pooling, separator, memory
apsr/data/         Dataset loading, paired targets and training sampling
apsr/losses/       SI-SDR, normalized spectral reconstruction and eSTOI
apsr/training/     Optimizer/scheduler, validation selection and resume
apsr/metrics/      Evaluation metric definitions
apsr/utils/        Checkpoint/runtime helpers and MAC accounting
configs/           WSJ0-2mix and WHAM! recipes for P
protocols/         Portable training/validation trial identities
scripts/           Manifest preparation and verified weight download
checkpoints/       Weight metadata and untrained source-initialization fixture
evaluation/        Frozen E102 results and portable test queries
docs/              Code map, data/training instructions and figures
tests/             Topology, conditioning, recurrence and protocol checks
site/              Existing listening demo
```

Start with `APSRP.forward()`, then read `separator.py` for the shared refinement schedule and `memory.py` for the acoustic-time recurrence. [Code-to-method map](docs/CODE_MAP.md).

## Training and evaluation

Prepare licensed audio using [DATA.md](docs/DATA.md), then install the additional dependencies:

```bash
python -m pip install -r requirements-train.txt
python train.py --config configs/wsj0_2mix.yaml --output runs/p-wsj0 --device cuda
python train.py --config configs/wsj0_2mix.yaml --output runs/p-wsj0 --resume runs/p-wsj0/latest.pt --device cuda
```

For WHAM!, use `configs/wham.yaml` after preparing `mix_both`, `s1`, `s2` and enrollment paths. The WHAM! recipe disables clean-source rebalancing. [Training and initialization details](docs/TRAINING.md).

```bash
python -m pip install -r requirements-eval.txt
python evaluate.py --checkpoint checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt --manifest data/wsj0_2mix/tt --output outputs/p-e102-test --device cuda
```

Use `--primary-only` to compute SI-SDR/SI-SDRi and TCR without PESQ/eSTOI/SDR dependencies. Missing secondary metrics are recorded as `PENDING`. Checkpoints are selected using validation, never this test command. [Evaluation details](docs/EVALUATION.md).

```bash
python benchmark.py --checkpoint checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt --device cuda --output outputs/runtime.json
python -m pip install -e ".[test]"
python -m pytest
```

The benchmark reports synchronized whole-utterance RTF and neural MACs. [Measurement scope](docs/EFFICIENCY.md). Integration evidence is recorded in [SOURCE_INTEGRATION.md](docs/SOURCE_INTEGRATION.md).

## License status

The source license is awaiting author confirmation; see [LICENSE_STATUS.md](LICENSE_STATUS.md). This source candidate does not imply an MIT or Apache license. Dependencies and corpus audio retain their respective terms.

## Listening demo

The [demo website](https://xiang-lin75.github.io/APSR-P/) provides **eight paired-target listening examples** with **Enrollment · Mixture · APSR-P · Clean target**, spectrograms, and A/B target switching for the same mixture. Comparison excerpts are at most six seconds; enrollment excerpts are at most three seconds. Outputs were precomputed using E102. The authors confirmed the applicable permission for these short research-demo excerpts; see [data notice](DATA_NOTICE.md).

The listening layout was informed by the [Universal Speech Enhancement Hybrid demo](https://nanless.github.io/universal-speech-enhancement-demo/#hybrid). GitHub Pages deploys only `site/`; the method figures, checkpoint links, and test artifacts remain in this repository.
