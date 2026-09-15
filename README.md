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

## Installation

Use Python 3.10+ and install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) for your CPU or CUDA device.

```bash
git clone https://github.com/Xiang-Lin75/APSR-P.git
cd APSR-P
python -m pip install -r requirements.txt
```

## Quick Start

### Extract with the pretrained model

Download the E102 weights and try the included audio example:

```bash
python scripts/download_checkpoint.py
python inference.py --mixture site/audio/example-01-mixture.wav --enrollment site/audio/example-01-a-enrollment.wav --checkpoint checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt --output outputs/target.wav --device auto
```

Replace the two input paths with your own **mono 8-kHz WAVs**. The extracted speech is saved to `outputs/target.wav`. [Inference and Python API](docs/INFERENCE.md).

### Train

Prepare the audio and manifests using [Data preparation](docs/DATA.md), then run:

```bash
python -m pip install -r requirements-train.txt
python train.py --config configs/wsj0_2mix.yaml --output runs/p-wsj0 --device cuda
```

Use `configs/wham.yaml` for WHAM!. [Full recipe and resume instructions](docs/TRAINING.md).

### Evaluate

```bash
python -m pip install -r requirements-eval.txt
python evaluate.py --checkpoint checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt --manifest data/wsj0_2mix/tt --output outputs/p-e102-test --device cuda
```

[Evaluation protocol](docs/EVALUATION.md) · [MACs and runtime](docs/EFFICIENCY.md).

## Documentation

| Guide | Contents |
|---|---|
| [Code map](docs/CODE_MAP.md) | Model modules and their correspondence to the method |
| [Data preparation](docs/DATA.md) | WSJ0-2mix / WHAM! layout and paired trial manifests |
| [Training](docs/TRAINING.md) | Loss, initialization, scheduler and checkpoint resume |
| [Inference](docs/INFERENCE.md) | WAV extraction, Python API and benchmark commands |
| [Evaluation](docs/EVALUATION.md) | Metrics and test protocol |
| [Checkpoint](checkpoints/README.md) | E102 weights and verification |
| [Source integration](docs/SOURCE_INTEGRATION.md) | Compatibility tests and validation scope |

## License status

The source license is awaiting author confirmation; see [LICENSE_STATUS.md](LICENSE_STATUS.md). Dependencies and audio retain their respective terms; see [third-party notices](THIRD_PARTY_NOTICES.md) and [data notice](DATA_NOTICE.md).
