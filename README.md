# APSR

**Frequency-Resolved Enrollment-Anchored Memory for Causal Target Speaker Extraction**

[Audio demo](https://xiang-lin75.github.io/APSR/) · [Pretrained model](https://github.com/Xiang-Lin75/APSR/releases/tag/wsj0-2mix-r4-seed43-e102) · [Results](docs/MODEL_CARD.md#evaluation) · [Documentation](docs/README.md)

APSR (Anchor-Preserving Spectrotemporal Refinement) extracts a target speaker from a mixture using a separate enrollment utterance. Frequency-resolved enrollment anchors condition causal memory, while enrollment tokens evolve across shared refinement calls. This repository provides the main model, pretrained WSJ0-2mix weights, and training, evaluation and inference tools.

![APSR architecture](docs/figures/apsr_overview.png)

## Installation

Use Python 3.10+ and a [PyTorch build](https://pytorch.org/get-started/locally/) for your CPU or CUDA device.

```bash
git clone https://github.com/Xiang-Lin75/APSR.git
cd APSR
python -m pip install -e .
```

## Quick start

Download the pretrained checkpoint and extract a speaker from an included example:

```bash
python scripts/download_checkpoint.py
python inference.py \
  --mixture site/audio/example-01-mixture.wav \
  --enrollment site/audio/example-01-a-enrollment.wav \
  --checkpoint checkpoints/apsr_wsj0_2mix_r4_seed43_e102.pt \
  --output outputs/target.wav --device auto
```

Use mono 8-kHz WAVs for your own inputs. Neural processing is causal; centered STFT analysis adds 16 ms of lookahead. See [inference and Python API](docs/INFERENCE.md) and [streaming scope](docs/MODEL_CARD.md#scope).

### Training

Prepare [data and manifests](docs/DATA.md), then run:

```bash
python -m pip install -e ".[train]"
python train.py --config configs/wsj0_2mix.yaml --output runs/p-wsj0 --device cuda
```

For WHAM!, use `configs/wham.yaml`. See the [training recipe](docs/TRAINING.md).

### Evaluation

```bash
python -m pip install -e ".[eval]"
python evaluate.py \
  --checkpoint checkpoints/apsr_wsj0_2mix_r4_seed43_e102.pt \
  --manifest data/wsj0_2mix/tt --output outputs/p-test --device cuda
```

Use the full-utterance [paired-query protocol](docs/EVALUATION.md) to reproduce the [published records](evaluation/README.md).

## Citation and questions

Software citation metadata is provided in [CITATION.cff](CITATION.cff). For questions or reproducible bug reports, open a [GitHub issue](https://github.com/Xiang-Lin75/APSR/issues).

## License

APSR's original code and model weights use the [MIT License](LICENSE). Adapted components retain their [upstream licenses](docs/THIRD_PARTY_NOTICES.md); [audio recordings](docs/DATA_NOTICE.md) retain their original rights.
