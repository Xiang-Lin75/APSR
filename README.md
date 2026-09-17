# APSR-P

**Enrollment-Anchored Memory and Progressive Refinement for Causal Target Speaker Extraction**

[Demo](https://xiang-lin75.github.io/APSR-P/) · [Pretrained model](https://github.com/Xiang-Lin75/APSR-P/releases/tag/wsj0-2mix-r4-seed43-e102) · [Documentation](docs/README.md)

APSR-P extracts a target speaker using a separate enrollment utterance. Frequency-resolved enrollment references condition causal mixture memory, while the enrollment state progresses across parameter-shared refinement calls. This repository provides the main P model, training, evaluation, and inference.

![APSR-P architecture](docs/figures/apsr_overview.png)

The neural separator uses current and past mixture frames; centered STFT analysis adds 16 ms of lookahead. Whole-utterance inference does not establish end-to-end streaming latency. See the [model and results](docs/MODEL_CARD.md).

## Installation

Use Python 3.10+ with a suitable [PyTorch build](https://pytorch.org/get-started/locally/) for your CPU or CUDA device.

```bash
git clone https://github.com/Xiang-Lin75/APSR-P.git
cd APSR-P
python -m pip install -r requirements.txt
```

## Quick Start

Download the validation-selected WSJ0-2mix E102 checkpoint and extract speech from an included example:

```bash
python scripts/download_checkpoint.py
python inference.py \
  --mixture site/audio/example-01-mixture.wav \
  --enrollment site/audio/example-01-a-enrollment.wav \
  --checkpoint checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt \
  --output outputs/target.wav --device auto
```

Replace the input paths with your own mono 8-kHz WAVs. See [inference and Python API](docs/INFERENCE.md).

### Training

Follow [data preparation](docs/DATA.md), then run:

```bash
python -m pip install -r requirements-train.txt
python train.py --config configs/wsj0_2mix.yaml --output runs/p-wsj0 --device cuda
```

For WHAM!, use `configs/wham.yaml`. Read the [training recipe](docs/TRAINING.md) for initialization, seed handling, and resume behavior.

### Evaluation

```bash
python -m pip install -r requirements-eval.txt
python evaluate.py \
  --checkpoint checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt \
  --manifest data/wsj0_2mix/tt --output outputs/p-e102-test --device cuda
```

See the [paired-query protocol](docs/EVALUATION.md) and [published results](evaluation/README.md). Demo excerpts are not benchmark inputs.

## License

The source-code license is [awaiting author confirmation](LICENSE_STATUS.md). [Software dependencies](docs/THIRD_PARTY_NOTICES.md) and [demo audio](docs/DATA_NOTICE.md) retain their respective terms.
