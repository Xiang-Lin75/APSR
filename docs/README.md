# Documentation

Start with the [quick start](../README.md#quick-start). This package includes the main APSR-P model.

| Task | Guide |
|---|---|
| Prepare WSJ0-2mix / WHAM! and paired manifests | [Data](DATA.md) |
| Train or resume; understand initialization | [Training](TRAINING.md) |
| Extract audio or use the Python API | [Inference](INFERENCE.md) |
| Run the paired benchmark and interpret metrics | [Evaluation](EVALUATION.md) |
| Inspect the released model and its results | [Model card](MODEL_CARD.md), [per-query records](../evaluation/README.md) |
| Download and verify weights | [Checkpoints](../checkpoints/README.md) |
| Map paper components to implementation | [Code map](CODE_MAP.md) |
| Measure MACs and runtime | [Efficiency](EFFICIENCY.md) |
| Inspect source-equivalence checks | [Verification](SOURCE_INTEGRATION.md) |
| Check reuse terms and attribution | [License](../LICENSE), [software](THIRD_PARTY_NOTICES.md), [audio](DATA_NOTICE.md) |

## Repository layout

```text
apsr/          Model, data loading, losses, metrics and trainer
configs/       WSJ0-2mix and WHAM! recipes
protocols/     Fixed training and validation trial identities
checkpoints/   Weight metadata and untrained initialization fixture
scripts/       Checkpoint download and manifest preparation
tests/         Model and protocol regression checks
evaluation/    Per-query test records and statistics verifier
docs/          Guides, figures and verification records
site/          Audio demo
```

`train.py`, `evaluate.py` and `inference.py` are the main entry points. `benchmark.py` measures MACs and runtime. Training and evaluation dependencies are optional.

## Architecture figures

- Overview: [PDF](figures/apsr_overview.pdf) · [Draw.io](figures/apsr_overview.drawio)
- Memory: [PDF](figures/apsr_memory.pdf) · [Draw.io](figures/apsr_memory.drawio)
