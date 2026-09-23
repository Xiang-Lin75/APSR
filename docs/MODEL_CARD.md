# Released model

## Checkpoint

| Field | Value |
|---|---|
| Task | Enrollment-conditioned target speaker extraction |
| Implementation | `apsr.APSRP`, the main P model |
| Training data | WSJ0-2mix, 8 kHz, minimum-length mixtures |
| Training seed / shared refinement calls | 43 / 4 |
| Selection | Highest validation SI-SDRi, epoch 102 |
| Validation SI-SDRi | 14.578261 dB |
| Trainable / registered parameters | 306,842 / 312,986 |
| Weights | `apsr_p_wsj0_2mix_r4_seed43_e102.pt` |

Use the [checkpoint guide](../checkpoints/README.md) to download and verify the weights. [inference_config.json](../checkpoints/inference_config.json) records the public constructor and signal settings. Source/export digests and training provenance are retained in the [full-precision summary](../evaluation/wsj0-2mix/summary.json); the source training commit belongs to the original research repository, not this public repository.

## Inputs and method

Inputs are a mono 8-kHz mixture and a separate utterance from the desired speaker. The output is an estimated target waveform aligned with the mixture. The model uses a 256-sample STFT window, 64-sample hop, 64 feature channels, eight enrollment tokens and 16-dimensional memory.

Enrollment tokens retain frequency structure and progress across block calls. Within each call, gated memory accumulates mixture-dependent candidates around a fixed enrollment anchor. The [code map](CODE_MAP.md) connects these operations to the implementation.

## Evaluation

The released checkpoint was evaluated on 3,000 WSJ0-2mix test mixtures with both target queries, giving 6,000 queries.

| SI-SDRi (dB) | SDR (dB) | PESQ | eSTOI (%) | TCR (%) |
|---:|---:|---:|---:|---:|
| 13.81 | 14.32 | 3.10 | 87.82 | 1.95 |

Both queries have positive target margins for 2,886/3,000 mixtures (96.20%). The benchmark used PyTorch 2.10.0+cu128, CUDA 12.8 and an RTX 3090, with one mixture/two target queries per batch. Follow the [evaluation protocol](EVALUATION.md), including paired enrollment padding. [Per-query records and verification](../evaluation/README.md) provide full precision and checkpoint digests.

## Scope

This checkpoint release reports one training seed on clean two-speaker WSJ0-2mix. The package includes a WHAM! training recipe, but no WHAM! checkpoint or ablation variants are released here.

Neural operations use current and past mixture frames. Centered STFT analysis adds 16 ms of lookahead. Whole-utterance inference does not establish total output delay or per-hop streaming performance; dense attention cost and state grow with sequence length. See [efficiency measurements](EFFICIENCY.md).

The [integration report](SOURCE_INTEGRATION.md) distinguishes source-equivalence checks from full retraining and benchmark evaluation. The [data notice](DATA_NOTICE.md) covers the short demo excerpts and corpus rights.
