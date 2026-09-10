# APSR-P E102 model card

## Model and checkpoint

| Field | Value |
|---|---|
| Task | Enrollment-conditioned target speaker extraction |
| Variant | `progressive_memory_only_r4` (`m5_ablation_v6`) |
| Training data | WSJ0-2mix, 8 kHz, minimum-length mixtures |
| Training seed / refinement rounds | 43 / 4 |
| Selection | Best validation SI-SDRi, epoch 102 |
| Validation SI-SDRi | 14.57826133685261 dB |
| Trainable / registered parameters | 306,842 / 312,986 |
| Weights | `apsr_p_wsj0_2mix_r4_seed43_e102.pt` |
| Source training commit | `894469ac03a3358cb159f0f26501c02b6b5a327f` |
| Source training checkpoint SHA256 | `43866ce6490cea19da3ab3883234656b48d8210e5f07fdc31816c54e26827e04` |

The source commit identifies the private frozen training implementation. It is not a commit in the public repository. The SHA256 of the inference-only export differs from the training checkpoint because the serialization excludes training state. Both digests are bound in `export_verification.json` and the release's `SHA256SUMS`.

## Inputs and output

Inputs are a mono 8-kHz mixture waveform and a separate utterance from the desired speaker. Output is an estimated target waveform aligned with the mixture. The frozen evaluation uses full mixture and enrollment utterances; paired enrollment queries are right-padded together by the evaluator. It does not use the 6-s/3-s listening-demo excerpts as benchmark inputs.

The configuration uses a 256-sample STFT window, 64-sample hop, 64 feature channels, eight enrollment tokens, 16-dimensional memory, and four shared refinement calls. `inference_config.json` preserves the exact constructor configuration, including compatibility and disabled-feature fields required by the frozen implementation. It contains no dataset or machine paths.

## Architecture and scope

Frequency-resolved enrollment tokens supply references for causal memory. Each block call anchors candidate generation, adaptive writes, and gated readout to its reference. Enrollment tokens progress along network depth independently of mixture errors; acoustic-time memory is reinitialized for each call. Shared-block corrections are bounded relative to mixture features.

Causality refers to neural processing in STFT frame order. Centered analysis introduces 16 ms of lookahead. The offline evaluation does not establish complete output delay, bounded streaming state, or cached per-hop execution performance. The published quality results are from a single training seed on clean two-speaker WSJ0-2mix. No WHAM! or ablation scores are claimed by this release.

## Verified test results

The test has 3,000 mixtures and 6,000 target queries. SI-SDRi is 13.814164 dB, SDR is 14.316960 dB, SDRi is 14.165916 dB, narrow-band PESQ is 3.102384, and eSTOI is 0.878152 (87.815238%). Target-confusion rate is 117/6,000 (1.95%), and mean target margin is 43.112424 dB. Both queries have positive target margin for 2,886/3,000 mixtures (96.2%).

The benchmark used PyTorch 2.10.0+cu128, CUDA 12.8, and an RTX 3090, with one mixture/two target queries per model batch. The export's paired forward equivalence check uses a separate recorded runtime and is not a new benchmark. Full precision results and protocol are available in the [evaluation directory](https://github.com/Xiang-Lin75/APSR-P/tree/main/evaluation).

## Availability

Inference weights, configuration, test artifacts, and integrity verification are released now. Architecture, training, inference, and audio-evaluation code are planned after acceptance. A plain state dictionary is not a standalone executable model and requires that architecture. The current artifact verifier recomputes published CSV statistics without running extraction.

No WSJ0 recordings, extracted recordings, optimizer/scheduler state, or deployment credentials are included. Corpus access and redistribution remain governed by the applicable data agreement; see the [data notice](https://github.com/Xiang-Lin75/APSR-P/blob/main/DATA_NOTICE.md).
