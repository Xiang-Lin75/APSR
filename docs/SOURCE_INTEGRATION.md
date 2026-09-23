# Source-equivalence verification

The public package reorganizes the evaluated APSR-P research implementation into a standalone main-model package. The recorded integration checks used Python 3.13 and PyTorch/torchaudio 2.7.0+cu118 on CPU and CUDA. The historical quality benchmark used a different runtime, documented in the [model card](MODEL_CARD.md).

## Recorded checks

| Check | Recorded result |
|---|---|
| Checkpoint loading | All 398 tensors load strictly with identical names, shapes and values |
| Model size | 306,842 trainable; 312,986 registered parameters |
| Forward equivalence | Eight cases: three synthetic cases and one full-length paired speech case on each of CPU and CUDA; exact output equality |
| Backward comparison | Tested gradients and parameter order matched |
| Dense neural MACs | Both implementations: 19,690,951,728 for a 4-s mixture and 3-s enrollment |
| Package installation | A built wheel loaded E102 and ran inference in an isolated directory |

The machine-readable report is [model_parity.json](verification/model_parity.json). Public tests cover model topology, prefix/future isolation, memory recurrence, trial pairing, RNG restoration and mixture-baseline subtraction. [GitHub Actions](https://github.com/Xiang-Lin75/APSR-P/actions/workflows/tests.yml) reports current test runs separately from these recorded integration checks.

## Validation scope

Integration exercised manifest preparation, the three training-loss terms, epoch-boundary resume, primary-metric evaluation and WAV inference using smoke checks. It did not repeat multi-day training or the full audio benchmark. Full PESQ evaluation was not exercised in the recorded Windows integration environment because its C++ build dependency was unavailable.

The frozen 6,000-query records remain unchanged and can be checked with `python evaluation/verify_results.py`. That verifier recomputes statistics from CSV records; it does not rerun extraction.

The public forward omits unused historical diagnostics and auxiliary sink-waveform reconstruction while preserving target output and neural MACs. Measure its runtime separately as described in [Efficiency](EFFICIENCY.md). Initialization provenance and resume limits are documented once in [Training](TRAINING.md).
