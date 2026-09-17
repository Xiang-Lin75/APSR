# Main-model source integration

The public package reorganizes the evaluated APSR-P implementation into a standalone package. Only the main P model is included. The separation of model components, configurations, data preparation, training, evaluation and inference follows the organizational approach of [TIGER](https://github.com/JusperLee/TIGER); its trainer was not imported.

## Verified behavior

Integration checks used Python 3.13, PyTorch/torchaudio 2.7.0+cu118, CPU and CUDA. The historical benchmark used another runtime; cross-runtime equality is not implied.

- All 398 checkpoint tensors load strictly, with identical names, shapes and values. Parameter ordering is preserved.
- The model has 306,842 trainable parameters and 312,986 registered parameters.
- Eight forward comparisons against the original implementation passed with exactly equal outputs: three synthetic length/batch cases and one full-length paired speech case on each of CPU and CUDA.
- The tested backward pass matched the original gradients exactly.
- Both implementations produced 19,690,951,728 neural MACs for a 4-s mixture and 3-s enrollment: 4,922,737,932 MAC/s. This includes the actual dense attention score products before masking.
- Public tests exercise strict loading, refinement calls, prefix/future isolation, memory recurrence, paired trial identities, RNG restoration and mixture-baseline subtraction.
- A synthetic integration smoke check exercises manifest preparation, all three training-loss terms, epoch-boundary resume, primary-metric evaluation and WAV inference. It does not produce paper scores.
- A built wheel installed in an independent directory passed strict E102 loading and inference under Python isolated mode, without importing the historical research model package.

The detailed source-equivalence report is [model_parity.json](verification/model_parity.json). The untrained initialization fixture's provenance and limitations are explained in [TRAINING.md](TRAINING.md).

## Scope and remaining validation

The frozen 6,000-query E102 CSV and summary are retained unchanged and can be checked with `python evaluation/verify_results.py`. The reorganized evaluator has not rerun the full benchmark. No new multi-day training experiment was performed for integration.

Local PESQ installation was blocked by the missing Windows C++ build toolchain. The full PESQ path therefore remains untested in this integration environment; use a compatible environment for full secondary-metric evaluation. Primary metrics and the differentiable eSTOI training term were exercised. Current CI runs are available in [GitHub Actions](https://github.com/Xiang-Lin75/APSR-P/actions/workflows/tests.yml); the integration checks described above are separate from CI.

Whole-utterance runtime of the reorganized forward path must be measured separately: it omits unused historical diagnostics and sink waveform reconstruction. This does not change target output or neural MACs. See [EFFICIENCY.md](EFFICIENCY.md).

Source license selection remains pending author confirmation. Public availability of the code does not complete the license review.

## Documentation reorganization verification (2026-09-17)

The documentation cleanup left all model, training, evaluation, configuration, protocol, checkpoint and demo files byte-identical. Rechecking the public implementation against the research implementation with E102 weights passed all eight CPU/CUDA forward cases, including paired full-length speech, with exact output equality. The tested gradients and parameter order also matched; MACs remained 19,690,951,728 for a 4-s mixture and 3-s enrollment. The eight public tests, the 6,000-query frozen-result verifier and the included-example inference command passed. These checks do not constitute new multi-day training or a new full audio benchmark.
