# Evaluating P

Select the checkpoint using validation before running `evaluate.py`. The evaluator reads one mixture and its two target/enrollment queries together, matching the original paired collation. Full enrollment waveforms are zero-padded to the longer enrollment within the pair before the model's standard-deviation normalization. Running each target independently or cropping enrollment to a demo excerpt changes this protocol.

```bash
python evaluate.py --checkpoint checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt --manifest data/wsj0_2mix/tt --output outputs/p-test --device cuda
```

Outputs are `metrics.csv` (one row per query) and `summary.json` (arithmetic means, query counts, checkpoint/manifest hashes and runtime identity). `--limit` labels a subset explicitly. Reusing an existing result directory is rejected.

Checkpoint epoch and smoke-test scope are propagated when present in a public training checkpoint. Tensor-only weights do not encode an epoch or training-completion status; those fields remain `UNKNOWN` and must be tied to the corresponding model card and checksum. An all-query evaluation does not itself make a checkpoint final.

`--primary-only` computes SI-SDR, measured mixture SI-SDR, SI-SDRi, competing-target SI-SDR, target margin and TCR. It records SDR/PESQ/eSTOI as `PENDING`. Full mode requires all metric dependencies and rejects nonfinite results rather than silently reporting a partial mean.

SI-SDR uses float64 mean removal, projection and energy ratios with epsilon 1e-8. SI-SDRi subtracts the mixture score for each query before averaging. TCR is the fraction of estimates whose target SI-SDR minus competing-source SI-SDR is <= 0. PESQ is narrowband at 8 kHz; eSTOI uses `extended=True`. SDR uses BSS Eval without permutation, not scalar-projection SDR. Both TCR and eSTOI are stored as fractions; multiply by 100 for percent.

WHAM! uses noisy `mix_both` against clean targets. Its mixture baseline must be measured and generally is not zero; absolute SI-SDR and SI-SDRi are both written.

The immutable E102 evidence remains in `evaluation/wsj0-2mix/`. New evaluations write a separate output directory. Source-integration parity tests and subset replay are not substitutes for the original 6,000-query benchmark.
