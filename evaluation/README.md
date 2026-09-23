# WSJ0-2mix evaluation artifacts

This directory records the full test of **APSR P / R4 / seed 43 / validation-selected E102**. It contains metadata and scores; recordings are not distributed.

## Available files

| File | Contents |
|---|---|
| `wsj0-2mix/summary.json` | Full-precision metric means, counts, selection, runtime, and source/export digests |
| `wsj0-2mix/metrics.csv` | All 6,000 original evaluator rows, preserved byte for byte |
| `wsj0-2mix/trials.csv` | Query-to-mixture/target/enrollment mapping with portable relative paths |
| `verify_results.py` | Standard-library integrity and aggregation verifier |

Run `python evaluation/verify_results.py` from the repository root. It verifies CSV digests, unique trial keys, two targets for each mixture, finite metrics, per-query improvement subtraction, confusion decisions, and summary means. It validates the published artifacts; it does not recreate model predictions from audio.

## Trial identities

The fixed benchmark contains 3,000 mixtures and both target queries (6,000 queries), using the validation-selected E102 checkpoint. Follow the [full-utterance paired-query protocol](../docs/EVALUATION.md), including enrollment padding, normalization and scoring conventions.

`trials.csv` preserves the original `key`, mixture index `idx`, and target query index `case_idx`. `target_source` identifies `s1` or `s2`; `case_idx` is the evaluator's within-pair order and should not substitute for the explicit source mapping.

Resolve `mixture_relpath` and `target_relpath` against your own `WSJ0_2MIX_ROOT` (containing `tt/mix`, `tt/s1`, `tt/s2`). Resolve `enrollment_relpath` against your own `ENROLLMENT_ROOT` (containing `wsj0/si_dt_05_8k` and `wsj0/si_et_05_8k`). These two root names describe local data locations, not download links. The manifest alone does not grant access to the recordings.

## Metrics

| CSV field | Meaning |
|---|---|
| `si_sdr` | Mean-centered, scale-invariant SDR against the requested target, in dB |
| `sdr` | BSS Eval SDR from `mir_eval.separation.bss_eval_sources`, one target reference, `compute_permutation=False` |
| `pesq` | PESQ narrow-band mode at 8 kHz |
| `estoi` | Extended STOI, stored as a fraction; multiply by 100 for percent |
| `*_mix` | The same metric computed for the unprocessed mixture against this query's target |
| `si_sdr_i`, `sdr_i` | Output score minus that query's own mixture score, in dB |
| `interferer_si_sdr` | SI-SDR of the target estimate against the competing clean source |
| `target_margin_db` | Target SI-SDR minus competing-source SI-SDR |
| `target_confusion_rate` | Per-query indicator: 1 when target margin is ≤ 0, otherwise 0 |
| `proj_sdr`, `proj_sdr_mix`, `proj_sdr_i` | Legacy scalar-projection SDR audit fields; not the paper's BSS Eval SDR |

Metric arithmetic and runtime options are defined in [Evaluation](../docs/EVALUATION.md). The CSV retains full precision and the additional audit fields listed above.

All headline means weight queries equally. TCR is 117/6,000 = 1.95%; both target queries have positive margins for 2,886/3,000 mixtures = 96.2%. These are different statistics. If comparing future systems, pair by `key` and keep both queries of a mixture together when resampling. This single-seed release does not estimate variability across training seeds.

## Running extraction and evaluation

The main model, `evaluate.py`, metric dependencies and manifest preparation are included. Follow [data preparation](../docs/DATA.md) and [evaluation instructions](../docs/EVALUATION.md) to evaluate licensed audio. New evaluations use separate output directories and do not overwrite these frozen records. Quality scores, model-forward throughput, and full streaming latency are separate measurements; no streaming RTF is inferred from this table.
