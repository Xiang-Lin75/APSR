# WSJ0-2mix evaluation artifacts

This directory records the full test of **APSR-P P / R4 / seed 43 / validation-selected E102**. It contains metadata and scores; recordings are not distributed.

## Available files

| File | Contents |
|---|---|
| `wsj0-2mix/summary.json` | Full-precision metric means, counts, selection, runtime, and source/export digests |
| `wsj0-2mix/metrics.csv` | All 6,000 original evaluator rows, preserved byte for byte |
| `wsj0-2mix/trials.csv` | Query-to-mixture/target/enrollment mapping with portable relative paths |
| `verify_results.py` | Standard-library integrity and aggregation verifier |

Run `python evaluation/verify_results.py` from the repository root. It verifies CSV digests, unique trial keys, two targets for each mixture, finite metrics, per-query improvement subtraction, confusion decisions, and summary means. It validates the published artifacts; it does not recreate model predictions from audio.

## Data and inference protocol

- **Dataset:** 8-kHz minimum-length WSJ0-2mix, 3,000 test mixtures, both targets queried separately (6,000 queries). Enrollment is a different utterance of the requested speaker, fixed by the trial list.
- **Checkpoint:** E102 selected by validation SI-SDRi, not test performance. One training seed, 43. Four shared refinement calls.
- **Inputs:** full mixture and enrollment utterances, no random evaluation crops or external waveform normalization. The model's configured enrollment normalization remains enabled. Test mixtures are fixed, without dynamic source rebalancing.
- **Batching:** one mixture and its two target queries per loader batch; the model receives two waveform/enrollment pairs. Enrollment utterances are right-padded to the longer enrollment in that pair. Retain this collation when reproducing the frozen evaluator.
- **Scoring:** full-length target/mixture/estimate arrays are aligned to their common length. All 6,000 rows have finite metrics; no failed rows are excluded from these published means.

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

SI-SDR is computed in float64 after subtracting the estimate/reference means, using epsilon `1e-8` in projection energy and signal/error energy ratios. Improvements are computed per query before taking the arithmetic mean. The mixture baseline is measured, not assumed to be zero. BSS Eval receives the requested target and its estimate without output permutation. eSTOI uses `extended=True`.

All headline means weight queries equally. TCR is 117/6,000 = 1.95%; both target queries have positive margins for 2,886/3,000 mixtures = 96.2%. These are different statistics. If comparing future systems, pair by `key` and keep both queries of a mixture together when resampling. This single-seed release does not estimate variability across training seeds.

## Later code release

Architecture, inference/evaluation entrypoints, pinned dependencies, and a data-preparation recipe are planned after acceptance. Those components are needed to reproduce predictions from licensed audio. The current release supports verification of the reported scores and trial mapping. Quality scores, model-forward throughput, and full streaming latency are separate measurements; no streaming RTF is inferred from this table.
