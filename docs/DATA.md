# Data preparation

Obtain WSJ0 and/or WHAM! under their applicable agreements and prepare the 8-kHz minimum-length version. Corpus WAVs are not part of this source release. The existing short listening excerpts have separate publication permission; they are not the benchmark input set.

Expected mixture layout:

```text
wav8k/min/
  tr/{mix,s1,s2}/            # WSJ0-2mix
  cv/{mix,s1,s2}/
  tt/{mix,s1,s2}/
```

WHAM! uses `mix_both` instead of `mix`, with its matching clean `s1`/`s2`. Use the official synthesis output; the preparation script below only resolves file paths. It does not generate mixtures or remove the noise.

The enrollment root contains `wsj0/si_tr_s_8k_all/<speaker>/<utterance>.wav` and the corresponding directories referenced by the trial lists. Audio must match the original utterance identities and 8-kHz preparation. Training samples enrollment files from the selected speaker's directory, so provide the complete enrollment pool rather than only the WAVs named in test queries.

## Fixed trial identities

`protocols/wsj0_2mix/train.csv` contains 40,000 target queries (20,000 mixtures). `validation.csv` contains 10,000 queries (5,000 mixtures). `evaluation/wsj0-2mix/trials.csv` contains 6,000 queries (3,000 mixtures). Each mixture retains both target queries. No new enrollment selection is performed during path preparation.

```bash
python scripts/prepare_manifest.py --trials protocols/wsj0_2mix/train.csv --mixture-root /path/to/wav8k/min --enrollment-root /path/to/enrollment8k --output data/wsj0_2mix/tr
python scripts/prepare_manifest.py --trials protocols/wsj0_2mix/validation.csv --mixture-root /path/to/wav8k/min --enrollment-root /path/to/enrollment8k --output data/wsj0_2mix/cv
python scripts/prepare_manifest.py --trials evaluation/wsj0-2mix/trials.csv --mixture-root /path/to/wav8k/min --enrollment-root /path/to/enrollment8k --output data/wsj0_2mix/tt
```

For WHAM!, repeat with `--wham`, its `wav8k/min` root, and output directories `data/wham/tr`, `data/wham/cv`, `data/wham/tt`. All mixture/target filenames must correspond to the supplied trial list. Missing files, invalid sample rates, duplicate queries and invalid target pairing fail explicitly.

The generated `mix.scp`, `ref.scp`, `aux.scp` contain local paths and should remain local. `manifest_info.json` records counts and hashes. Published trial CSVs contain relative paths only. Validation and test use fixed complete enrollment waveforms; training uses 4-s mixture crops, 3-s enrollment crops, and the disclosed random enrollment policy.

Full synthesis/resampling provenance must be retained for any independently regenerated corpus. Matching filenames alone does not establish waveform equivalence to the historical benchmark.
