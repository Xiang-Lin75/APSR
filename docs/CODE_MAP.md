# Reading APSR

The package follows the paper's computation, with one canonical main model.

| Method element | Implementation | Tensor contract |
|---|---|---|
| Full extraction | `models/apsr.py: APSR.forward` | mixture `(B,L)`, enrollment `(B,E)` → target `(B,1,L)` |
| STFT and ERB | `models/apsr.py: _extract_features`, `models/frontend.py` | spectra `(B,2,T,129)`; unfolded features `(B,9,T,65)` |
| Shared encoder | `models/encoder.py: SpeechEncoder` | `(B,9,T,65)` → `(B,64,T,65)` |
| Gated token pooling | `models/enrollment.py: GatedAttentiveEnrollmentPooling` | enrollment `(B,64,T_e,65)` → `(B,64,8,65)` |
| Refinement | `models/separator.py: ProgressiveRefinement` | prefix and mixture evolve through block calls |
| Frequency/temporal modeling | `models/blocks.py: SeparatorBlock` | frequency bidirectional grouped GRUs; time unidirectional grouped GRUs |
| Anchored memory | `models/memory.py: TargetAnchoredTimeConstantMemory` | channel-last input `(B,T,F,C)`; anchor/state `(B,F,16)` |
| Causal self-attention | `models/blocks.py: MemoryOnlyCausalAttention` | `(B,C,K+1+T,F)`; no prefix keys for mixture queries |
| Gated skip and decoder | `models/decoder.py: SpeechDecoder` | separator output plus mixture encoder skip |
| Complex masks | `models/decoder.py: ComplexMaskHead` | target and sink masks, followed by mask-sum constraint |
| Training objective | `losses/objective.py`, `losses/operators.py` | final waveform and differentiable pre-iSTFT target spectrum |

Paths in this table are relative to `apsr/`. `B` is the query batch, `T` the mixture frame count, `T_e` enrollment frames, `F` frequency bins, `C` channels and `K` enrollment tokens.

## Two different recurrence axes

Physical block calls are `0 → 1 → 2 → 2 → 2 → 2 → 3` at R4. The four calls to block 2 share its parameters. Each call starts a fresh acoustic-time memory trajectory. Enrollment tokens progress across calls independently of the extracted mixture features.

Within a call, memory builds a frequency-indexed enrollment anchor, computes the current evidence-dependent write rate and bounded candidate, accumulates memory causally, and adds a gated readout to mixture features. The prefix itself is not overwritten by memory readout.

Temporal GRU states and cumulative attention-normalization statistics are reset at the prefix/mixture boundary. Mixture attention queries cannot access enrollment or separator keys. These boundaries make memory the conditioning route while retaining a separate enrollment stream.

## Parameter naming and loading

The public model retains the published E102 state-dictionary names and all 398 tensors. The five historical schema-buffer names are compatibility metadata; they do not enable experimental variants. Loading is strict. No private package, historical model superclass, dynamic class replacement, or variant registry is needed.

The constructor assembles only active P modules. The published seed-43 initialization fixture avoids changing the source implementation's initial tensors merely because unused historical construction steps were removed. See [TRAINING.md](TRAINING.md).
