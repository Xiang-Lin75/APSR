# Training the main model

The public trainer supports P on one CPU or CUDA device. The recipes are in `configs/wsj0_2mix.yaml` and `configs/wham.yaml`. All paths are interpreted relative to the working directory; commands in the README run from the repository root.

## Paper recipe

| Setting | Value |
|---|---|
| Sampling rate | 8 kHz |
| Train crop / enrollment crop | 4 s / 3 s |
| Train loader batch | 2 mixtures = 4 target queries |
| Validation loader batch | 1 mixture = 2 target queries |
| Objective | negative SI-SDR + 0.1 normalized magnitude/RI error + 0.1 negative eSTOI |
| Spectral grid | model's 256-point STFT, hop 64, centered Hann window |
| Spectral input | differentiable pre-iSTFT target estimate; no waveform reanalysis fallback |
| Magnitude / RI weights | 0.5 / 0.5, normalized by target spectral energy |
| Differentiable eSTOI | every step from epoch 1; VAD enabled; `torch-stoi==0.2.3` with explicit 10-kHz resampling |
| Optimizer | Adam, learning rate 0.001, weight decay 0 |
| Gradient norm limit | 5 |
| Scheduler | ReduceLROnPlateau, mode=max, factor=0.5, patience=3; PyTorch default threshold semantics |
| Selection | highest validation SI-SDRi, strict improvement |
| Stop | 6 consecutive validation non-improvements, or 150 epochs |
| WSJ0-2mix augmentation | rebalance the existing source pair to SIR in [-3,3] dB |
| WHAM! augmentation | source rebalancing disabled, retaining `mix_both` noise |

The dynamic-SIR setting does not select new speaker pairs. The loader preserves source alignment, paired target crops and enrollment sampling. Evaluation metrics use an independent float64 implementation; validation retains the historical float32 SI-SDR objective arithmetic.

## Initialization

`checkpoints/paper_initialization_seed43.pt` contains **untrained** tensors and the post-construction CPU RNG state from the frozen source implementation initialized with seed 43 during integration. It was generated under PyTorch 2.7.0; it is not a saved epoch-zero artifact from the original cloud run. Its source equivalence and checksum are recorded in the integration report.

The historical implementation constructed and replaced unused modules, consuming random draws. Simply rebuilding the active modules in a cleaner order changes their initial values even with the same seed. The supplied fixture preserves the source implementation's initial values in the verified runtime without distributing those historical modules. The canonical public training recipe therefore uses this seed-43 fixture. `APSR()` itself remains a normal randomly initialized PyTorch module for programmatic use.

A fresh multi-day training run has not been repeated as part of code integration. Identical final results across library versions, devices or worker scheduling are not promised.

## Saving and resuming

`latest.pt` saves the last complete epoch; `best.pt` saves the validation-selected checkpoint. Both include model, optimizer, scheduler, selection state, stopping counter, configuration and RNG states. Resume accepts the public trainer format and checks the recipe. Extending the total epoch limit is permitted; changing the numerical recipe is rejected.

The trainer saves only at epoch boundaries. Unfinished batches must be repeated after a crash. Persistent data-loader worker processes are recreated on restart, so their private sampling state is not preserved; epoch-boundary resume is not claimed to be a bitwise replay of an uninterrupted run. GPU/CPU RNG state and loader generators are retained.

## Smoke checks

Use `--smoke-steps 1` with a small local config and manifest to exercise training and validation. Such checkpoints and logs are labeled `SMOKE_ONLY`; they cannot silently resume into an unrestricted training run. Smoke checks do not supply paper results.

Only the main APSR implementation is included. Experimental variant constructors and ablation launchers are not part of this package.
