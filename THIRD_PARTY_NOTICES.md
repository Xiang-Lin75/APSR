# Software provenance and attribution

The main-model source was integrated from the APSR-P research implementation and checked against the evaluated P checkpoint. File organization was informed by [TIGER](https://github.com/JusperLee/TIGER); this integration did not import TIGER's trainer or inference scripts. The source-license and upstream attribution review remains pending author confirmation; this document does not relicense any dependency.

External dependencies remain separately licensed:

- [PyTorch](https://github.com/pytorch/pytorch) and [torchaudio](https://github.com/pytorch/audio): neural operators and differentiable resampling.
- [NumPy](https://github.com/numpy/numpy), [SoundFile](https://github.com/bastibe/python-soundfile), [PyYAML](https://github.com/yaml/pyyaml), [tqdm](https://github.com/tqdm/tqdm): numerical operations, audio I/O, configuration and progress reporting.
- [torch-stoi](https://github.com/mpariente/pytorch_stoi): differentiable negative eSTOI; the local wrapper supplies explicit resampling and preserves the upstream loss.
- [mir_eval](https://github.com/craffel/mir_eval), [python-pesq](https://github.com/ludlows/PESQ), [pystoi](https://github.com/mpariente/pystoi): independent evaluation metrics.

Dataset use is described in `DATA_NOTICE.md`. No corpus license is changed by the code publication.
