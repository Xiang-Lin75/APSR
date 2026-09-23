# Software attribution and license scope

APSR-P's original code, documentation and project-authored weights use the [MIT License](../LICENSE). Adapted components retain their original terms; the source package therefore includes MIT and Apache-2.0 material. [NOTICE](../NOTICE) and the full texts in [licenses/](../licenses) accompany the package. Dataset recordings are covered separately by the [data notice](DATA_NOTICE.md).

## Components and implementation references

| Source | APSR-P component | License and retained notice |
|---|---|---|
| [GTCRN](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/gtcrn.py) | `models/frontend.py`: ERB and SFE; `models/layers.py`: ConvBlock and grouped GRUs; `models/blocks.py`: grouped dual-path processing | [MIT](../licenses/GTCRN-MIT.txt), Rong Xiaobin |
| [TIGER](https://github.com/JusperLee/TIGER/blob/9f18d4a10a7137e1ce8052cfb62215179f1287b6/look2hear/models/tiger.py) | Gated local/global fusion reference for `models/layers.py: LA_2D`, adapted to two-dimensional causal features; repository organization | [MIT](../licenses/TIGER-MIT.txt), Kai Li |
| [Asteroid](https://github.com/asteroid-team/asteroid/blob/fce87469132760fbab41c20616ea0f0e079aad38/asteroid/losses/sdr.py) | `losses/operators.py: compute_si_snr`, corresponding to the mean-centered SI-SDR calculation in `SingleSrcNegSDR` | [MIT](../licenses/Asteroid-MIT.txt), Pariente Manuel |
| [ESPnet / TF-GridNet](https://github.com/espnet/espnet/blob/1a753b4905932f50e7ea38012e1a289f2c3ed5d8/espnet2/enh/separator/tfgridnet_separator.py) | `models/blocks.py`: per-head attention projections and head recombination, with APSR-P's normalization and causal/prefix-boundary handling | [Apache-2.0](../licenses/ESPnet-Apache-2.0.txt), Johns Hopkins University (Shinji Watanabe) |

Component paths are relative to `apsr/`. The GTCRN components are configured for APSR-P's dimensions and conditioning. The attention changes and source references are also marked in `blocks.py`. TIGER's trainer and inference scripts are not bundled.

## Installed dependencies

These packages are installed separately and retain their own licenses:

- [PyTorch](https://github.com/pytorch/pytorch) and [torchaudio](https://github.com/pytorch/audio): neural operators and resampling.
- [NumPy](https://github.com/numpy/numpy), [SoundFile](https://github.com/bastibe/python-soundfile), [PyYAML](https://github.com/yaml/pyyaml) and [tqdm](https://github.com/tqdm/tqdm): numerical operations, audio I/O, configuration and progress reporting.
- [torch-stoi](https://github.com/mpariente/pytorch_stoi): differentiable negative eSTOI. APSR-P's wrapper supplies resampling and calls the upstream loss.
- [mir_eval](https://github.com/craffel/mir_eval), [python-pesq](https://github.com/ludlows/PESQ) and [pystoi](https://github.com/mpariente/pystoi): evaluation metrics, available through the optional evaluation dependencies.

The MIT grant for APSR-P does not grant rights to third-party datasets or replace any dependency's terms.
