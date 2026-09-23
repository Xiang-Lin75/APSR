# Pretrained checkpoint

From the installed repository, download the validation-selected WSJ0-2mix checkpoint with checksum verification:

```bash
python scripts/download_checkpoint.py
```

This saves `checkpoints/apsr_p_wsj0_2mix_r4_seed43_e102.pt`. Use [inference.py or the Python API](../docs/INFERENCE.md) to load it. Checkpoint selection and measured quality are in the [model card](../docs/MODEL_CARD.md).

## Release assets

The [E102 release](https://github.com/Xiang-Lin75/APSR-P/releases/tag/wsj0-2mix-r4-seed43-e102) also includes:

| File | Purpose |
|---|---|
| `apsr_p_wsj0_2mix_r4_seed43_e102.pt` | Inference state dictionary; model parameters and buffers |
| `inference_config.json` | Historical architecture configuration |
| `MODEL_CARD.md` | Model and evaluation information at release time |
| `test_summary.json` | Full test summary and provenance |
| `tensor_inventory.json` | Tensor names, shapes, dtypes and digests |
| `export_verification.json` | Export/source digests and equivalence checks |
| `SHA256SUMS` | Checksums of the other six release assets |

To download all assets:

```bash
gh release download wsj0-2mix-r4-seed43-e102 --repo Xiang-Lin75/APSR-P --dir checkpoints/e102
```

Verify `SHA256SUMS` using `sha256sum -c SHA256SUMS` on Linux, or compare `Get-FileHash -Algorithm SHA256` output on Windows. The current repository's [inference_config.json](inference_config.json) describes the standalone public API; disabled-feature arguments in the historical release configuration are not public constructor arguments. Existing release files and checksums remain unchanged.

## Training initialization

`paper_initialization_seed43.pt` contains untrained source-model tensors and CPU RNG state, separate from the pretrained checkpoint. Its origin, intended use and reproducibility limits are documented in [Training](../docs/TRAINING.md#initialization).
