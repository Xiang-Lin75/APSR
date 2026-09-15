# Pretrained checkpoint

Download the validation-selected P checkpoint from the [E102 GitHub Release](https://github.com/Xiang-Lin75/APSR-P/releases/tag/wsj0-2mix-r4-seed43-e102). Weights are release assets rather than files committed to Git history.

| Release asset | Purpose |
|---|---|
| `apsr_p_wsj0_2mix_r4_seed43_e102.pt` | Plain PyTorch state dictionary: model parameters and buffers |
| `inference_config.json` | Matching architecture constructor settings and sample rate |
| `MODEL_CARD.md` | Model, checkpoint selection, evaluation scope, and availability |
| `test_summary.json` | Frozen 6,000-query test summary with provenance |
| `tensor_inventory.json` | Tensor names, shapes, dtypes, and per-tensor digests |
| `export_verification.json` | Source/export digests, strict load, and forward-equivalence evidence |
| `SHA256SUMS` | SHA256 of all six other release assets |

Download all assets with GitHub CLI, if installed:

```bash
gh release download wsj0-2mix-r4-seed43-e102 --repo Xiang-Lin75/APSR-P --dir checkpoints/e102
```

On Linux, run `sha256sum -c SHA256SUMS` inside that directory. On Windows, use `Get-FileHash -Algorithm SHA256` and compare the corresponding digest in `SHA256SUMS`.

The `.pt` can be read as tensors using PyTorch:

```python
import torch
state = torch.load("apsr_p_wsj0_2mix_r4_seed43_e102.pt", map_location="cpu", weights_only=True)
```

The standalone implementation is now included:

```python
from apsr import APSRP
model = APSRP.from_pretrained("apsr_p_wsj0_2mix_r4_seed43_e102.pt", device="cpu")
```

The current `inference_config.json` documents this public API. The immutable GitHub Release's original configuration describes the historical source constructor; its disabled-feature fields are not constructor arguments to the public class. Existing checkpoint bytes and release checksums are preserved.

The export retains every evaluated model tensor exactly, including non-trainable parameters and buffers. It excludes optimizer, scheduler, full training configuration, and recovery metadata. Strict loading and identical full-length paired outputs were verified locally; no retraining or quantization was applied.

## Untrained initialization fixture

`paper_initialization_seed43.pt` is separate from the trained E102 weights. It contains the frozen source implementation's **untrained** seed-43 tensors and CPU RNG state generated during integration, enabling the reorganized trainer to start with the same tensors as that source implementation in the verified environment. It is not a saved epoch-zero checkpoint from the original cloud run. See [training details](../docs/TRAINING.md).
