# APSR-P

**Enrollment-Anchored Memory and Progressive Refinement for Causal Target Speaker Extraction**

[Audio demo](https://xiang-lin75.github.io/APSR-P/) · [Pretrained checkpoint](https://github.com/Xiang-Lin75/APSR-P/releases/tag/wsj0-2mix-r4-seed43-e102) · [Model card](MODEL_CARD.md) · [Evaluation](evaluation/README.md)

APSR-P extracts a target speaker from a mixture using a separate enrollment utterance. It combines frequency-resolved enrollment references, an enrollment-anchored causal memory interface, and shared-block progressive refinement in the time–frequency domain.

## Architecture

![APSR-P overview and separator-block details](docs/figures/apsr_overview.png)

The shared encoder processes mixture and enrollment independently. Gated pooling preserves frequency information in enrollment tokens. Separator blocks model frequency and time, read enrollment-conditioned memory, and apply causal self-attention. The decoder fuses the resulting features with the mixture encoder's skip features and estimates a complex target mask.

- **Enrollment-anchored memory:** a frequency-indexed reference guides candidate content, adaptation rate, and gated readout. A bounded candidate and convex update keep the latent memory close to its reference within each block call.
- **Progressive enrollment state:** enrollment tokens evolve through shared refinement calls, while each call maintains its own acoustic-time memory trajectory. Memory provides the enrollment-to-mixture conditioning route.
- **Shared refinement:** four calls to a shared block refine mixture features with bounded corrections and no additional parameter set per call.

<details>
<summary><strong>Memory detail</strong></summary>

<p align="center"><img src="docs/figures/apsr_memory.png" width="420" alt="Enrollment-anchored causal memory: anchor, candidate, gated recurrence, and residual readout"></p>

The reference is fixed over mixture time within a call. Candidate saturation bounds latent displacement; it does not by itself guarantee correct speaker selection. Target margin and target-confusion rate assess that behavior empirically.

</details>

Vector figures: [overview PDF](docs/figures/apsr_overview.pdf) · [memory PDF](docs/figures/apsr_memory.pdf).

Neural operations use current and past mixture frames. The released configuration uses a centered 32-ms STFT window with 16-ms analysis lookahead. End-to-end streaming latency is not established by the offline test below.

## WSJ0-2mix test

Validation-selected **epoch 102**, **R4**, **seed 43**; 8 kHz, minimum-length mixtures. The full test evaluates both targets of all **3,000 mixtures**, yielding **6,000 queries**.

| SI-SDRi ↑ (dB) | SDR ↑ (dB) | SDRi ↑ (dB) | PESQ ↑ | eSTOI ↑ (%) | TCR ↓ (%) |
|---:|---:|---:|---:|---:|---:|
| 13.81 | 14.32 | 14.17 | 3.10 | 87.82 | 1.95 |

306,842 trainable parameters; 312,986 registered parameters. TCR is 117/6,000. Each improvement subtracts the corresponding mixture score for that query. These are the frozen full-test results for one training seed; ablation and WHAM! results will be added after verification.

[Exact summary](evaluation/wsj0-2mix/summary.json) · [Per-query metrics](evaluation/wsj0-2mix/metrics.csv) · [Portable trial list](evaluation/wsj0-2mix/trials.csv)

## Checkpoint and reproducibility

Download **`apsr_p_wsj0_2mix_r4_seed43_e102.pt`** from the [E102 release](https://github.com/Xiang-Lin75/APSR-P/releases/tag/wsj0-2mix-r4-seed43-e102). It contains inference parameters and buffers only. The release also provides the matching inference configuration, model card, test summary, tensor inventory, export verification, and SHA256 checksums.

The exported state was checked tensor by tensor against the evaluated E102 checkpoint. Strict loading and a full-length paired-query forward comparison passed with identical outputs in the same runtime. The conversion does not replace or rerun the 6,000-query benchmark.

**Training, model implementation, and inference/evaluation code are planned after acceptance.** The weights are available ahead of that code release; they require the matching architecture to perform extraction. The artifact verifier below is available now and needs only Python's standard library:

```bash
python evaluation/verify_results.py
```

It verifies checksums, query pairing, mixture-baseline subtraction, identity decisions, and all published metric means. See [checkpoint details](checkpoints/README.md) and [evaluation protocol](evaluation/README.md).

## Listening demo

The [demo website](https://xiang-lin75.github.io/APSR-P/) focuses on **Enrollment · Mixture · APSR-P · Clean target**, with A/B target switching for the same mixture. Public WSJ0 audio awaits confirmation of the applicable publication permissions; the live site currently shows an availability notice. See [data notice](DATA_NOTICE.md).

The listening layout was informed by the [Universal Speech Enhancement Hybrid demo](https://nanless.github.io/universal-speech-enhancement-demo/#hybrid). GitHub Pages deploys only `site/`; the method figures, checkpoint links, and test artifacts remain in this repository.
