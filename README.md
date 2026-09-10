# APSR-P · Audio Demo

[Open the demo website](https://xiang-lin75.github.io/APSR-P/)

Compare **Enrollment · Mixture · APSR-P · Clean target** and switch between
target A and B for the same mixture. The page is dedicated to listening, without
benchmark tables, architecture diagrams or author biographies.

The eight paired-target examples are prepared locally. **Public WSJ0 audio is
pending confirmation of the applicable publication permissions**, so the live
page currently shows an availability notice. No audio is included in this public
release. Training/inference code and model weights are planned after acceptance.

The layout follows the side-by-side spectrogram/player comparison in the
[Universal Speech Enhancement Hybrid demo](https://nanless.github.io/universal-speech-enhancement-demo/#hybrid).
No audio, figures, results or source code from that website are copied.

To preview: `python -m http.server 8766 --bind 127.0.0.1 --directory site`, then
open http://127.0.0.1:8766/. GitHub Actions deploys `site/` to GitHub Pages.
See [data notice](DATA_NOTICE.md). `SHA256SUMS` binds every release file except itself.
