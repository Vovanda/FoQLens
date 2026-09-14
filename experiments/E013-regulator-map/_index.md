---
title: "E013 - The map of the regulator"
date: 2026-09-12
weight: 13
hypotheses: [H3]
statuses: [reset]
params:
  fixed: "2026-09-12 - PREREG"
  run: "2026-09-13"
  results: "2026-09-13"
  corrected: "2026-09-13 - the shuffle check"
  verdict: "reset 2026-09-13 - nothing is claimed from this run until the corpus, the metric and the mask are ones the bench trusts (docs/corpus.md)"
---

# E013 - The map of the regulator

The regulator's settings - floor, size and strength of the zones - mapped against uniform quantization at the same bits, scored by the top letter of four.

- Preregistration: [PREREG](PREREG.md). Results and runs deleted with the corpus (`1e89a81`).
- Corrected by the shuffle check before the reset: three quarters of the win over uniform quantization was the order of the answer options (`runs/reference/shuffle-answers/`).
- Random zones are not a reference here: they overlap less, so they spend more memory ([E011](../E011-depth-caps/_index.md)).
