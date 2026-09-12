---
title: "E013 - The map of the regulator"
date: 2026-09-12
weight: 13
hypotheses: [H3]
statuses: [reset]
params:
  fixed: "2026-09-12 - ADDENDUM-14"
  run: "2026-09-13"
  results: "2026-09-13"
  corrected: "2026-09-13 - the shuffle check"
  verdict: "reset 2026-09-13 - nothing is claimed from this run until the corpus, the metric and the mask are ones the bench trusts (docs/corpus.md)"
---

# E013 - The map of the regulator

Set to what does the model make the fewest mistakes, and what does that cost in memory. The engineer's question, measured as an error rate over a map of floor, size and strength, against uniform quantization at the same bits.

- Preregistration: [ADDENDUM-14](ADDENDUM-14.md). Results: [results.md](results.md). Runs: `runs/E013-regulator-map/`.
- Corrected by the shuffle check: three quarters of the win over uniform quantization was the order of the answer options ([results](results.md#what-the-shuffle-did-to-this-page), `runs/reference/shuffle-answers/`).
- Random zones are not a reference here: they overlap less, so they spend more memory and answer worse ([E011](../E011-depth-caps/results.md)).
