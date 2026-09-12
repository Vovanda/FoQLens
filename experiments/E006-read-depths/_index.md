---
title: "E006 - Read depths from one sliced copy"
date: 2026-09-11
weight: 6
hypotheses: []
statuses: [reset]
params:
  fixed: "- (engineering, docs/plan.md step 5; no prediction)"
  run: "2026-09-11 - c35709b"
  results: "ddf5dbc"
  verdict: "reset 2026-09-13 - nothing is claimed from this run until the corpus, the metric and the mask are ones the bench trusts (docs/corpus.md)"
---

# E006 - Read depths

One residual-sliced copy of the weights (after MoBiQuant) read at 2 / 4 / 6 / 8 bits; perplexity per depth and the memory of the resident bench.

- Results: [results.md](results.md). Runs: `runs/E006-read-depths/`.
- Engineering: it tests no hypothesis.
