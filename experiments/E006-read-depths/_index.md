---
title: "E006 - Read depths from one sliced copy"
date: 2026-09-11
weight: 6
hypotheses: []
statuses: [done]
params:
  fixed: "- (engineering, docs/plan.md step 5; no prediction)"
  run: "2026-09-11 - c35709b"
  results: "ddf5dbc"
  verdict: "D8 as good as int8; without bf16 the bench frees 1.63 GiB on E2B"
---

# E006 - Read depths

One residual-sliced copy of the weights (after MoBiQuant) read at 2 / 4 / 6 / 8 bits; perplexity per depth and the memory of the resident bench.

- Results: [results.md](results.md). Runs: `runs/E006-read-depths/`.
- Engineering: it tests no hypothesis.
