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
  verdict: "engineering, stands through the reset: it depends on no question set - one sliced copy read at 2/4/6/8 bits, D8 bit-exact from the resident copy, 1.63 GiB freed on E2B; the perplexities by depth are on 40 texts of the old set"
---

# E006 - Read depths

One residual-sliced copy of the weights (after MoBiQuant) read at 2 / 4 / 6 / 8 bits; perplexity per depth and the memory of the resident bench.

- Results: [results.md](results.md). Runs: deleted in `1e89a81`, in git history.
- Engineering: it tests no hypothesis.
