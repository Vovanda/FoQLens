---
title: "E011 - Depth caps: when storage follows the queries"
date: 2026-09-12
weight: 11
hypotheses: []
statuses: [reset]
params:
  fixed: "2026-09-12 - ADDENDUM-12"
  run: "2026-09-12 - c1bd457"
  results: "c1bd457"
  verdict: "reset 2026-09-13 - nothing is claimed from this run until the corpus, the metric and the mask are ones the bench trusts (docs/corpus.md)"
---

# E011 - Depth caps

A layout is read per query, storage is one copy: a block keeps the deepest slice any query asks of it. How much that saves, against how many queries and how varied they are.

- Preregistration: [ADDENDUM-12](ADDENDUM-12.md). Results: [results.md](results.md). Runs: `runs/E011-depth-caps/`.
- An engineering measurement, no hypothesis at stake: reading is unchanged by a cap, so quality is not measured again.
