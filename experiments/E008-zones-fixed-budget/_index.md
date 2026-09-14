---
title: "E008 - Expert zones at a fixed 5-bit budget"
date: 2026-09-11
weight: 8
hypotheses: [H3.1, H3.2]
statuses: [reset]
params:
  fixed: "2026-09-11 - ADDENDUM-07 fddb919, terms ADDENDUM-08 84cb427"
  run: "2026-09-11 - a4ead2f"
  results: "aafe21c"
  verdict: "reset 2026-09-13 - nothing is claimed from this run until the corpus, the metric and the mask are ones the bench trusts (docs/corpus.md)"
---

# E008 - Expert zones, fixed budget

Expert zones from the query's mask on the weight map, fitted to a preset mean of 5 bits.

- Preregistration: [ADDENDUM-07](ADDENDUM-07.md), terms [ADDENDUM-08](ADDENDUM-08.md). Results deleted with the corpus (`1e89a81`). Runs: `runs/E008-zones-fixed-budget/`.
- Legacy: the fixed budget is not the project's picture; the lens layout (E010, [docs/quantization-filter.md](../../docs/quantization-filter.md)) replaces it.
