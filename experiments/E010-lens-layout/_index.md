---
title: "E010 - Lens layout"
date: 2026-09-12
weight: 10
hypotheses: [H3, H3.1, H3.2]
statuses: [reset]
params:
  fixed: "2026-09-12 - PREREG"
  run: "2026-09-12 - 73b4216"
  results: "781bf7a, random zones 84d1b9f"
  verdict: "reset 2026-09-13 - nothing is claimed from this run until the corpus, the metric and the mask are ones the bench trusts (docs/corpus.md)"
---

# E010 - Lens layout

The whole network at a floor, the query's expert zones read above it. The mechanism is [docs/quantization-filter.md](../../docs/quantization-filter.md) - then docs/lens.md, at commit `4fa3e7f`.

- Preregistration: [PREREG](PREREG.md). Results deleted with the corpus (`1e89a81`). Runs: deleted with the corpus.
- Random zones were run separately on the same commit (`runs/E010-lens-layout-random/`). They are no longer a reference: they overlap less, so they cost more ([E014 ADDENDUM](../E014-moved-zones/ADDENDUM.md)).
