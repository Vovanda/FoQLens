---
title: "E010 - Lens layout"
date: 2026-09-12
weight: 10
hypotheses: [H3, H3.1, H3.2, H4]
statuses: [done]
params:
  fixed: "2026-09-12 - ADDENDUM-11"
  run: "2026-09-12 - 73b4216"
  results: "781bf7a, random zones 84d1b9f"
  verdict: "the mechanism works and beats generic importance (L6, both pairs); the address does not beat the same memory without a mask (L3)"
---

# E010 - Lens layout

The whole network behind a glass, lenses in the query's expert zones, memory following the lenses - the test of the idea itself. The mechanism is [docs/lens.md](../../docs/lens.md) at commit `4fa3e7f`.

- Preregistration: [ADDENDUM-11](ADDENDUM-11.md). Results: [results.md](results.md). Runs: `runs/E010-lens-layout/`.
- Random zones, the floor of the comparison, were run separately on the same commit: `runs/E010-lens-layout-random/`.
