---
title: "E001 - Run 1: separation, masks, overlap, geometry, confirmation"
date: 2026-09-11
weight: 1
hypotheses: [H0, H1, H2, H2+]
statuses: [reset]
params:
  fixed: "2026-09-11 - preregistration be66b77, ADDENDUM-01 39e90ab, THRESHOLDS-01 aed8a29"
  run: "2026-09-11 - step0 3118130, calibration ac393f5, step1 c407c45, step2plus 9e008e3, confirm 8214822, diagnostic c9f8cb2"
  results: "d20428c"
  verdict: "reset 2026-09-13 - nothing is claimed from this run until the corpus, the metric and the mask are ones the bench trusts (docs/corpus.md)"
---

# E001 - Run 1

The first pass of steps 0-2+ on Gemma 4 E2B with the naive block score, then the confirmation on the held-out pair.

- Preregistration: [main](../../prereg/PREREGISTRATION.ru.md) (steps 0-2+), [ADDENDUM-01](ADDENDUM-01.md) (three center modes), [THRESHOLDS-01](THRESHOLDS-01.md) (instrument and pass criteria for the confirmation).
- Results: [results.md](results.md).
- Runs: `runs/E001-run1-exploration/` - `step0`, `calibration`, `step1`, `step2plus`, `confirm`, `diagnostic`.
- Verdict at the time, withdrawn 2026-09-13: topics separate in representations (ARI 0.98); masks separate in their means after background subtraction and zones overlap in exploration, but neither holds on the held-out pair (history-geography p 0.85), which barely separates in the model's own representations.
