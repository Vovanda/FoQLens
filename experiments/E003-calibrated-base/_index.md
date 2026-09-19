---
title: "E003 - The ladder over a calibrated base"
date: 2026-09-18
weight: 18
hypotheses: []
statuses: [done]
params:
  fixed: "c89bede"
  run: "2026-09-18..19 - the ladder D2-D8 over bartowski's Q2_K, judge 02171b8, the kernel round (5de1fdd, 38757ad)"
  results: ""
  verdict: "D2 keeps 76.7% of bf16's knowledge (51.4% on our base), D4 90.8%, D6 96.2%, D8 97.5%; T1, T2 met, P1 not at D6, D8"
---

# E003 - The ladder over a calibrated base

Our base precision D2 keeps 51.4% of bf16's knowledge, a published two-bit file 80.0% (E002). The model format lets the
stack stand on another quantizer's base: I cut E2B-it over bartowski's Q2_K, quantized with an imatrix, and answer the
frozen corpus at D2, D4, D6 and D8 as E002's ladder did. I expect the quality to rise at every level. Preregistration: [PREREG](PREREG.md), results: [results](results.md).
