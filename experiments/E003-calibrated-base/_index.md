---
title: "E003 - The ladder over a calibrated base"
date: 2026-09-18
weight: 18
hypotheses: []
statuses: [fixed]
params:
  fixed: ""
  run: ""
  results: ""
  verdict: ""
---

# E003 - The ladder over a calibrated base

Our base precision D2 keeps 51.4% of bf16's knowledge, a published two-bit file 80.0% (E002). The model format lets the
stack stand on another quantizer's base: I cut E2B-it over bartowski's Q2_K, quantized with an imatrix, and answer the
frozen corpus at D2, D4, D6 and D8 as E002's ladder did. I expect the quality to rise at every level. Preregistration: [PREREG](PREREG.md).
