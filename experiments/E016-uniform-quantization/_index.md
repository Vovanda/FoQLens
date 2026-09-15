---
title: "E016 - Uniform quantization on the corpus"
date: 2026-09-15
weight: 16
hypotheses: [H6]
statuses: [fixed]
params:
  fixed: "cab0c69"
  run: ""
  results: ""
  verdict: ""
---

# E016 - Uniform quantization on the corpus

How the knowledge of gemma-4-E2B-it degrades when every block is read at the same depth: the share of
what it knows at bf16 that D8, D6, D4 and D2 keep, on the frozen corpus, by regime, judged by the model
at bf16. Preregistration: [PREREG](PREREG.md).

It gives [H6](../../docs/hypotheses.md) its half about coarsening: the curve over the steps and the step at
which coarsening slides into nonsense. The comparison with a request for a shorter answer, the junction of
areas and the fact-by-fact count stay with H6.
