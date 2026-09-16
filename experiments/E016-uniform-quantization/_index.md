---
title: "E016 - Uniform quantization on the corpus"
date: 2026-09-15
weight: 16
hypotheses: [H6]
statuses: [done]
params:
  fixed: "cab0c69"
  run: "2026-09-16 - answers 0e08906, the reasoning judge 5b54c9e, bf16 on the unknown share again 5b54c9e"
  results: "e082414"
  verdict: "D8 keeps 98.8%, D6 96.6%, D4 85.7% - losing the facts in the weights first (19.8% against 5.6% with the passage) to wrong answers, not garbage; naive D2 is garbage (99.5%); on the unknown share D4 answers where bf16 refuses and is accepted 6.3 points more"
---

# E016 - Uniform quantization on the corpus

How the knowledge of gemma-4-E2B-it degrades when every block is read at the same depth: the share of
what it knows at bf16 that D8, D6, D4 and D2 keep, on the frozen corpus, by regime, judged by the model
at bf16. Preregistration: [PREREG](PREREG.md).

It gives [H6](../../docs/hypotheses.md) its half about coarsening: the curve over the steps and the step at
which coarsening slides into nonsense. The comparison with a request for a shorter answer, the junction of
areas and the fact-by-fact count stay with H6.

The curve it measures is also the base every test of the quantization filter is compared with: uniform
quantization at the same memory is what a deployment would otherwise do.

The runs: the answers of every level in `runs/E016-uniform-quantization/e2b-it/answers/`, the verdicts of the
reasoning judge in `.../e2b-it/judge/` (one run per pass over a file, see [the data](../../docs/data.md)), and
bf16 answering the unknown share again in `.../bf16-unknown-again/`. The judge that reads them is described in
[the corpus](../../docs/corpus.md). Results: [results.md](results.md).
