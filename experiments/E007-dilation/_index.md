---
title: "E007 - Dilation of the topic fill"
date: 2026-09-11
weight: 7
hypotheses: [H3.1]
statuses: [reset]
params:
  fixed: "2026-09-11 - ADDENDUM-06 a2f93f7"
  run: "2026-09-11 - 427613d"
  results: "01d208e"
  verdict: "reset 2026-09-13 - nothing is claimed from this run until the corpus, the metric and the mask are ones the bench trusts (docs/corpus.md)"
---

# E007 - Dilation

The topic fill widened to structural neighbours (same neuron, same stream coordinate) against index neighbours.

- Preregistration: [ADDENDUM-06](ADDENDUM-06.md). Results: [results.md](results.md). Runs: `runs/E007-dilation/`.
- Verdict: the model's blocks work in structural groups, but width is not why the topic address failed.
