---
title: "E002 - Gradient x activation as the block score"
date: 2026-09-11
weight: 2
hypotheses: [H1]
statuses: [reset]
params:
  fixed: "2026-09-11 - ADDENDUM-02 3b338ef"
  verdict: "reset 2026-09-13 - nothing is claimed from this run until the corpus, the metric and the mask are ones the bench trusts (docs/corpus.md)"
---

# E002 - Gradient x activation

The second instrument after the naive score failed: per block, gradient x activation of the query's own language-model loss.

- Preregistration: [ADDENDUM-02](ADDENDUM-02.md).
- Not run as its own step 1 check on the held-out pair. The instrument is the `gradient` mask source of every later experiment (E004 on).
