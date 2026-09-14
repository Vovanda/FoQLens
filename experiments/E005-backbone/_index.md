---
title: "E005 - Backbone + topic fill"
date: 2026-09-11
weight: 5
hypotheses: [H3.1, H3.3]
statuses: [reset]
params:
  fixed: "2026-09-11 - PREREG e1f8a6d"
  run: "2026-09-11 - 72efa3f"
  results: "16bc38b"
  verdict: "reset 2026-09-13 - nothing is claimed from this run until the corpus, the metric and the mask are ones the bench trusts (docs/corpus.md)"
---

# E005 - Backbone + topic

Generic block importance for a part of the blocks kept, the rest filled by the own topic, the other topic or random blocks.

- Preregistration: [PREREG](PREREG.md). Results and runs deleted with the corpus (`1e89a81`).
- Verdict at the time, withdrawn 2026-09-13: static importance carries the budget; an untrained topic address adds nothing on top of it where the model still answers.
