---
title: "E005 - The precision map of a query"
date: 2026-09-20
weight: 20
hypotheses: [H3, H4]
statuses: [done]
params:
  fixed: ""
  run: "2026-09-19..20 - the oracles over 2% of the corpus, then over 103 questions only the top rung answers"
  results: ""
  verdict: "On the hard questions a map holds 0.816 of the answers at 0.523 bytes; the uniform D4 at 0.558 holds 0.000 and the uniform D6 at 0.775 holds 0.039"
---

# E005 - The precision map of a query

Does a layout of precision built for one question hold knowledge more cheaply than a uniform rung of the same memory.
The oracles build such a map by looking at the answer - they cannot run at inference and are here as a ceiling: if even
they have no map, there is no mechanism to build. Preregistration: [PREREG](PREREG.md), results: [results](results.md).

## Where the project stands, 2026-09-20

- **The corpus is there.** The frozen small corpus, with the ladder measured on it: the whole network at D2 keeps 76.7%
  of the questions the model knows, the uniform D4 90.8%, D6 96.2%, the whole D8 97.5% (E003).
- **The left half is there** - the address of a query: the hybrid of neuron activity and head energy identifies a
  paraphrase in 0.917 of the cases, and the first 4-8 layers at base precision give the same address as a full
  pass (E004).
- **The right half is there** - the precision map: this experiment shows that a layout built for the question holds
  knowledge at half the memory where a uniform rung loses it entirely.
- **What is missing** is a cheap predictor of the map. A bridge from the address to the map is not distinguishable from
  a constant map by the answers. The layer-wise regulator (variant B) decides the layout as the pass runs and needs no
  map predicted in advance; that is where Volodya puts his bet.
