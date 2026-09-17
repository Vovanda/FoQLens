---
title: "E002 - Uniform quantization with a working base precision D2"
date: 2026-09-16
weight: 17
hypotheses: []
statuses: [done]
params:
  fixed: ""
  run: "2026-09-17 - the k-quant ladder and two GGUF, judge 5b54c9e"
  results: ""
  verdict: "D2 keeps 51.4% of bf16's knowledge, D4 91.0%, D6 96.8%, D8 98.8%"
---

# E002 - Uniform quantization with a working base precision D2

After E001 the filter could only be tested from D4: the naive D2 answered with strings of symbols. Yet two-bit GGUF
files of the same checkpoint answered the same words sensibly ([associations](exploration-associations.md)). So the
model survives two bits, and what failed was our quantization method. I compared what these files keep above two
bits and saw that raising the k, v, o, down and per-layer modules brings the knowledge back
([module classes](exploration-module-classes.md)). That alone is not enough: our slices with this layout still give
garbage, the quantization method has to change too ([topology](exploration-topology.md)).

I moved k-quant from llama.cpp into the bench, as one copy with residual slices over it, and put these classes on
Q4_K. Before the measurement I decided the floor had to keep at least half of bf16's knowledge
([why](../../notes/2026-09-17-floor-at-least-half.md)), though I expected less. On the whole corpus D2 keeps 51.4%,
D4 91.0%, D6 96.8%, D8 98.8% ([results](results.md)).

Now the filter is tested from D2, and this ladder is the baseline for its tests. If I change the topology, the layout
or the precision of the classes, I run the corpus again on the new configuration ([preregistration](PREREG.md)).

When D2 ran into the answer cap, I gave it 1,024 tokens instead of 512: 40 ARC-Challenge solutions reached the right
answer ([cut answers](exploration-reask.md)).
