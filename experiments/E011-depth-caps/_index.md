---
title: "E011 - Depth caps: when storage follows the queries"
date: 2026-09-12
weight: 11
hypotheses: []
statuses: [done]
params:
  fixed: "2026-09-12 - ADDENDUM-12"
  run: "2026-09-12 - c1bd457"
  results: "c1bd457"
  verdict: "caps save storage only for a narrow profile: one topic stores 4.0 bits of 8, 395 mixed questions store 6.5; addressing saves reading, not the file"
---

# E011 - Depth caps

A layout is read per query, storage is one copy: a block keeps the deepest slice any query asks of it. How much that saves, against how many queries and how varied they are.

- Preregistration: [ADDENDUM-12](ADDENDUM-12.md). Results: [results.md](results.md). Runs: `runs/E011-depth-caps/`.
- An engineering measurement, no hypothesis at stake: reading is unchanged by a cap, so quality is not measured again.
