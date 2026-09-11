# Addendum 05 - backbone + topic: the address on top of the scale

Fixed 2026-09-11, **before the run**. Not edited after its commit.

## 1. Why

The injection run ([results](../docs/results-injection.md)) showed that ranking blocks by their background-subtracted score removes the shared foundation first: at apertures 0.9-0.99 every mask-directed layout was worse than random blocks. The masks do carry the topic (a linear probe reads it at 95%), so the fix is structural: a generic importance **backbone** is always read sharp, and only the rest of the aperture is allocated by the topic. The backbone is the scale, the topic is the address.

## 2. Design

- Questions, pairs and aperture grid as in ADDENDUM-04: biology-math, history-geography; apertures 0.99, 0.98, 0.97, 0.95, 0.9, 0.8, 0.5; outside the aperture every block is removed (ZERO).
- **Backbone:** generic block importance = the mean raw gradient x activation mask over all questions of the run (no background subtracted); the same for every question.
- **Share:** the backbone takes a share s of the aperture (s = 0.5, 0.8, 0.95); the rest of the aperture is filled by:
  - `own` - the question's topic mask (leave-one-out, background subtracted), pooled and gradient sources;
  - `other` - the paired topic's mask;
  - `random` - random blocks.
- Also: `backbone` alone for the whole aperture, random blocks for the whole aperture, uniform bf16 and ZERO.
- Comparison: paired bootstrap over questions (10 000 resamples, seed 0) of the right-letter log-probability, 95% interval.

## 3. Predictions

- **B1.** The backbone alone beats random blocks at the same aperture (static importance works).
- **B2 - the address.** Backbone + own topic beats backbone + other topic at the same aperture and share: the interval of own - other lies above zero somewhere on the grid.
- **B3.** Backbone + own topic beats backbone + random fill.

If own, other and random fill are indistinguishable over the backbone everywhere, an untrained topic address adds nothing to static importance: the directed idea would reduce to known importance-based allocation.
