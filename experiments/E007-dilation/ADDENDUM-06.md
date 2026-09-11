# Addendum 06 - dilation: widening the sharp windows of the topic mask

Fixed 2026-09-11, **before the run**. Not edited after its commit.

## 1. Why

Over a generic importance backbone the topic fill added nothing where the model still answers ([results](../E005-backbone/results.md)). One explanation, raised by the author: a mask points at blocks one by one, while what a topic uses may be wider - a sharp block whose partners are removed does not help. Dilation keeps the aperture and spends it on fewer seeds, each widened to its neighbours.

## 2. Design

- Questions, pairs, the backbone (mean raw gradient x activation mask) and ZERO outside the aperture as in ADDENDUM-05. Backbone share 0.8 of the aperture. Apertures 0.97, 0.95, 0.9 - at 0.8 and below every policy of ADDENDUM-05 was at chance.
- Fill: the own or the other topic's mask (pooled and gradient sources, background subtracted, leave-one-out).
- The fill order is dilated: each fill block is followed at once by its neighbours, until the aperture is spent.
  - `none` - as in ADDENDUM-05;
  - `struct` - structural neighbours: the gate and up rows of the same block in the same layer (the same MLP neurons); the residual-stream writers (o_proj, down_proj, per_layer_projection) with the same block index in the same layer and in the layers L-1 and L+1 (the same stream coordinates);
  - `index` - control: for every block, as many of the nearest blocks of the same matrix as it has structural neighbours.
- Also: the backbone alone, backbone + random fill, uniform bf16 and ZERO.
- Comparison: paired bootstrap over questions (10 000 resamples, seed 0) of the right-letter log-probability, 95% interval. A cell is one pair x source x aperture: 6 cells per pair.

## 3. Predictions

A prediction **holds for a pair** when at least 2 of its 6 cells have the interval above zero and none below. With no effect, 2 or more of 6 cells above zero by chance has a probability of about 1%.

- **W1 - the address appears with width.** own - other under `struct` dilation.
- **W2 - structure, not width alone.** `struct` - `index`, own fill.
- **W3 - width helps.** `struct` - `none`, own fill.

If none holds for either pair, the width of the mask's windows is not why the topic address failed.
