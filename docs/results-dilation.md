# Results - dilation on E2B (prereg/ADDENDUM-06.md)

Run 2026-09-11 after ADDENDUM-06 was committed (`a2f93f7`), code `88c49f0`. 395 questions (biology 95, math 100, history 100, geography 100); backbone share 0.8 of the aperture; outside the aperture every block is removed (ZERO). 2.06 neighbours per block on average in both tables (q/k/v and the per-layer input gate have none; gate/up one; residual-stream writers up to eight). Raw numbers: [runs/dilation/e2b/summary.json](../runs/dilation/e2b/summary.json).

## Verdict against the predictions

A prediction holds for a pair when at least 2 of its 6 cells (source x aperture) have the 95% interval above zero and none below.

| Prediction | biology-math | history-geography | Result |
| --- | --- | --- | --- |
| W1 own - other under struct dilation | 2 above / 1 below | 0 / 0 | **not supported** |
| W2 struct - index, own fill | 2 / 0 | 5 / 0 | **supported for both pairs** |
| W3 struct - none, own fill | 1 / 0 | 4 / 0 | **supported for history-geography** |

## What the absolute levels show

Mean right-letter log-probability (uniform bf16 -1.012).

| Aperture | backbone alone | random fill | own pooled: none / struct / index | other pooled: none / struct / index |
| --- | --- | --- | --- | --- |
| 0.97 | **-1.063** | -1.061 | -1.113 / -1.070 / -1.113 | -1.161 / -1.118 / -1.171 |
| 0.95 | **-1.110** | -1.135 | -1.187 / -1.125 / -1.220 | -1.247 / -1.194 / -1.285 |
| 0.90 | **-1.245** | -1.351 | -1.371 / -1.373 / -1.394 | -1.464 / -1.360 / -1.455 |

- **The structure is real.** Taking a block together with the rows that compute with it - the gate and up rows of the same neurons, the writers of the same stream coordinates - costs less than taking the same number of index neighbours, and often less than taking the next blocks by the mask.
- **It is not an address.** Structural dilation helps the other topic's fill as much as the own topic's; own - other stays at zero. What it improves is generic importance, not the topic.
- **The backbone alone is still best** at every aperture: any topic fill, dilated or not, costs quality against spending the same budget on generic importance.

## What it means

The width of the mask's windows is not why the topic address failed (W1). But the model's blocks do work in structural groups (W2), so a layout should keep a neuron's gate and up rows together and a stream coordinate across its writers. The natural next use is to dilate the generic importance itself, and to allocate memory (per-block storage depth) in structural groups.

## Bench

Mask phase: GPU utilization 24% mean (the gradient pass). Evaluation: 54% mean, peak 9.9 GB allocated.
