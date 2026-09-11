# Addendum 09 - the matrix of precision x spread, and the parameter names

Fixed 2026-09-11, **before the run**. Not edited after its commit.

## 1. Names

The parameters are named by what they do ([docs/zones.md](../docs/zones.md)); the earlier documents read with this table:

| Earlier | Now |
| --- | --- |
| aperture (ADDENDUM-03 to 06) | **precision** `p` in [0, 1] - the share of the precision range spent; with two levels, the share of weights read at the higher one |
| regulator `s` (ADDENDUM-07), focus `f` (ADDENDUM-08, results-zones) | **spread** `s` in [0, 1] - 0 zones at their centers (hard edge), 0.5 as found, 1 spread over the map (no mask); `R = r s / (1 - s)`, `s = 1 - s_ADDENDUM-07 = f` |

## 2. Why

At one precision (5 bits) the expert zones of the gradient mask beat random zones, the paired topic and no mask for biology-math, not for history-geography ([results](../docs/results-zones.md)). One cell of precision does not say where the address lives or whether it grows with the budget. This run spans both parameters.

## 3. Design

- Questions, pairs, weight map (co-activation of the raw gradient masks, no labels), zones and the D4 / D6 / D8 levels as in ADDENDUM-07.
- **Grid:** precision 0.125, 0.25, 0.5, 0.75 (4.5, 5, 6, 7 bits per weight) x spread 0, 0.2, 0.333, 0.5, 0.667, 0.8 - the spread steps halve or double the radius (`s / (1 - s)` = 0, 1/4, 1/2, 1, 2, 4). Spread 1 is the reference of every row: the precision spread without a mask.
- **In every cell:** own-topic zones, the paired topic's zones, random zones of the same count and radii, the backbone's zones; pooled and gradient sources.
- **Order:** the cells run from the most promising to the least - nearest to precision 0.375, spread 0.5 first - and the summary is written after every cell.
- **After the grid:** cells between the best ones may be added; they are exploration and do not count for the predictions.
- **Comparison:** paired bootstrap over questions (10 000 resamples, seed 0) of the right-letter log-probability, 95% interval; a cell is one pair x source x precision x spread.

## 4. Predictions - the gradient source

The pooled zones failed in ADDENDUM-07 and are reported as a description. For the gradient source a prediction **holds for a pair** when at least 5 of its 24 cells have the interval above zero and none below (by chance well below 0.1%).

- **M1 - the zones find the topic.** Own zones beat random zones of the same count and radii.
- **M2 - the address.** Own zones beat the paired topic's zones.
- **M3 - zones beat no mask.** Own zones beat the precision spread without a mask.

Expected from ADDENDUM-07: M1-M3 hold for biology-math; for history-geography M1 may hold and M2 is open.

## 5. Reproducing ADDENDUM-07

`scripts/step3_zones.py --precision 0.25 --spread 0.8 0.65 0.5 0.35 0.2 0.0`.
