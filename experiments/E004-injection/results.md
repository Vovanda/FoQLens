# Results - mask injection on E2B (experiments/E004-injection/ADDENDUM-04.md)

Run 2026-09-11 after ADDENDUM-04 was committed (`7848717`). 395 questions (biology 95, math 100, history 100, geography 100); outside the aperture every block is removed (ZERO). Raw numbers: [runs/E004-injection/e2b/summary.json](../../runs/E004-injection/e2b/summary.json).

## Verdict against the predictions

| Prediction | Result |
| --- | --- |
| I1 own-topic mask beats the other topic's | **not supported**: own - other overlaps zero at every aperture for both sources and pairs; the one significant point goes the other way (history-geography, pooled, a = 0.99: -0.124 [-0.238, -0.011]) |
| I2 larger effect for the polar pair | not applicable: no effect in either pair |
| I3 the question's own mask adds little | **not supported**: the question's own mask is usually *worse* than its topic mask |

## What the absolute levels show

Mean right-letter log-probability (uniform bf16: -1.13 biology-math, -0.89 history-geography; every block removed: -1.39, the uniform guess).

| Aperture | random | pooled self / own / other | gradient self / own / other |
| --- | --- | --- | --- |
| biology-math 0.99 | **-1.34** | -2.35 / -2.16 / -1.91 | -3.24 / -4.21 / -4.31 |
| biology-math 0.95 | **-2.09** | -4.28 / -3.00 / -3.14 | -4.54 / -5.74 / -5.73 |
| biology-math 0.80 | -4.51 | -6.02 / -6.58 / -5.97 | -6.05 / **-4.36** / **-3.80** |
| history-geography 0.99 | **-1.22** | -1.97 / -1.56 / -1.44 | -2.96 / -2.54 / -2.48 |
| history-geography 0.90 | -2.90 | -4.19 / -3.45 / -3.73 | -5.59 / **-1.92** / **-2.07** |
| history-geography 0.80 | -4.82 | -5.65 / -6.40 / -6.51 | -6.53 / **-2.99** / **-2.72** |

- **At apertures 0.9-0.99 every mask-directed layout is worse than random blocks**, the question's own mask included. The rule "keep the blocks whose score is highest after subtracting the background" removes the wrong blocks: a block that is large and active for every question has a background-subtracted score near or below zero, so it goes first - and those are the blocks the model cannot live without. Random removal rarely hits them.
- **Under heavy removal (0.8-0.9) gradient topic masks beat random** (history-geography 0.8: -2.99 vs -4.82) - but own and other are equal, so what they capture is general block importance, not the topic.

## What it means

The masks carry the topic (a linear probe reads it at 95% on the hard pair), but allocating precision by the background-subtracted mask alone throws away the shared foundation. The preregistered topology already predicted such a foundation (property 5, hierarchy). The next design keeps it: a generic importance backbone always sharp, and only the remaining budget allocated by the topic - tested as backbone + own topic vs backbone + other topic vs backbone alone.
