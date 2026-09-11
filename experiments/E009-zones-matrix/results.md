# Results - expert zones over precision share x focus area on E2B (experiments/E009-zones-matrix/ADDENDUM-09.md, names: ADDENDUM-10)

> **Legacy approach, checked along the way.** This run uses the fixed-budget layout: the mean bits are set in advance and the zones are stretched or shrunk to hit them, so even at focus area 0 precision is pushed into the zones. That is not the project's picture - frosted glass with lenses in the expert zones, where memory follows the lenses - which ADDENDUM-11 tests. The matrix was run because it was cheap on the bench already built; its predictions held, but it is not the test of the idea.

Run 2026-09-11/12 on the bench of `287d5b9`, after ADDENDUM-09 (`ac10204`); the names of ADDENDUM-10 (`db61af1`) are used here, while the summary keeps the names of the run's code - `precision` / `p` for precision_share, `spread` / `s` for focus_area. 395 questions (biology 95, math 100, history 100, geography 100); weight map from the raw gradient masks of all questions; levels D4 background, D6, D8 centers. GPU share 0.8. Raw numbers: [runs/E009-zones-matrix/e2b/summary.json](../../runs/E009-zones-matrix/e2b/summary.json).

References (mean right-letter log-probability over all four topics): bf16 -1.012, uniform D8 -1.012, uniform D4 -1.138. No mask at each precision share: 4.5 bits -1.090, 5 bits -1.075, 6 bits -1.014, 7 bits -1.007. Zones per question: pooled 4.7, gradient 3.4, backbone 5.

## Verdict against the predictions

Gradient source; a prediction holds for a pair when at least 5 of its 24 cells have the 95% interval above zero and none below.

| Prediction | biology-math | history-geography |
| --- | --- | --- |
| M1 own zones beat random zones (same count and radii) | 11 above / 0 below - **holds** | 5 / 0 - **holds** |
| M2 own zones beat the paired topic's | 15 / 0 - **holds** | 0 / 1 - not met |
| M3 own zones beat no mask at the same precision share | 7 / 0 - **holds** | 0 / 4 - not met |

As expected in ADDENDUM-09: M1-M3 for biology-math, M1 for history-geography; M2 for history-geography, left open there, is not met.

## Biology-math, gradient zones

Differences of the right-letter log-probability over the pair; rows precision share (bits per weight), columns focus area; `*` the 95% interval above zero, `!` below.

**Own - random zones (M1)**

| precision share | 0 | 0.2 | 0.333 | 0.5 | 0.667 | 0.8 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.125 (4.5 bits) | +0.062* | +0.076* | +0.082* | +0.097* | +0.086* | +0.085* |
| 0.25 (5 bits) | +0.078* | +0.071* | +0.076* | +0.063* | +0.035 | +0.036 |
| 0.5 (6 bits) | +0.053* | +0.032 | +0.032 | +0.028 | +0.030 | +0.005 |
| 0.75 (7 bits) | +0.029 | +0.045 | +0.041 | +0.032 | +0.022 | +0.017 |

**Own - the paired topic's zones (M2)**

| precision share | 0 | 0.2 | 0.333 | 0.5 | 0.667 | 0.8 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.125 | +0.051* | +0.073* | +0.078* | +0.108* | +0.083* | +0.082* |
| 0.25 | +0.094* | +0.084* | +0.098* | +0.078* | +0.071* | +0.056* |
| 0.5 | +0.080* | +0.053* | +0.035 | +0.033 | +0.029 | +0.001 |
| 0.75 | +0.010 | +0.033 | +0.027 | +0.021 | +0.014 | +0.017* |

**Own - no mask (M3)**

| precision share | 0 | 0.2 | 0.333 | 0.5 | 0.667 | 0.8 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.125 | +0.025 | +0.036 | +0.040 | +0.057* | +0.046 | +0.048 |
| 0.25 | +0.062* | +0.054* | +0.067* | +0.074* | +0.064* | +0.064* |
| 0.5 | +0.006 | -0.003 | -0.007 | +0.003 | +0.019 | +0.006 |
| 0.75 | -0.012 | +0.011 | +0.012 | +0.014 | +0.002 | +0.008 |

## History-geography, gradient zones

| precision share | own - random, per focus area 0 ... 0.8 | own - other | own - no mask |
| --- | --- | --- | --- |
| 0.125 | +0.007 +0.027 +0.034 +0.034 +0.043 +0.048 | -0.032 ... +0.003 | -0.035 ... +0.010 |
| 0.25 | +0.050 +0.059* +0.079* +0.047 +0.041 +0.044 | -0.027 ... -0.001 | -0.018 ... +0.004 |
| 0.5 | +0.043 +0.039 +0.048 +0.030 +0.045* +0.027 | -0.029! ... +0.016 | -0.053! -0.047! -0.045! -0.031! -0.011 +0.003 |
| 0.75 | +0.038 +0.041* +0.036 +0.050* +0.030 +0.018 | -0.012 ... +0.010 | -0.029 ... -0.007 |

## What the matrix shows

- **The address lives where bits are scarce.** For biology-math the own zones beat random zones in all six cells at 4.5 bits, four of six at 5 bits, one at 6 and none at 7; the point estimates fall from +0.06 ... +0.10 to +0.02 ... +0.05. Against the paired topic's zones the picture is the same: 12 of 12 cells above zero at 4.5 and 5 bits.
- **Zones beat no mask only at 5 bits** (6 of 6 cells, one more at 4.5). At 6 and 7 bits the budget without a mask is already at bf16 (-1.014 and -1.007 against -1.012): there is nothing left to win. At 4.5 bits the own zones are above no mask in every point estimate, but only one interval clears zero.
- **At 5 bits** the own gradient zones at focus area 0.5 read -1.045: 74% of the gap between D4 (-1.138) and D8 (-1.012), against 50% for the same budget without a mask - the result of ADDENDUM-07 (-1.048) reproduced.
- **The focus area matters less than the precision share.** Within a row the differences are smaller than between rows. The hard edge (focus area 0) is the weakest of its row at 4.5 and 7 bits (-1.095 against -1.063 at focus area 0.5; -1.027 against -1.004, all four topics) and close to the rest at 5 bits (-1.054 against -1.045): a falloff around the centers is better than a plateau.
- **The close pair is not separated.** On history-geography the own zones are above random zones in all 24 point estimates (5 intervals above zero), but not above the paired topic's zones (-0.03 ... +0.02) and below no mask at 6 bits in four cells: the gradient zones find what matters for these questions, not which of the two topics they are about.
- **Pooled zones fail again.** Biology-math: 0 cells above random zones, 5 below the paired topic, 8 below no mask; history-geography: 21 of 24 below no mask.
- **The backbone's zones** are below no mask from 6 bits on (-1.048 against -1.014 at 6 bits, focus area 0.5): a fixed set of zones for all questions is worse than spending evenly once the budget is large.

## What it means

The zones of the gradient mask carry an address for topics far apart, and the address is worth most when precision is scarce - the regime the project is about, memory saving. Above 6 bits the model is at bf16 anyway and no layout can help. For close topics the zones mark a shared region of the weights: they beat random zones of the same shape but carry no topic-specific address. Next is making the address sharper for close topics, and taking it online from the first layers instead of from a full gradient pass.

## Bench

Mask phase: GPU utilization 30% mean (median 19%), reserved peak 16.6 GiB. Evaluation: 40% mean (median 24%), reserved peak 13.2 GiB; both within the 19.2 GiB of the 0.8 share. The utilization is well below the share - a bug under the bench standards, now the first item of work. The numbers do not depend on it: the pauses and the speed change when a batch runs, not what it computes, and every comparison runs on the same batches.
