# Results - expert zones on E2B at a fixed budget (experiments/E008-zones-fixed-budget/ADDENDUM-07.md, terms: ADDENDUM-08)

> **Legacy approach.** A fixed-budget layout, like the matrix ([results](../E009-zones-matrix/results.md)); the lens layout of ADDENDUM-11 replaces it.

Run 2026-09-11 on the bench of `8d68874`, after ADDENDUM-07 (`fddb919`) and ADDENDUM-08 (`84cb427`). 395 questions (biology 95, math 100, history 100, geography 100); weight map from the raw gradient masks of all questions; every layout at 5 mean bits (D4 background, D6, D8 centers). Raw numbers: [runs/E008-zones-fixed-budget/e2b/summary.json](../../runs/E008-zones-fixed-budget/e2b/summary.json).

**Scale of this run.** The run used the first scale of the regulator, `R = r (1 - s) / s`, where s = 1 collapses the zones and s -> 0 spreads them. The agreed focus is the mirror of it, `R = r f / (1 - f)` with f = 1 - s ([docs/zones.md](../../docs/zones.md)). The tables below give both.

References (mean right-letter log-probability): bf16 -1.012, uniform D8 -1.012, uniform D4 -1.138, the 5-bit budget spread without a mask -1.075. Zones per question: pooled 4.7, gradient 3.4, backbone 5.

## Verdict against the predictions

A prediction holds for a pair when at least 3 of its 12 cells (source x scale) have the 95% interval above zero and none below.

| Prediction | biology-math | history-geography | Result |
| --- | --- | --- | --- |
| S1 own zones beat random zones (same count and radii) | 4 above / 0 below | 3 / 1 | **holds for biology-math** |
| S2 own zones beat the paired topic's | 6 / 1 | 0 / 0 | not met (the one negative cell is a pooled one) |
| S3 own zones beat no mask at the same budget | 6 / 0 | 0 / 5 | **holds for biology-math** |

## By source

Mean right-letter log-probability of the own-topic zones and the differences, biology-math:

| s (run) | f (agreed) | own gradient | own - random | own - other | own pooled | own - random |
| --- | --- | --- | --- | --- | --- | --- |
| 0.20 | 0.80 | -1.046 | +0.043 | +0.059 | -1.089 | -0.018 |
| 0.35 | 0.65 | -1.045 | +0.048 | +0.069 | -1.108 | -0.024 |
| 0.50 | 0.50 | -1.048 | +0.066 | +0.077 | -1.126 | -0.010 |
| 0.65 | 0.35 | -1.049 | +0.073 | +0.090 | -1.122 | -0.004 |
| 0.80 | 0.20 | -1.058 | +0.067 | +0.087 | -1.118 | -0.002 |
| 1.00 | 0.00 | -1.056 | +0.081 | +0.097 | -1.121 | +0.002 |

(The log-probabilities are over all four topics; the differences are over the pair.)

- **Zones from the gradient mask carry an address for the polar pair.** Their point estimates are above random zones of the same count and radii, above the paired topic's zones and above the budget without a mask at every scale; the intervals lie above zero in 4 of 6 cells against random zones, 6 of 6 against the paired topic and 6 of 6 against no mask, none below. At 5 bits they recover about 73% of the gap between D4 and D8 (-1.046 against -1.138 and -1.012), the budget without a mask 50%.
- **Zones from the pooled mask do not.** No pooled cell is above zero in any comparison; one is below the paired topic on biology-math and five below the budget without a mask on history-geography.
- **The hard pair is not separated.** On history-geography the gradient zones beat random zones in 3 of 6 cells (+0.04 to +0.06, none below) - on the gradient source alone S1 would hold there; the preregistered count over both sources fails on one pooled cell. They do not beat the paired topic's zones (-0.01 to -0.03, within noise): the zones find what matters for the questions, not which of the two topics they are about.
- The generic importance backbone's zones (-1.067 to -1.082) are between the budget without a mask and the own gradient zones.

## What it means

This is the first layout in the project where the own topic beats the other topic at the same budget - and it came from giving the mask the shape of the picture: centers, radii, a stepped falloff. The address is there for topics that are far apart (biology and math) and not yet for close ones (history and geography).

Next: the agreed focus scale, where the budget follows the zones and every comparison is made at the same mean of bits; the gradient source first.

## Bench

Mask phase: GPU utilization 45% mean, peak 15.5 GB allocated (gradient batch 8). Evaluation: 50% mean, peak 11.9 GB. The run took the whole card: the GPU share limit came after it.
