# Results - the lens layout on E2B (experiments/E010-lens-layout/ADDENDUM-11.md)

Run 2026-09-12 on the bench of `73b4216`, after ADDENDUM-11 (`efa863b`); the mechanism is [docs/lens.md](../../docs/lens.md) at `4fa3e7f`. 395 questions (biology 95, math 100, history 100, geography 100); weight map from the raw gradient masks of all questions; 3.7 zones per question, 5 in the backbone. GPU share 0.8. Raw numbers: [runs/E010-lens-layout/e2b/summary.json](../../runs/E010-lens-layout/e2b/summary.json).

References (mean right-letter log-probability): bf16 -1.012 at 16 bits, uniform D8 -1.012 at 8, uniform D6 -1.014 at 6, uniform D4 -1.138 at 4. The floors alone: D4 -1.138, D2 -2.750, ZERO -1.386.

## Verdict against the predictions

A prediction holds for a pair and a floor when at least 3 of its 10 cells have the 95% interval above zero and none below. Counts are cells above / below zero.

| Prediction | biology-math D4 | D2 | ZERO | history-geography D4 | D2 | ZERO |
| --- | --- | --- | --- | --- | --- | --- |
| L2 own beats the paired topic | 1/0 | **6/0 holds** | 1/2 | 1/0 | 0/0 | 1/3 |
| L3 own beats the same memory, no mask | 2/0 | 2/0 | 2/1 | 0/0 | 1/2 | **4/0 holds** |
| L6 own beats the backbone's zones | **3/0 holds** | **6/0 holds** | 2/2 | **4/0 holds** | **7/0 holds** | 2/8 |
| L5 own beats the bare floor (D2) | - | **4/0 holds** | - | - | **4/0 holds** | - |
| H4 own beats bf16 (exploratory) | 0/8 | 0/10 | 0/10 | 0/8 | 0/9 | 0/9 |
| L1 own beats random zones | 0/0 | 0/5 | **6/2** | 0/6 | 2/6 | 4/2 |

- **L6 holds for both pairs** behind the D4 and the D2 floor: the zones of the query beat one fixed set of zones for every question. This is the sharpest result of the run - it separates "the query matters" from "some blocks are simply important", which E005 left open.
- **L5 holds for both pairs**: behind a D2 floor, where the model alone is broken (-2.750), lenses carry it back - up to -1.18 at 5.6 bits.
- **L2 holds only for biology-math behind the D2 floor.** The address shows where bits are scarce and the topics are far apart, as in E009; a close pair carries none.
- **L3 does not hold** by its criterion, except for history-geography behind an empty floor - but see the next section: the two cells strong enough to test it do show it for biology-math, and the criterion counted cells that lift 0.4% of the net alongside cells that lift 56%.
- **H4 fails as expected.** No cell is above bf16; every floor is below it.
- **L1 cannot be read as a like-for-like test**, and that is a finding of this run: random zones of the same count and radii cost far more memory than the query's own - 5.72 bits against 4.28 at focus area 0.5 - because a query's zones overlap each other while random ones are scattered over the map. Behind the D4 and D2 floors the richer control wins on memory alone. Behind an empty floor own zones win anyway, in 6 of 10 cells for biology-math and 4 of 10 for history-geography, spending less: with nothing outside the zones, hitting the right blocks is what carries the answer.
- **L4 holds for both pairs**: the gap own minus random is far larger behind the ZERO floor than behind the D4 floor (+1.355 against -0.013 for biology-math, +0.463 against -0.029 for history-geography). The address shows sharpest when the rest of the network gives nothing.

## How much a cell moves, and where the address shows (exploratory)

The criterion of ADDENDUM-11 counts cells without asking how much each one changes. It should have:
the share of blocks a cell lifts over the floor runs from 0.4% to 55.7% across the grid, a hundredfold,
and the shuffle control can only relocate what was lifted. Behind a D4 floor:

| focus area | blocks lifted | own minus no-mask, biology-math (fs 0.5 / 1.0) |
| --- | --- | --- |
| 0.2 | 0.4% | +0.002 / +0.002 |
| 0.333 | 1.3% | +0.003 / +0.002 |
| 0.5 | 4.3% | +0.006 / +0.009 |
| **0.667** | **16.3%** | **+0.045 [+0.005, +0.086] / +0.047 [+0.007, +0.088]** |
| 0.8 | 55.7% | +0.018 / +0.023 |

The address shows where the layout moves enough of the net to matter and not so much that place stops
mattering. Below 5% lifted the control is nearly the layout itself - at focus area 0.5 the shuffle
changes the level of 6% of the blocks, at 0.2 it changes 0.7%, so there is nothing to detect. Above
half the net lifted, every block that matters is deep already, whoever placed it.

For history-geography no cell shows it at any strength, which fits its scattered masks ([E011](../E011-depth-caps/results.md)).

So L3 fails by the letter of its criterion and holds in the two cells that can test it, for the pair
that has an address at all. The lesson is about the design, not the data: a grid whose cells differ
a hundredfold in what they do needs a criterion that says which cells count, fixed before the run.

## What the numbers say

- **Against uniform quantization at the same memory** (the uniform curve interpolated at each layout's bits), the D4 floor with lenses is +0.002 ... +0.018 - at or just above the curve, within the noise of E009. The D2 and ZERO floors are far below it: -0.1 ... -3.4. Uncalibrated low bits cost more than a mask can win back.
- **An empty floor is better than a broken one.** ZERO alone reads -1.386, D2 alone -2.750: a block that is not read at all costs less than a block read as noise. Two bits without calibration are worse than nothing - the sharpest argument yet for the calibrated first slice ([d2-glass](../E006-read-depths/results.md)).
- **Lenses are worth most in the middle.** At the D4 floor, focus area 0.67 and strength 1 reads -1.072 at 4.91 bits with accuracy 0.539, against -1.103 at the same memory with no mask and 0.501 accuracy. At focus area 0.8 the layout costs 6.65 bits and reaches -1.005, accuracy 0.539 - bf16 quality on 42% of its memory, but so is uniform D6 at 6 bits (-1.014, accuracy 0.522).
- **Small lenses change nothing.** At focus area 0.2 every layout costs 4.00 bits and reads the floor: the zones are too small to lift a block a whole rung.
- **Memory is not equal across layouts.** Own zones cost a little more than the paired topic's (6.65 against 6.59 bits at focus area 0.8) because a query's own zones are wider. Only own against no-mask is an exact pair by memory - which is why L3 is the honest test of the address and L2 is not.

## What it means

The lens layout works as a mechanism: memory follows the zones, the invariants hold, and a broken floor is carried back by the lenses. The address is weaker than the mechanism. It beats a fixed set of zones for every question (L6, both pairs) but not the same memory placed without a mask (L3), and it beats the paired topic only for far-apart topics where bits are scarce (L2, biology-math at D2).

The random-zone run (`runs/E010-lens-layout-random/`) adds one lesson about controls: matching the count and the radii of zones does not match their memory once zones overlap. The preregistration's random control is honest only where a budget is fixed, as in E008 and E009. Under a layout whose memory is a result, the exact control is the shuffle of the layout's own levels - L3.

Read together with E009: gradient zones find what matters for a question, and now they also beat generic importance - but a lens layout still does not buy quality per bit over uniform quantization on E2B. What would change that, in order: a calibrated first slice, so the floor can be D2 without the model breaking; the address taken online from the first layers instead of a full gradient pass; a sharper score for close pairs.

The floor of the comparison - random zones of the same count and radii - was run separately on the same commit (`84d1b9f`), and is read above under L1 and L4.

## Bench

Mask phase: GPU utilization 43% mean (median 38%), reserved peak 18.2 GiB. Evaluation: 70% mean (median 95%), peak 14.4 GiB; both within the 19.2 GiB of the 0.8 share. The whole grid of 30 cells took 26 minutes, the random-zone grid 15.
