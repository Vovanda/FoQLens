# Addendum 13 - E012: where the address is sharpest and where it dies

Fixed 2026-09-12, **before the code of the experiment and before the run**. Not edited after its commit.

## 1. Why

[E010](../E010-lens-layout/results.md) measured a grid whose cells differ a hundredfold in how much of the net they move, and read the address only after the fact: it shows at 16% of the net lifted and disappears at 0.4% and at 56%. The strongest signal in the project came from the empty floor, where own zones beat random zones of the same count and radii by +1.355 on average for biology-math, in 6 cells of 10.

This experiment goes straight at that edge instead of stumbling on it. Behind an empty floor the model has nothing but what the lenses open, so hitting the right blocks is the whole answer - and the question is only where that is sharpest, and where it stops mattering.

One thing to keep in mind while reading: behind an empty floor a model can be **worse than silent**. The bare floor reads -1.386, which is a uniform guess over four letters; own lenses at focus area 0.5 read -3.838 - a net with holes answers confidently and wrongly. So the measure here is not quality against a guess but the gap between own zones and the controls.

## 2. Given

- **The mechanism**: [docs/lens.md](../../docs/lens.md) at commit `c91af59`, floor ZERO, the default profile with the lowest non-zero rung pushed past the lens edge, sum of lifts, attention level D4.
- **Model and data**: Gemma 4 E2B at the pinned revision; the 395 questions of E010; pairs biology-math and history-geography. Held-out topics are not opened.
- **The scan**: focus area 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 at focus strength 1 - ten points from "almost nothing is open" to "almost everything is".
- **Reported with every cell**: the share of blocks lifted over the floor and the mean bits, so a reader sees how much each point moves and what it costs.

## 3. Compared

In every cell:

- the query's own zones;
- **random zones** of the same count and radii around random blocks (`zones.random_zones`, seed 0) - the control this experiment is about;
- the query's own levels shuffled inside groups of equal block weight - the same memory, no mask;
- the paired topic's zones.

References, once: bf16, uniform D4, and the bare floor.

## 4. Expected

- **A1 - the address is sharpest where little is open.** The gap own minus random, in log-probability, is largest at the small end of the scan and falls as focus area grows. Expected: the maximum sits below focus area 0.5, and the gap at 0.9 is at most a tenth of the maximum.
- **A2 - the gap dies when nearly everything is open.** At focus area 0.9 and above, the interval of own minus random contains zero for both pairs.
- **A3 - random zones cost more memory at every point.** They overlap less, so they store and read more ([E011](../E011-depth-caps/results.md)); the gap in bits is reported at every cell, and a win by own zones while spending less is the strong form of the result.
- **A4 - the shuffle control behaves the other way.** Own minus no-mask is small where little is open, since the shuffle can only move what was lifted, and grows with focus area up to the middle of the scan.
- **A5 - close topics stay flat.** For history-geography no cell shows a gap above zero for own minus random at any focus area, as in E010.

A prediction holds when it holds at every cell it names, not on average. There are no counted thresholds here: the scan is read as a curve, and the verdict states where the maximum sits and where the interval first contains zero.

## 5. Criterion

- Comparison: paired bootstrap over the questions of a pair (10 000 resamples, seed 0) of the right-letter log-probability, 95% interval, as in E010.
- The curve is read on the point estimates; a cell counts as showing the address when its interval is clear of zero.
- The run writes `runs/E012-address-edges/e2b/summary.json`; `results.md` names the focus area of the maximum gap, the focus area where the interval first contains zero, and the memory of own against random at both.
