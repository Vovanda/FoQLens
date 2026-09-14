# Addendum 12 - E011: when a depth cap saves storage

Fixed 2026-09-12, **before the code of the experiment and before the run**. Not edited after its commit.

## 1. Why

A layout is read per query, but storage is one copy. A block has to keep the deepest slice any query asks of it, so its cap is the maximum over the queries the model serves. [E006](../E006-read-depths/results.md) built that storage (`CappedSlicedWeight`, `set_caps`) and showed a block can keep only the depth it is read to; what was never measured is how much that saves once many different queries share one copy.

The probe that prompted this experiment, on the 395 questions of [E010](../E010-lens-layout/_index.md): behind a D4 floor at focus area 0.5 a question reads 4.28 bits per weight while the caps have to store 6.47; at focus area 0.8 the caps store 8.00 - every block is read at D8 by someone, and nothing is saved. Behind an empty floor a question reads 0.82 and storage still costs 6.07. Addressing precision by the query saves reading, not storage, unless the queries are few or alike - and that is what this experiment measures.

## 2. Given

- **The mechanism**: [docs/quantization-filter.md](../../docs/quantization-filter.md) at commit `c91af59` - the rules as written there. The floor stays D4 (calibrating a D2 floor was measured and dropped, see the reading notes at the same commit).
- **The layouts**: the graded zone layout of [E010 PREREG](../E010-lens-layout/PREREG.md) on the masks of the E010 run, floor D4 and ZERO, focus area 0.2 ... 0.8, focus strength 1.
- **The cap of a block**: the deepest level any question of the set asks of it. Reading is unchanged by construction - a cap is never below what a question reads - so quality is not measured again here.
- **Model and data**: Gemma 4 E2B at the pinned revision; the 395 questions of E010 (biology 95, math 100, history 100, geography 100). Held-out topics are not opened.
- **The cost**: mean bits per weight, weighted by block size, as in E010; the reference is the full sliced copy at 8 bits.

## 3. Measured

For every cell of floor x focus area:

- **storage against the number of questions** `k` = 1, 2, 5, 10, 20, 50, 100, 200, 395, each drawn at random from the set, 20 draws per `k`, mean and spread;
- **storage against the variety of the set**: `k` questions from one topic, from one pair, and from all four;
- **the reference layouts**: caps from the backbone's zones (one set of zones for every question) and caps from random zones of the same count and radii, both at the same `k`;
- **what a question reads** at the same cell, so the gap between reading and storing is visible.

## 4. Expected

- **S1 - storage grows with the number of questions and saturates.** At `k` = 1 the caps are the layout itself; by `k` = 395 they are close to the full copy. Expected: monotone in `k`, with the steepest rise in the first few dozen questions.
- **S2 - one topic is cheaper than four.** At the same `k`, questions from one topic need less storage than questions spread over four. Expected: holds at every `k` above 5, and the gap grows with `k`.
- **S3 - a cap saves at least 30% of the full copy only for a narrow profile.** Expected: 30% saving (storage at or below 5.6 bits behind a D4 floor) holds for one topic up to some `k`, and fails for four topics beyond a few dozen questions. The verdict names that `k` for each case.
- **S4 - the query's own zones store no less than the backbone's.** One fixed set of zones for every question caps fewer blocks deep, so it stores less; the question is how much quality that costs, which E010 already answered against the backbone (L6). Expected: backbone caps are cheaper, own caps read better - reported as a pair, not a winner.

No hypothesis of the preregistration is at stake here: this is an engineering measurement of what the storage of ADDENDUM-11 costs, and it carries no hypothesis id.

## 5. Criterion

- Everything is computed from the masks of one run on the CPU: the layouts of every question, their caps over a subset, and the weighted mean bits. No quality evaluation, so no bootstrap - the numbers are exact, not sampled, except for the draws over subsets, which are reported as mean and min-max over 20 draws.
- S1 and S2 hold when the ordering they state holds at every `k` measured, not on average.
- The run writes `runs/E011-depth-caps/e2b/summary.json` and `results.md` states, for each floor and focus area, the `k` at which storage passes 30% saving - or that it never does.
