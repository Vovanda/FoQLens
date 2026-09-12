# Clarification 1 to ADDENDUM-16 - what P2 now means

Written 2026-09-13, still **before the run**. [ADDENDUM-16](ADDENDUM-16.md) is not edited; this file sits beside it.

Between fixing ADDENDUM-16 and running E015, the shuffle check landed ([E013 results](../E013-regulator-map/results.md#what-the-shuffle-did-to-this-page)). It withdrew the claim P2 was built on: over six orders of the answer options E013's +3.8 points against uniform quantization fall to +1.05 with the interval through zero.

**P2 as written stays, and it is now a harder prediction than it was.** It said: the lenses beat uniform quantization at the same memory on at least two of the three pairs, and on at least one the interval clears zero. That was phrased as "the E013 claim carried to new topics"; since the E013 claim no longer stands on its own topics, P2 is no longer expected to hold, and the honest expectation before the run is:

- **P2 is expected to fail.** If it holds on new topics anyway, that is evidence the mechanism does something the original order of the options was masking, not evidence that E013 was right.

Two additions to the run, so E015 is not measured on one order of the options either:

- **Every pair is run on the original order of its options, as ADDENDUM-16 says, and the cell that comes out best is then rerun through `scripts/shuffle_check.py` on six orders.** A pair counts toward P2 only on the shuffled numbers.
- **The address is read off the question alone, without the options, in a second pass** - the shuffle check found that recomputing the mask on a reordered prompt costs 2 points of accuracy, so part of the address is the surface form. This pass tests whether reading the question alone removes that dependence. It is exploratory and reported separately; it does not enter the verdict on P1-P4.

Nothing else in ADDENDUM-16 changes: the topics, the grid, the measure and P1, P3 and P4 stand as fixed.
