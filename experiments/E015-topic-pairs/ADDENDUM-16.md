# Addendum 16 - E015: three topic pairs the bench has not used

Fixed 2026-09-13, **before the run**. Not edited after its commit.

## 1. Why

Two findings stand on the same four topics. [E013](../E013-regulator-map/results.md) says lenses beat uniform quantization at the same memory by 3.8 points of error rate. [E014](../E014-moved-zones/results.md) says the query's own zones beat the same figure carried elsewhere - by 0.02 nats, and not measurably on the error rate. Both were read on biology, math, history and geography, and the two pairs already disagree with each other: biology-math carries the effect, history-geography does not.

With one pair working and one not, "lenses help" is indistinguishable from "lenses help on biology-math". Volodya asked for more runs on other topics. This is that.

E014 additionally owes a confirming run under a fixed prediction, since its addendum was written after its run. That debt is paid here: the same control on new topics, predicted in advance.

## 2. Given

- **The mechanism**: [docs/lens.md](../../docs/lens.md); sum of lifts, the default profile, floor D4.
- **Model**: Gemma 4 E2B at the pinned revision.
- **Topics**: chemistry (99 questions), physics (97), biology (95), math (100) - the files already in `prompts/`. The held-out topics stay closed.
- **Pairs**: chemistry-physics, biology-chemistry, math-physics. The first two are neighbouring sciences, the third puts calculation against a science.
- **The grid**: floor D4 x focus area 0.70, 0.75, 0.80 x focus strength 1 - the working range E013 found, its best cell in the middle.
- **Layouts per cell**: the query's own zones and the same figure carried elsewhere (`moved_zones`, 40 landings matched by covered weight). References once: bf16, uniform D4, D6, D8, bare floor D4.

## 3. The measure

- **Error rate** - the share of questions whose highest-scoring letter is wrong; **log-probability** of the right letter as the second number.
- **Cost** - mean bits per weight read, weighted by block size.
- Comparisons by paired bootstrap over the questions of the pair (10 000 resamples, seed 0), 95% interval. The own-against-moved comparison is **pooled over the three cells of a pair** before the bootstrap, as in E014: three cells give three chances for one interval to land.

## 4. Expected

- **P1 - lenses hold bf16 quality on every pair.** At floor D4, size 0.75, strength 1 the error rate is within 2 points of bf16 on all three pairs, at 5.5 to 6.5 bits.
- **P2 - lenses beat uniform quantization at the same memory on at least two of the three pairs**, and on at least one the 95% interval clears zero. This is the E013 claim carried to new topics; if it fails on two pairs of three, E013's result is about its topics and not about the mechanism.
- **P3 - the address beats the carried figure on the log-probability, pooled over the three pairs**, with the interval clear of zero; on the error rate the difference is positive but the interval need not clear zero. This is the E014 result predicted in advance.
- **P4 - the gain is smallest on math-physics.** E013 found the lenses win where the answer is knowledge held in a place of the network and lose where it is calculation (math 65% error against bf16's 68%). If math-physics shows the largest gain of the three, the reading of E013 is wrong.

## 5. Criterion

- P1 holds if the difference to bf16 is within 2 points on all three pairs at the named cell.
- P2 holds on a pair if the point estimate beats uniform quantization interpolated at the same bits; the interval is reported per pair.
- P3 holds if the pooled log-probability interval over the three pairs clears zero.
- P4 is read on the point estimates of the three pairs at the named cell.
- The run writes `runs/E015-topic-pairs/e2b/summary.json`; `results.md` gives the table per pair and the verdict against P1-P4.
