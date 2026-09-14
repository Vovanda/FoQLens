# Addendum 11 - E010: the lens layout on E2B

Fixed 2026-09-12, **before the code of the lens layout and before the run**. Not edited after its commit.

## 1. Why

ADDENDUM-07 to 10 fixed the mean bits in advance and stretched the zones to hit them, so even at focus area 0 precision was pushed into the zones ([zones.md](../../docs/zones.md), legacy). That is not the picture of the project. Here the whole network sits behind a glass, lenses are inserted into the query's expert zones, and memory is the result of the lenses, not a budget.

From the legacy runs: the gradient zones find what matters for a question (E009 M1 on both pairs) and carry a topic address for far topics (M2 biology-math); the address is worth most where bits are scarce; a falloff around the centers beats a plateau ([E009 results](../E009-zones-matrix/results.md)).

## 2. Given

- **The mechanism**: [docs/quantization-filter.md](../../docs/quantization-filter.md) at commit `4fa3e7f` - the controls and rules 1-7 as written there. This addendum does not restate them; a later change of lens.md does not change E010.
- **The configuration of this run** - values, not rules: the E2B ladder ZERO, D2, D4, D6, D8 from the sliced copy of E006; the attention level D4 (uncalibrated D2 diverges on E2B, [E006](../E006-read-depths/results.md), so D2 is only a glass and the ring pushed past a lens edge); the sum strategy; the default profile.
- **Model and data**: Gemma 4 E2B at the revision pinned in `foqlens.model.REVISIONS`; the 395 questions of E009 (biology 95, math 100, history 100, geography 100); pairs biology-math and history-geography. Held-out topics are not opened.
- **Zones**: the weight map (co-activation of the raw gradient masks of all questions, no labels) and the zones of each question as in ADDENDUM-07; the gradient source only (pooled zones failed in ADDENDUM-07 and E009), on the deterministic gradient pass of `7a06b67`.
- **Grid**: glass D4, D2, ZERO x focus_area 0.2, 0.333, 0.5, 0.667, 0.8 (radius factor 1/4, 1/2, 1, 2, 4) x focus_strength 0.5 and 1 - 30 cells. focus_strength 0 is the glass alone, a reference. GPU share 0.8.

## 3. Compared

In every cell, at the same glass, focus_area and focus_strength:

- the lenses of the query's own topic;
- the lenses of the paired topic;
- the lenses of the backbone - the zones of the mean mask over all questions, one set for every question;
- no mask at the same memory - the mean bits of the own-lens layout of that question on the same ladder above the same glass, levels given to blocks at random (the no-mask layout of ADDENDUM-09).

References, once: bf16; uniform D4, D6, D8; each glass alone.

Order: references first, then the cells from glass D4, focus_area 0.5, strength 1 outwards, then D2, then ZERO; the summary is written after every cell. Memory (mean bits) is reported for every layout.

**Random lenses** - the same count and radii as the own lenses, centered on random blocks of the map (`zones.random_zones`, seed 0 per question) - are a separate run at any later time, on the commit and the batches of the main run. They are a floor, not the main test: a meaningful mask is expected to beat noise (even the generic importance of E005 beats random blocks).

## 4. Expected

- **L2 - the address** (H3.1). Own lenses beat the paired topic's lenses. Expected: holds for biology-math; open for history-geography, where M2 failed.
- **L3 - lenses beat no mask** (H3). Own lenses beat the same memory without a mask. Expected: holds for biology-math.
- **L5 - lenses carry a broken glass.** Behind the D2 glass own lenses beat the D2 glass alone. Expected: holds for both pairs if the zones hold what the answers need.
- **L6 - the query, not generic importance** (H3). Own lenses beat the backbone's lenses. Expected: holds for biology-math; open for history-geography.
- **L1 - the lenses find the topic** (H3.2), from the random-lens run. Own lenses beat random lenses. Expected: holds for both pairs, as M1 in E009.
- **L4 - an empty glass shows the address sharper**, from the random-lens run. The gap own minus random is larger behind the ZERO glass than behind the D4 glass, at focus_strength 1, where the ceiling is D8 behind both. Holds for a pair when the ZERO glass has more of its 5 cells above zero. Expected: holds for biology-math.
- **H4, the single-pass part - exploratory.** Own lenses against bf16 in every cell, reported with intervals. No cell above bf16 is expected: in E009 no layout was. A cell above bf16 would be the first sign of H4 and is to be replicated on a fresh draw before it is claimed.
- **H5**, the MoE end of the scale, is not tested here: it needs lenses that work first.

## 5. Criterion

- Comparison: paired bootstrap over questions (10 000 resamples, seed 0) of the right-letter log-probability, 95% interval; a cell is one pair x glass x focus_area x focus_strength.
- A prediction **holds for a pair and a glass** when at least 3 of its 10 cells have the interval above zero and none below. At 2.5% per cell and independent cells that is below 0.2% by chance; the cells share questions, so the true rate is higher.
- Before the run, the invariants of the rules of lens.md are covered by tests: focus_area 0 or focus_strength 0 gives the glass everywhere; a single lens reads its stepped profile exactly; an overlap lowers no block, and neighbouring blocks differ by at most one rung unless the profile skips one; the mean bits grow with focus_area and focus_strength; a malformed profile is refused where it enters.
