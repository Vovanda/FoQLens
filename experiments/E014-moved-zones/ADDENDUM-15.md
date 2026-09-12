# Addendum 15 - E014: the query's own zones, carried elsewhere on the map

Written 2026-09-13, **after the run**. This is stated first because it matters: E014 is exploratory, not preregistered. The bench's rule is preregistration before the run, and it was not followed here - the control was built and run in one sitting while [E013](../E013-regulator-map/results.md) was still open. Nothing below was edited to fit the numbers, but the reader has only my word for that, which is exactly what a preregistration exists to replace. A confirming run under a fixed prediction is owed, and it is named in section 6.

## 1. Why

Every result so far compares a layout against uniform quantization. That answers "does addressing pay", not "does the address pay": a layout with zones of the right shape might do as well anywhere on the map, and then what works is the shape, not where it points.

The control E010 used - zones of the same count and radii around random blocks - is not honest about cost. Random zones overlap less, so they store and read more: 5.72 bits against 4.28 at focus area 0.5 ([E011](../E011-depth-caps/results.md)). A control that spends a third more and answers worse says nothing.

Volodya set the condition for taking the comparison up again: do it **when we can choose zones at random honestly**. This is that attempt.

## 2. The control

The query's own zones, **carried elsewhere as one rigid figure**: the count, the radii and every distance between the centers are kept exactly; the figure is turned about the center of the map and shifted. Only the place changes.

Blocks do not lie evenly on the map, so the same figure in a new place can cover more or less weight. The landing is therefore chosen out of `PLACES_TRIED = 40` candidates by how closely the covered weight matches what the figure covered at home (`foqlens.zones.moved_zones`). Without that matching the control came out 18% dearer than the query's own zones.

Rejected: fitting the radii to the memory budget (breaks the figure), and drawing centers from the distribution of distances (approximate and fiddly).

## 3. Given

- **The mechanism**: [docs/lens.md](../../docs/lens.md); sum of lifts, the default profile, floor D4.
- **Model and data**: Gemma 4 E2B at the pinned revision, the 395 questions of E010 (biology 95, math 100, history 100, geography 100).
- **The grid**: the working range E013 found - floor D4, focus area 0.70, 0.75, 0.80, focus strength 0.5 and 1. Six cells, own and moved in each.
- **Reported per cell**: error rate, right-letter log-probability, mean bits, and the ratio of the memory the moved figure spends to the memory its own spends.

## 4. The measure

- **Accuracy** - the share of questions whose highest-scoring letter is right; and the **log-probability** of the right letter, which is what E008-E012 compared on. How both are computed and what they can be trusted for: `.claude/session-context/metric-trust.md`.
- **The comparison is pooled over the whole grid**, not read off the best cell. Per question, the six cells are averaged for own and for moved, and the pair is compared by paired bootstrap over the questions (10 000 resamples, seed 0). Twelve cells give twelve chances for one interval to clear zero; the pooled number does not.
- **The cost check**: the moved figure must not spend less. If it spends more, the difference is named and the comparison is read against it.

## 5. What it takes to say the address pays

The address pays if, pooled over the grid, the query's own zones beat the moved figure with the 95% interval clear of zero, at memory that is not higher for the own zones.

## 6. What is owed

E014 was run before this text existed, so it can support a claim only as an exploratory result. A confirming run is preregistered separately: the same control, on topic pairs the bench has not used ([E015](../E015-topic-pairs/), chemistry-physics, biology-chemistry, math-physics), with the prediction fixed before the run.
