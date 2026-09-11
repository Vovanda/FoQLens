# Goals

FoQLens goals in order of execution. Each next goal opens only if the previous one passed. Details, measurements and reasoning are in the [plan](plan.md). Updated 2026-09-11.

## Main goal

Show that precision allocated over the weights by the meaning of the query gives better quality at the same mean bit budget than uniform quantization, and than a random mask of the same concentration.

## Goals by step

**Step −1. Preregistration in git** - status: done.
Done when: before the first run the repo holds the seven properties of the expected topology, direction predictions for every test, the run code, and the list of held-out topics.

**Step 0. Topics separate in representations** - not started.
Done when: mid-layer activations on ~10 biology and ~10 math questions cluster. If not → the next model.

**Step 1. Masks are separable and concentrated** - not started. The main kill switch.
Done when: mask cosine within a topic is strictly greater than between topics. If not → subtract the background, then change the score (gradient → block ablation).

**Step 2. Zones overlap** - not started.
Done when: related topics (biology–chemistry) overlap more than unrelated ones (biology–math).

**Step 2+. Mask geometry** - not started. On step 1 data.
Additivity, two bundles or a blob, junction zone, isthmus, isthmus ablation, reverse ablation, linearity of representation → mask.

**Step 3. Quality against budget** - not started.
Done when: the quality-vs-mean-bits curve lies above **both** baselines - uniform and random mask.

**Step 4. Learned score** - beyond solo work. Only if the naive score gave an effect; needs co-authors or a group.

**Step 5. Residuals instead of copies** - engineering, not a test of the hypothesis.

## Boundaries

- Hands-on up to step 3 inclusive, about two weeks of dense work.
- No memory savings are claimed: with three copies there are none.
- A paper after step 1, not before.
- Blind analysis: all runs at once, opened afterwards. The exception is step 0.
