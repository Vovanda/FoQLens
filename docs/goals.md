# Goals

FoQLens goals in order of execution. Each next goal opens only if the previous one passed. Details, measurements and reasoning are in the [plan](plan.md). Updated 2026-09-11.

## Main goal

Show that precision allocated over the weights by the meaning of the query gives better quality at the same mean bit budget than uniform quantization, and than a random mask of the same concentration.

## Goals by step

**Step −1. Preregistration in git** - status: done.
Done when: before the first run the repo holds the seven properties of the expected topology, direction predictions for every test, the run code, and the list of held-out topics.

**Step 0. Topics separate in representations** - done 2026-09-11 on E2B: **yes**.
Done when: mid-layer activations on ~10 biology and ~10 math questions cluster. If not → the next model.
Result (95 biology, 100 math questions, [`runs/step0/e2b/summary.json`](../runs/step0/e2b/summary.json)): at the middle layer cos_in 0.687 / 0.644 vs cos_between 0.566 (centered +0.148 / +0.111 vs −0.136), silhouette +0.23, k-means ARI 0.98. Separation holds at every layer and peaks at layers 16-20; it already shows at the embeddings (ARI 0.71), so part of it is lexical.

**Step 1. Masks are separable and concentrated** - **not confirmed with the naive score** (2026-09-11): exploration passed weakly, the E2B confirmation on held-out domains failed (history–geography p 0.85). Next: instrument change (output gradient per block). The main kill switch.
Done when: mask cosine within a topic is strictly greater than between topics. If not → subtract the background, then change the score (gradient → block ablation).
Result ([results-run1.md](results-run1.md)): all 10 pairs separate only in the pooled mode with the background subtracted; the preregistered norm mode gives 3/10 raw, 7/10 subtracted. Separation is in the means, single masks barely cluster (ARI 0.17 vs 0.98 for representations). Bets: Claude's side by the preregistered rule.

**Step 2. Zones overlap** - exploration passed (biology–chemistry −0.007 between biology–math −0.377 and within biology +0.353), **E2B confirmation failed** (history–geography +0.199 above within history +0.157). Unread until an instrument passes step 1.
Done when: related topics (biology–chemistry) overlap more than unrelated ones (biology–math).

**Step 2+. Mask geometry** - exploration done: mixed; the hand-written biophysics set came out biology-like in both representations and masks, so a better mixed domain is needed before reading it. On step 1 data.
Additivity, two bundles or a blob, junction zone, isthmus, isthmus ablation, reverse ablation, linearity of representation → mask.

**Step 3. Quality against budget** - in progress. Generic importance beats random allocation by a wide margin; background-subtracted topic masks, alone ([injection](results-injection.md)) or on top of the importance backbone ([backbone](results-backbone.md)), do not beat the other topic's mask or random blocks where the model still works. Shaped as expert zones from the gradient mask ([zones](results-zones.md)), the own topic beats random zones, the paired topic and no mask at the same budget for a polar pair (biology-math) - not yet for a close one (history-geography).
Done when: the quality-vs-mean-bits curve lies above **both** baselines - uniform and random mask.

**Step 4. Learned score** - beyond solo work. Only if the naive score gave an effect; needs co-authors or a group.

**Step 5. Residuals instead of copies** - engineering, not a test of the hypothesis. Status: first version done ([results](results-residual.md)) - one sliced copy read at 2 / 4 / 6 / 8 bits, the bf16 weights can leave the GPU (-1.63 GiB on E2B, D8 as good as int8).

## Boundaries

- Hands-on up to step 3 inclusive, about two weeks of dense work.
- No memory savings are claimed: with three copies there are none.
- A paper after step 1, not before.
- Blind analysis: all runs at once, opened afterwards. The exception is step 0.
