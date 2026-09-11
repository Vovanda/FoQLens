# Preregistration addendum 02 - the second instrument: gradient × activation per block

Fixed 2026-09-11, **before any run of this instrument**. It follows the [preregistration](PREREGISTRATION.ru.md), [addendum 01](ADDENDUM-01.md) and [thresholds 01](THRESHOLDS-01.md), and changes none of their predictions. Not edited after its commit.

## 1. Why a second instrument

The naive per-block score failed the preregistered confirmation on E2B (`8214822`): history–geography p 0.85. The preregistration fixes the next move for that case: change the instrument by increasing cost - the output gradient per block first, block ablation after it.

## 2. The instrument

For one query, in bf16 with eager attention and every controlled module at bf16:

- **L** - the model's own next-token cross-entropy on the query text (labels = the query tokens). No answer is needed.
- **y** - the output of a controlled linear module, `[tokens, rows]`; **g** = ∂L/∂y.
- **Block score** = | Σ over tokens except `<bos>` and over the block's rows of g · y |.

This is the first-order Taylor estimate of how much the loss on this query would change if the block's output were zeroed - a gradient-based precursor of block ablation. Unlike the naive score it measures what the block contributes to this particular text, not how large its output is.

Implementation: one forward and one backward pass; model parameters are frozen, the graph runs through activations only (the embedding output is made to require gradients - `inputs_embeds` is not used, because Gemma 4 then computes its per-layer inputs differently and the model would change).

## 3. How it is read

- **One variant, fixed now:** masks with the background subtracted (the mean mask over all queries of the run). Run 1 showed that raw masks are dominated by a profile common to every query; raw masks are still reported, but not read against the criteria.
- **Exploration**: the same 421 debugging queries as run 1, blind, sealed with the sha256 of the raw vectors before opening. Criterion: the preregistered step 1 direction, cos_in(A) > cos(A, B) on all 10 pairs.
- **Confirmation**: the criteria of THRESHOLDS-01 sections 2-3 unchanged - history–geography passes the cosine direction and the label-permutation test (p < 0.05, 1000 permutations, seed 0); step 2 direction with math; E2B, then E4B, both must pass.

## 4. The held-out pair and the bar

History (MMLU `prehistory`) and geography stay the held-out pair - the author's decision of 2026-09-11. They are known to be hard: in the model's own representations they barely separate (middle layer ARI 0.04, diagnostic `c9f8cb2`), though the cosine direction holds there. The bar this sets, stated before the run: **an address has to keep the separation that the model's representations have**. An address that cannot tell hard domains apart where the model can is of little use.

## 5. What follows

- Confirmed on both models → step 3 (quality against budget, two baselines) with this instrument.
- Not confirmed → the untrained address is not supported with gradient scores either; the preregistration's next instrument is block ablation. The learned score (step 4) remains outside solo work.
- Step 2+ is not part of this pass: the mixed domain is rebuilt first.
