# Results - run 1, exploration (Gemma 4 E2B)

Run 1 exploration on the debugging domains, read against the [preregistration](../prereg/PREREGISTRATION.ru.md) and [addendum 01](../prereg/ADDENDUM-01.md). Opened 2026-09-11, all at once, after the step 1 summary (`c407c45`) and the step 2+ summary (`9e008e3`) had been committed unopened. Data: [docs/data-sources.md](data-sources.md) - biology 95, math 100, chemistry 99, physics 97, biophysics 30 questions.

**This is exploration, not a result.** Three center modes were tried; by addendum 01 one of them is now chosen and only that one is tested on the held-out domains and E4B.

## Summary

| Step | Preregistered direction | Exploration | Verdict |
| --- | --- | --- | --- |
| 0 | topics separate in representations | ARI 0.98 at the middle layer | **pass** - E2B is fit |
| 1 separability (load-bearing) | cos_in(A) > cos(A,B) on all pairs | 10/10 pairs only in mode B (pooled) with the background subtracted | **pass in exploration, weak** |
| 1 concentration (load-bearing) | mask not flat | raw Gini ≈ 0.6, entropy ≈ 0.91, but the same shape for every topic | **pass formally**, concentration is shared, not topical |
| 2 | bio–chem between bio–math and within bio | -0.377 < -0.007 < 0.353 | **pass** |
| 2+ | seven geometry tests | see below | mixed; the mixed domain turned out biology-like |
| bets on step 1 | resolved on mode A by background subtraction | A: 3/10 pairs raw → 7/10 subtracted | **Claude's side** by the preregistered rule |

## Step 1 - are masks separable and concentrated

Pairs where the preregistered cosine direction holds (cos_in of both domains above their cross cosine), out of 10:

| Mode | Raw | Background subtracted | ARI, all domains (raw / subtracted) | Silhouette (raw / subtracted) |
| --- | --- | --- | --- | --- |
| A - norm (preregistered) | 3 | 7 | 0.02 / 0.03 | −0.09 / −0.07 |
| B - pooled | 7 | **10** | 0.10 / 0.17 | −0.03 / −0.04 |
| C - attention | 6 | 8 | 0.04 / 0.03 | −0.18 / −0.11 |

- **Raw masks are almost the same for every query** (cosine 0.96-0.99 everywhere, as the calibration already showed for unrelated texts). A common per-block magnitude profile dominates - the "large blocks light up" failure the plan warned about.
- **With the background subtracted, mode B separates every pair**: cos_in biology +0.353, math +0.414, biophysics +0.344, physics +0.126, chemistry +0.008, against a cross cosine of −0.063 over all domains. Chemistry is barely coherent inside itself.
- **The separation lives in the means, not in single masks**: k-means recovers the domains with ARI 0.17 and the silhouette is negative, while the representations of the very same questions give ARI 0.98 (step 0). The topic is in the model; the naive score carries only a small part of it.
- Concentration: median Gini of raw masks 0.58-0.64, normalized entropy 0.90-0.92 (flat would be 0 and 1). The masks are concentrated, but since raw masks are nearly identical across topics, that concentration is the model's common shape, not a topical zone.

**Bets.** Resolved on mode A, as addendum 01 fixes: separability went from 3/10 pairs raw to 7/10 with the background subtracted - a sharp improvement, so by the preregistered rule the raw score caught mostly background: Claude's side. Vladimir's claim of a substantial quality gain is a step 3 claim and stays open.

## Step 2 - do zones overlap

Mode B, background subtracted: cos(biology, chemistry) −0.007 lies between cos(biology, math) −0.377 and cos_in(biology) +0.353 - **the preregistered direction holds**. It also holds in modes A and C with the background subtracted; it fails in mode A raw.

## Step 2+ - mask geometry (mode B, background subtracted)

Raw masks are left out here: they are nearly identical, and every geometry test on them is trivially "additive" (R² 0.998).

| Test | Prediction | Exploration (top fractions 1% / 5% / 10%) | Reading |
| --- | --- | --- | --- |
| Additivity | not additive | R² 0.572; coefficients biology +0.55, physics **−0.53** | partly additive; the biophysics mask looks like biology, not biology + physics |
| Junction zone | a third group beyond chance | 0.28 / 0.24 / 0.22 of biophysics top blocks outside both component tops, against 0.98 / 0.90 / 0.81 by chance | **fails**: biophysics top blocks sit inside the component tops |
| Isthmus | exists | 0.112 / 0.106 / 0.103 vs chance 0.110 / 0.102 / 0.093 | inconclusive: at chance level at 1%, slightly above at 5-10%, no significance test |
| Support shape | concentration like biology, wider mass | Gini / participation: biology 0.955 / 0.027, physics 0.841 / 0.058, biophysics 0.896 / 0.050 | direction holds formally (mass ~1.9× wider) |
| Hierarchy | excess at both ends | top in all 4 science domains: 5 / 119 / 139 vs ~0 / 0.1 / 1.5; top in one only: 192 / 754 / 1957 vs 571 / 2522 / 4289 | **half**: a strong shared foundation, but fewer domain-specific blocks than chance |
| Linearity | mask R² clearly below representation R² | representation R² 0.370, mask R² 0.572 | **fails** |
| Isthmus ablation, reverse ablation | - | need quality measurement (step 3) | not run |
| Uneven zone size (property 7) | no procedure preregistered | - | not run |

**The mixed domain did not come out mixed.** In representation space biophysics is close to biology (cos +0.59) and away from physics (−0.22); in mask space it is the same (+0.71 vs +0.01). The high-school physics questions are mostly calculation problems, and the hand-written biophysics questions lean on biology vocabulary. The step 2+ tests assume a query that is half one domain and half the other; this set is not that, so their failures say more about the probe than about the topology. A better mixed set is needed before step 2+ is read seriously.

## Confirmation - E2B, held-out domains: **not confirmed**

Run under [THRESHOLDS-01](../prereg/THRESHOLDS-01.md) (committed in `aed8a29` before the run): mode B, background subtracted, history (MMLU `prehistory`) 100, geography 100, math 100 questions, 1000 label permutations.

| Claim | Criterion | E2B | Verdict |
| --- | --- | --- | --- |
| Step 1 | history–geography: cosine direction and permutation p < 0.05 | cos_in history +0.157, geography +0.282, cos_between +0.199; margin −0.042, p 0.85, ARI 0.00 | **fail** |
| Step 2 | cos(history, geography) between cos(history, math) and cos_in(history) | −0.298 < **+0.199** > +0.157 | **fail** |

By THRESHOLDS-01 a claim is confirmed only if it holds on both E2B and E4B, so neither can be confirmed any more. **The E4B pass is not run with this instrument**: it cannot change the verdict, and running it anyway would be a second try at the same claim.

A caveat, recorded but not used to rescue the result: prehistory partly overlaps with geography (migrations, sites, regions) and is loosely coherent inside itself (cos_in +0.157). The preregistration does not allow "it failed, so the domains were wrong" for a load-bearing claim.

**What the run 1 instrument shows, in the end:** the naive per-block score separates the debugging domains in their means after background subtraction, but that does not carry over to held-out domains. The step 1 claim is not supported with this instrument. Nothing is concluded about the topology: by the preregistration, its predictions stay unread until an instrument passes.

## Diagnostic - do the held-out domains separate in representations? (exploratory)

Not part of any preregistered test; run after the confirmation verdict was committed (`8214822`), with the step 0 script on history and geography ([runs/diagnostic/e2b/summary.json](../runs/diagnostic/e2b/summary.json)). It does not change the verdict.

**The model itself barely tells these two domains apart.** At the middle layer, centered: cos_in +0.040 / +0.061 against cos_between −0.058, silhouette +0.10, **ARI 0.04** - against ARI 0.98 for biology–math in step 0. No layer gets above ARI 0.06 centered.

What it means:

- The confirmation was hard by construction: masks were asked to separate a pair that the representations separate only faintly. The preregistration required held-out domains, but not that they be checked for being distinguishable first.
- The instrument is still not off the hook: in representations the cosine direction holds (weakly), in masks it does not - the masks lost even that faint signal.
- For the next instrument, the held-out pair must pass a step 0 check on representations before it is fixed; and history and geography are no longer untouched, since this pair is now known to be hard.

## What is next

1. **Instrument change**, as the preregistration fixes: the output gradient per block, then block ablation - each through the same path of addendum, exploration, thresholds and confirmation.
2. **A new held-out set** for that confirmation, checked for separability in representations before it is fixed.
3. **A better mixed domain** for step 2+: questions that need both components in equal measure, checked in representation space first.
