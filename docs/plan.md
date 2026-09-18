# Step-by-step plan and preregistration

Assembled 2026-09-10, moved into FoQLens 2026-09-11. What exactly to do for the [problem statement](problem-statement.md), in order of execution. Each step gates the next: if it fails, do not go further. The steps are numbered as in the preregistration; the roadmap in the [goals](goals.md) has its own numbering and says where each step stands.

## Stack and model

- **Gemma 4, dense variants: 2.3B (E2B) for debugging, 4.5B (E4B) for confirmation.** Apache 2.0. The model is familiar from experience.
- **Do not take 26B A4B** - it is MoE. The split there is already done by the router; you would measure the router, not emerging zones.
- **transformers + bitsandbytes.** Not ollama and not llama.cpp - they hide logits and activations.
- **lm-evaluation-harness (EleutherAI)** as the benchmark runner: MMLU, ARC, HellaSwag, GSM8K in one command, numbers comparable to published ones, no baseline to compute.
- Hardware: RTX 3090 Ti 24 GB - with headroom.

**Model quality is secondary**: what is measured is the delta against the model itself, and the absolute level cancels out. The only constraint is that the model must not be so weak that representations do not separate by topic. Do not take 0.6B; 1.7–2.3B is fine.

---

## Step 0. Do topics separate in representations

**Ten minutes of work; without it there is no point going further.**

Take activations from a middle layer on ~10 biology and ~10 math questions. See whether they cluster.

- They separate → the model is fit, the score can be built on top.
- They do not → take the next size up. The scheme is not the problem.

## Step 1. Stability and separability of masks

**The main kill-switch experiment. Before any quantization.**

The simplest score, without training: take the heaviest tokens (by activation norm or by the share of attention on them) as **centers of expertise**, and expand them into a per-block score through which blocks respond more strongly on these tokens. Everything is computed on the first pass.

Measurement: 10 biology queries, 10 math queries. Cosine between score vectors.

- Masks are similar within a topic and diverge between topics → the centers are real, go further.
- Masks are roughly the same for all queries → the score catches something common to the model, not the topic. **The variant is killed within an hour.**

## Step 2. Overlapping zones

A direct test of the shared-foundation thesis.

Masks for related topics (biology and chemistry) should overlap more than for unrelated ones (biology and math).

If so, compactness through refusing to duplicate is confirmed as an effect, not as an argument.

## Step 3. Precision follows the meaning

Only if steps 1–2 passed. *Updated 2026-09-12: the layout is the zone layout over a floor ([quantization-filter.md](quantization-filter.md)).*

Pass scheme: the first N layers at base precision → a per-block score from the intermediate representation → the query's expert zones in the distance between blocks → each zone read more precisely, the rest of the weights at base precision (any rung of the ladder, down to empty). Until the online version exists, the mask comes from a full pass - an upper bound.

The weights are stored once, as a base with refinements read to a depth (step 5), so a zone costs only the depth it reads.

Measurement: quality against **memory**. The baseline is uniform quantization at the same memory (see «The baseline» below); the paired topic's zones and generic importance test the address. A random mask is not a baseline. Curve above the baseline → the scheme works.

## Step 4. Learned score

Only if the untrained one gave an effect.

A small matrix from the intermediate representation to a score vector, trained on the final quality through soft depth, rounded at inference.

Training the mask directly does not work - bit depth is discrete, no gradient flows through it. Hence the soft version.

A side effect in favor of the statement: with a continuous learnable mask **zones are not assigned but converge by themselves** - overlaps appear where they pay off.

## Step 5. Residuals instead of copies

An engineering optimization, not a test of the hypothesis. *Done 2026-09-11: 4 slices of 2 bits after MoBiQuant, read at 2 / 4 / 6 / 8 bits.*

A 2-bit base plus residual levels (the difference between the real weight and what the base layer gave). 2/4/6 bits from one data set, without three copies and without repacking. Sharpening stops costing separate memory.

A side benefit: no need to re-cut blocks by meaning - the layout stays aligned, and direction is expressed by **read depth**, not by the shape of the region.

---

## What to honestly expect

*Updated 2026-09-12.* Memory savings are real on the bench now: one sliced copy replaces the bf16 weights (-1.63 GiB on E2B) and every block can store only the depth it is read to. What is not shown at home is speed and energy: the slices are unpacked before the multiplication, and reading only the needed bits takes a kernel of its own.

What can honestly be shown at home:
- **memory** - the zones of a query cost only what they read;
- **quality against memory** - if precision is laid out by meaning, the same memory describes the model better;
- **compactness in parameter count** - through overlapping zones, not through bits per weight;
- **the shape of the degradation curve** - see experiment 2 on hardware inputs, value and degradation priority (author's private notes).

## The main open question

The block score: **how much this block is needed for this meaning**. The naive version through norms will almost certainly be too coarse - it is about the magnitude of influence, not specificity, and will light up large blocks instead of topic-related ones. Step 1 will tell.

It is the same hole as in the problem statement, but in a form you can probe: not "where is the region in weight space" but "how to compute a scalar per block". The scheme is ready; the question is open.

---

## Step 2+. Mask geometry - tests on the same data

Everything below is computed on step 1 data; there is almost no separate work. But exactly these tests turn "zones exist" into statements about how they are built and whether they are needed.

The main tool is **mixed queries**: biophysics against pure biology and pure physics.

### Mask additivity

Does the biophysics mask fit a linear combination of the biology and physics masks.

- **Additive** → zones behave like a basis, a mask for a complex query can be assembled from components, and overlap becomes literally an intersection of supports. The scheme is simpler and more predictable.
- **Not additive** → at the junction there are blocks that rise on neither component alone. The junction expert is an entity of its own, not a sum of parts. More likely, and stronger for the statement: additivity would mean topics are stored independently, while the whole point is that they are intertwined.

### Support shape: two bundles or a blurred blob

The question is not about values but about what the mask looks like geometrically.

Sort the scores in descending order and look at the concentration - the Gini coefficient or the entropy of the score distribution.

- **Two bundles** - the mass is in a small share of blocks, with a gap between the clusters. Zones are real as objects; the magnifier points at two places.
- **A flat blob** - the score rose a little everywhere. There are no zones, only a gradient; nothing to cut, the region construction collapses.

The distinguishing sign: biophysics has roughly the concentration of biology but twice the spread → two bundles.

### Do the bundles coincide with the pure zones

Intersect the top blocks of the biophysics mask with the top blocks of biology and physics.

- They consist almost entirely of those → the bundles coincide with the zones.
- There is a third group found in neither pure one → **a specific junction zone**.

### Is there an isthmus

Blocks between the bundles with a score above the background but below the centers.

A drop to zero → there is no isthmus, the branch is closed.

### Is the isthmus needed - ablation

**The strongest tool in the whole scheme.** Coarsen the isthmus while keeping both bundles precise.

Prediction: pure tasks do not suffer, mixed ones drop.

If so, the isthmus carries a function, and that is a direct measured confirmation of bridges between zones. Ablation turns the observation "zones exist" into the statement "zones are needed".

The substance: if the isthmus is coarsened down to the background, both experts survive separately but there is nothing to connect them - the model will answer about biology and about physics, while biophysics falls apart into two.

### Is it worth strengthening the isthmus - reverse ablation

Give the isthmus precision above the background at the same mean budget.

Mixed tasks improve → a **third knob** appears besides the scale (how much) and the address (where): strengthening connections.

### Linearity of the representation → mask mapping

Compare the cosines between the representations of the three queries and, separately, between their masks.

Do not mix up two different "betweens": the biophysics vector will almost certainly land in the middle **in representation space** - an ordinary property of embeddings. It does not follow that the mask is also in the middle **in block space**.

- **The vector is in the middle, the mask is bimodal** → the mapping is non-linear and meaningful: closeness of meanings does not transfer into closeness of blocks mechanically, the isthmus is not an averaging artifact but a structure of its own. More interesting.
- **Both pictures agree** → the mapping is nearly linear, the construction is simpler than expected.

Both outcomes are useful. This is also the first direct measurement of that very transition from representation space to weight space - the main hole of the statement.

---

## Expected topology of the ideal system

**Written before the runs, 2026-09-10.** This is the preregistration: the result is measured against it, not fitted to it afterwards. Rewording it after seeing the data would devalue the whole bench.

Seven properties the system should show if the statement is right:

**1. Not additive.** Masks do not add linearly. Reason: if they did, topics would be stored independently, and the whole construction about intertwining would be unnecessary - MoE would be enough. Additivity would mean the model does not use a shared foundation but keeps copies.

**2. Strongly overlapping.** Related zones share a significant part of their support. This is the source of compactness and what distinguishes the scheme from a router.

**3. With isthmuses that carry a function.** Between zones there is not emptiness but a working channel. Tested by ablation: coarsen the isthmus → pure tasks intact, mixed ones drop.

**4. Sparse.** For any given query a small share of blocks is sharpened. Otherwise there is nothing to cut and no budget gain.

**5. Hierarchical.** The shared foundation is wider and rises more often than specifics: the "natural sciences" zone covers both biology and physics. Follows from the compression criterion. Test: some blocks rise on almost all topics of a domain, others on one only.

**6. The representation → mask mapping is non-linear.** Closeness of meanings does not transfer into closeness of blocks mechanically.

**7. Uneven in zone size.** Frequent topics take more space and lie more precisely - they got more data. Tail topics are narrow and fragile.

## Method discipline of the bench

The bench is cheap and fits many questions - one set of runs, a dozen conclusions. Two conditions without which this is worthless:

- **Predictions are written in advance** (done above and for every test in "Step 2+"). When seven hypotheses are tested on the same data, some will "confirm" by chance. The only cure is that the expectation is fixed before looking.
- **A held-out set of topics.** The score is debugged on some domains (biology, math, chemistry, physics) and checked on others not touched during debugging (history, geography). The domains are school subjects. Otherwise it is easy to fit the score to specific examples without noticing.

---

## Step −1. Preregistration in git - do it first

**Before the first run. This is the only thing that really ties your hands.**

The point: a dated commit cannot be rewritten after the fact. Same as a risk register - dates give a provable order.

**What goes into the repo before the runs:**
- the expected topology, seven items;
- predictions for every test in "Step 2+" and steps 0–3, as directions (what is greater than what), not numbers;
- the run code;
- the list of held-out topics, marked as untouched during debugging.

**Commit order:** preregistration → run → results in separate commits. The history shows the predictions came before the data.

**On thresholds - an important caveat.** Numeric thresholds are not set in advance, and demanding them would be a mistake: **this is an exploratory experiment**, its purpose is to see for the first time how these quantities behave. The cosine scale depends on the dimension of the score vector, the normalization and the real spread - any number named before the run would be pulled out of thin air, and a preregistration with made-up thresholds is worse than none.

What is fixed is **directions, not values**:
- the within-topic cosine is strictly greater than the between-topic one, on all domain pairs;
- related pairs (biology–chemistry) fall between these two levels;
- the mask is concentrated, not flat.

This is enough for the result to be able to fail - nothing more is required from a preregistration.

**Numbers appear after the first run**, once the scale is known, and go into the second - confirmatory - pass on held-out topics. There they are meaningful. Before the main measurements it is useful to calibrate: known-unrelated texts and known paraphrases, to see the working range of the cosine. The calibrated thresholds are recorded in a separate commit, after calibration and before the confirmatory measurements.

**Publication only with a result.** Before a result it is an announcement, and an announcement without data is exactly the genre discussed in the Navier–Stokes story (a press release versus a published Lean formalization). *Updated 2026-09-12:* a second model and a text for others come only after a stable positive result on E2B with more than one score; external review comes after publication.

## What each prediction tests - the instrument or the topology

A distinction without which the preregistration is misread.

**Predictions about the instrument** (a trivial score through the activation norm): masks separate by topic, masks are concentrated.

If they fail, the conclusion is **only about the score**: the activation norm cannot catch the topic. It says nothing about the topology: a blind instrument will show identical masks for any structure of the model.

Not a dead end but a change of instrument. The next candidates by increasing cost: the output gradient per block instead of the norm; influence through removing the block (ablation); a learned version of the score.

**Predictions about the topology** (additivity, isthmuses, hierarchy, uneven zones) are read **only with a working instrument**. Until the score is confirmed, their results cannot be interpreted.

**On "it failed - so we simplified".** This reading is allowed for refining items but not for load-bearing ones - otherwise the construction becomes unfalsifiable and turns into a metaphor.

- **Load-bearing**: mask separability and concentration. If they fail with a known-working score, there are no zones and nothing to hold on to.
- **Refining**: additivity, isthmuses, hierarchy, uneven zones. They may fail without harm to the statement - that is the "simplified, fixable" case.

## Bets on the outcome of step 1 (fixed before the runs)

The predictions diverge, which is more useful than agreement - one cheap measurement resolves them.

**Vladimir:** the trivial score is enough for a **measurable, substantial gain** already in the naive setup.

**Claude:** there will be separation, but **dirty**. A score through the activation norm will light up both the topic and simply large blocks - separation is visible but weak. This intermediate outcome is more likely than either extreme.

**What resolves the dispute - background subtraction.** Average the score over many different queries and look at the deviation from the mean, not the absolute value.

- Quality after subtraction **did not change** → the signal was clean, Vladimir is right.
- Quality after subtraction **improved sharply** → the raw score caught mostly background, Claude is right.

Background subtraction is worth keeping in the plan as a prepared next step: it is cheap and removes exactly the artifact that spoils the naive version. With an intermediate outcome it is the right move - not to change the instrument entirely but to subtract the background first.

## The baseline: uniform quantization at the same memory

**One baseline, and it is uniform quantization at the same number of bits.** That is what a deployment would otherwise do, and it is the only comparison whose outcome changes a decision.

**A random mask is not a baseline here.** It was written into this plan as mandatory, and it is dropped, 2026-09-13, for two reasons:

- **It need not cost the same.** Zones around random blocks overlap less than the query's own, so at the same count and size they can store and read more. A control that spends more and answers worse says nothing about the address.
- **Beating it proves nothing anyone needs.** Nobody ships a model quantized by random mask. Being better than deliberate damage is not a result; being better than uniform quantization at the same cost is.

Where a control for shape is genuinely needed, it is the query's own zones carried elsewhere in the network, landed where they cover the same weight - same count, same cost. That isolates *where the zones point* without changing what they cost.

## The boundary of solo work

**The learned score (step 4) is taken outside what is done alone.** It is not "run the model and look" but training with a non-standard gradient through soft depth: parameterization, loss, stability, regime. A profession of its own; on one 3090 Ti and one person it takes not weeks but months.

**The construction does not collapse because of this.** The whole testable part is done on untrained scores: separability, concentration, overlaps, support shape, isthmuses, ablations, quality against uniform quantization at the same memory. The learned version improves the result, it does not create it.

If the naive score gives a signal, the learned one becomes exactly what co-authors, a supervisor or a group are needed for. This is a natural boundary: by hand up to step 3, beyond that someone else is needed.

## On timing

One day is only step 0 plus the naive score on a couple of domains, i.e. the answer to "is there a signal at all".

Realistically, a showable result takes **about two weeks**: a query set over several domains with a held-out part, run and logging infrastructure, scale calibration, the baseline, repeats for stability, then mask geometry and ablations. Plus what breaks along the way - on a mix of transformers, quantization and work with intermediate activations things break regularly.

Steps 4 and 5 are not part of these two weeks at all.

## Order of work: blind analysis

The bench is written in full, **all runs go at once, analysis only afterwards**.

This is not about suspense but about method: you cannot tune the next measurement to what you saw in the previous one. During the runs look only at whether the script crashed - no plots and no aggregates.

Raw results are written to files. They are all opened at once, after everything has run. The order of opening does not matter - there is nothing left to redo.

**The only exception is step 0**, whether activations separate by topic at all. It is not a result but a check that the model is fit: if representations do not separate by topic, the rest of the bench is pointless. Look at it separately and on the very first evening, before writing everything else.
