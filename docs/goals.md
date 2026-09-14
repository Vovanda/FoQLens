# Goals

FoQLens goals in order of execution. Each next goal opens only if the previous one passed. Details, measurements and reasoning are in the [plan](plan.md); every experiment is in [experiments/](../experiments/_index.md), every hypothesis in [hypotheses.md](hypotheses.md). Updated 2026-09-13.

## Main goal

**A universal mechanism that lets a model work more economically and react to its environment and context: a precision regulator.** It sets how finely the model works right now - not an optimizer on top of the system but its constant state, made up of three inputs: the difficulty of the task, the resources of the machine (load, heat, memory) and the value of the query. The idea and its background are in the author's article [«Квантование - всё, что вам нужно»](https://sawking.tech/blog/kvantovaniie-vsio-chto-vam-nuzhno).

Mixture of Experts is a special case of it: experts are zones with hard edges fixed at training, the rest not computed, a router choosing which. The regulator makes the same thing continuous - the zones emerge from the query, their edges fall off smoothly, and how sharp they are follows the context and the machine. The rest of the weights is read coarsely or, at a ZERO base, not loaded at all - the topology of MoE, with the zones chosen by the query. One set of weights then serves every device and every load, lean where little is needed and at full precision where the query needs it.

**The main hypothesis:** on a hard question the first pass gives a draft read mostly at base precision; the draft goes back into the input with the refinement, the zones of the next step land more precisely, and the answer ends better than that of the same model at native precision. The gain is expected where the iterations are - an agent or a model's reasoning ([H4](hypotheses.md)). It is tested once the FoQLens model exists (step 7). Further out, a horizon: a network trained with zoning and read with FoQZones should beat a Mixture of Experts trained the classical way on the same data, holding no more in memory at any moment ([H5](hypotheses.md)). It needs training and zeros that are never loaded, and is reproduced on a small transformer.

## What this bench tests

FoQLens tests the core of the regulator on the weights: **can precision follow the meaning of the query?** The whole network is read at a base precision - any rung, down to nothing at all - and each expert zone of the query is read more precisely ([quantization-filter.md](quantization-filter.md)).

**The question to answer first, and the one that decides whether any of this is worth building** (fixed 2026-09-13): **over what interval of the regulator's settings does the model stay usable, and how much memory does that interval actually save - if it saves any.** Not "is the address better than a control", but "where can this be set, and what does it buy". An interval that saves nothing is an answer; so is an interval too narrow to hold a regulator.

That makes the comparison uniform quantization at the same memory - what a deployment would otherwise do. A random mask is not a baseline: it costs more than the query's own zones and nobody ships one, so beating it settles nothing ([plan.md](plan.md)). Where the *place* of the zones has to be isolated, the control is the query's own zones carried elsewhere at the same cost.

The input from the machine (a governor that lowers precision under load or heat) and from the value of the query come after this core works.

## Goals by step

> **Verdicts before 2026-09-13 are withdrawn:** they rested on questions the model did not know. This came out at the start of checking the corpus - whether the model understands its questions: under every order of the options it answered 31% of them right, 3% in mathematics; the rest was guessing. Every hypothesis is untested. Details: [corpus.md](corpus.md). The statuses below are what the runs reported at the time.


**Step −1. Preregistration in git** - done.
Before every run the repo holds its hypotheses and predictions, fixed as directions; the main preregistration is in [prereg/](../prereg/), each experiment's addenda in its folder.

**Step 0. Topics separate in representations** - done on E2B: **yes** ([E001](../experiments/E001-run1-exploration/_index.md)).
At the middle layer k-means ARI 0.98 for biology-math (the run's summary was deleted with the corpus; the number is in [E001 results](../experiments/E001-run1-exploration/results.md)). The held-out pair (history-geography) barely separates even here (ARI 0.04).

**Step 1. Masks are separable and concentrated** - **not confirmed with the naive score** ([E001](../experiments/E001-run1-exploration/_index.md)): exploration passed weakly, the held-out confirmation failed. The gradient score (E002, deleted with the corpus) replaced it as the mask source but was never checked on its own. Load-bearing.

**Step 2. Zones overlap** - exploration passed, confirmation failed ([E001](../experiments/E001-run1-exploration/_index.md)). Read again when an instrument passes step 1.

**Step 2+. Mask geometry** - mixed in exploration; the hand-written mixed domain came out biology-like, so a better mixed domain is needed first.

**Step 2.5. Exploration, before any hypothesis is stated again** - in progress, 2026-09-13.
Its first piece is the corpus: the full model answers five datasets in its own words, and the questions it gets right become the set every later run reads ([corpus.md](corpus.md)).
What the model is actually competent at and on which corpora; what a gradient mask tracks and whether anything about it is stable; what the regulator's settings do to an answer a person would accept. This step states no predictions and settles nothing. **Its output is a list of invariants** - properties that survive a change of corpus, of metric or of settings ([invariants.md](invariants.md)) - and the hypotheses worth preregistering are then written about those. A property seen once under one setup does not qualify; that is the mistake this whole reset came from. The hypotheses that follow it are expected to differ from the ones listed today ([hypotheses.md](hypotheses.md)).

**Step 3. Precision follows the meaning** - **not started.** Eight runs were made ([E004](../experiments/E004-injection/_index.md), [E005](../experiments/E005-backbone/_index.md), [E007](../experiments/E007-dilation/_index.md) through [E014](../experiments/E014-moved-zones/_index.md)) and none of them measured what it set out to; their code and preregistrations remain as a record. Before this step is attempted again: a corpus and a metric that can carry a verdict ([corpus.md](corpus.md)), and a mask that is asked about the answer rather than about the prompt.
Done when: the interval of settings over which the model stays usable is named, and the memory that interval saves against uniform quantization is named with it - including the honest outcome that it saves nothing.

**Step 4. Learned score** - beyond solo work; only if the untrained scores give an effect.

**Step 5. Memory follows the zones** - engineering. Done so far ([E006](../experiments/E006-read-depths/_index.md)): one residual-sliced copy read at 2 / 4 / 6 / 8 bits, the bf16 weights leave the GPU (-1.63 GiB on E2B, D8 as good as int8), and every block can store only the depth it is read to. Speed and energy savings need a kernel that reads only the bits it needs.

**Step 6. The regulator reacts to the machine** - after step 3: precision lowered under load or heat, the zones of the query kept sharpest.

**Step 7. Agents on the FoQLens model - the main hypothesis** - once the model exists (zones with the address taken online from the first layers). An agent solves a hard multi-step task - designing a software architecture, for example - as a chain of a draft and refinements, on the zone model, on the same model at native precision and on uniform quantization at the same memory; compared are the result of the chain and what it cost (memory, compute, tokens).
Done when: the chains are compared; [H4](hypotheses.md) expects the zone model to end better than native precision at a lower cost.

## Order and boundaries

- Blind analysis: all runs of an experiment first, then everything is opened at once. The exception is step 0.
- A second model (E4B) only after a stable positive result on E2B with more than one score.
- Publication only if there is a result to publish; external review comes after it.
- Literature is read when the stage that needs it comes, with notes that cite the passage.
