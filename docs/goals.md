# Goals

The goals of FoQLens and the roadmap of the bench. Details, measurements and reasoning are in the [plan](plan.md); every experiment is in [experiments/](../experiments/_index.md), every hypothesis in [hypotheses.md](hypotheses.md). Updated 2026-09-15.

## Main goal

**A precision regulator: the precision of the weights set by the meaning of the query - a coarse reading gives the structure of things, a sharp one the details - and adapted to the machine and to the value of the query.** It sets how finely the model works right now - its constant state, made up of three inputs: the difficulty of the task, the resources of the machine (load, heat, memory) and the value of the query. The idea and its background are in the author's article [«Квантование - всё, что вам нужно»](https://sawking.tech/blog/kvantovaniie-vsio-chto-vam-nuzhno).

Mixture of Experts is a special case of it: experts are zones with hard edges fixed at training, the rest not computed, a router choosing which. The regulator makes the same thing continuous - the zones emerge from the query, their edges fall off smoothly, and how sharp they are follows the context and the machine. The rest of the weights is read coarsely or, at a ZERO base, not loaded at all - the topology of MoE, with the zones chosen by the query. One set of weights then serves every device and every load, lean where little is needed and at full precision where the query needs it.

**The main hypothesis:** on a hard question the first pass gives a draft read mostly at base precision; the draft goes back into the input with the refinement, the zones of the next step land more precisely, and the answer ends better than that of the same model at native precision. The gain is expected where the iterations are - an agent or a model's reasoning ([H4](hypotheses.md)). It is tested once the FoQLens model exists (item 8 of the roadmap). Further out, a horizon: a network trained with zoning and read with FoQZones should beat a Mixture of Experts trained the classical way on the same data, holding no more in memory at any moment ([H5](hypotheses.md)). It needs training and zeros that are never loaded, and is reproduced on a small transformer.

## What this bench tests

FoQLens tests the core of the regulator on the weights: **can precision follow the meaning of the query?** The whole network is read at a base precision - any rung, down to nothing at all - and each expert zone of the query is read more precisely ([quantization-filter.md](quantization-filter.md)).

**The question to answer first** (2026-09-14): **does the regulator work** - does it give the model the right scale, the structure where a coarse reading is enough and the details where sharpness is needed, and does the model gain over iterations. Saving memory is a secondary goal, plan B: even without a gain in quality the mechanism saves memory at the same usability.

That makes the comparison uniform quantization at the same memory - what a deployment would otherwise do. A random mask is not a baseline: it costs more than the query's own zones and nobody ships one, so beating it settles nothing ([plan.md](plan.md)). Where the *place* of the zones has to be isolated, the control is the query's own zones carried elsewhere at the same cost.

The input from the machine (a governor that lowers precision under load or heat) and from the value of the query come after this core works.

## Roadmap

In order of work; items 3 and 4 ran in parallel. Each hypothesis is tested at its step.

1. **The bench: its design and optimization** - done, 2026-09-11 → 2026-09-14, and optimized as it goes. One stored copy of the
   weights read at 2 / 4 / 6 / 8 bits - since 2026-09-17 a k-quant base with residual slices over it
   ([E017](../experiments/E017-uniform-quantization-floor/results.md)); a decoding step is one CUDA graph over a
   static cache. A kernel that reads only the bits it needs is written for the former slice format, not yet wired
   into decoding.
2. **A corpus of what the model knows** - done, 2026-09-13 → 2026-09-15. Selected by the model's own answers in three regimes
   and frozen: 20,640 questions - 18,576 E2B-it knows and 2,064 it does not ([corpus.md](corpus.md)).
3. **Uniform quantization on the corpus** - done, 2026-09-15 → 2026-09-17. The answers at D8, D6, D4 and D2, judged by the
   model at source quality ([invariants.md](invariants.md)): D8 keeps 98.8% of the full model's knowledge,
   D6 96.8%, D4 91.0%, D2 51.4% ([E017 results](../experiments/E017-uniform-quantization-floor/results.md)).
   It is the baseline the filter is compared with, from D2. The first measurement, on naive rounding, left D2
   incoherent ([E016](../experiments/E016-uniform-quantization/results.md)). On the questions the full model does not
   know, D4 answers where bf16 refuses (E016) - on HotpotQA refusals fall from 12.1% to 4.6% and accepted answers rise from 13.4% to
   25.2%: the premise of [H4](hypotheses.md), a guess can be refined and a refusal cannot, came up on data not
   built to show it.
4. **The filter: how the zones are built** - now, since 2026-09-15. Grounded variants of the mask and
   of the zones built from it, read from the papers before any is coded.
5. **The address** - waits for the filter. Topics separate in the model, the mask is concentrated, zones
   of related topics overlap.
6. **Precision follows the meaning** - waits for the filter. How much of what the model knows the zones
   keep, against uniform quantization at the same memory - including the outcome that they keep no more.
7. **The regulator answers to the machine** - waits. Precision lowered under load or heat, the zones of
   the query kept sharpest.
8. **Agents on the FoQLens model - the main hypothesis** - waits for the model (zones with the address
   taken online from the first layers). An agent solves a hard multi-step task - designing a software
   architecture, for example - as a chain of a draft and refinements, on the zone model, on the same
   model at native precision and on uniform quantization at the same memory; compared are the result of
   the chain and what it cost (memory, compute, tokens). [H4](hypotheses.md) expects the zone model to
   end better than native precision at a lower cost.

Beyond solo work: a learned score, only if the untrained ones give an effect.

**The steps as numbered before 2026-09-15** keep their numbers where they are written - the
preregistration is never edited, and [plan.md](plan.md), [hypotheses.md](hypotheses.md) and
[data-sources.md](data-sources.md) refer to it:

| Step | Roadmap item |
| --- | --- |
| −1 - preregistration in git | a rule of the work, below |
| 0, 1, 2, 2+ - separation, masks, overlap, geometry | 5 |
| 2.5 - the corpus | 2 |
| 3 - precision follows the meaning | 6 |
| 4 - learned score | beyond solo work |
| 5 - residuals instead of copies | 1 |
| 6 - the regulator and the machine | 7 |
| 7 - agents | 8 |

## Order and boundaries

- Preregistration in git: before every run the repo holds its hypotheses and predictions, fixed as
  directions; the main preregistration is in [prereg/](../prereg/), each experiment's addenda in its folder.
- Blind analysis: all runs of an experiment first, then everything is opened at once. The exception is the
  check that the model is fit for the bench at all - do topics separate.
- A second model (E4B) only after a stable positive result on E2B with more than one score.
- Publication only if there is a result to publish; external review comes after it.
- Literature is read when the stage that needs it comes, with notes that cite the passage.
