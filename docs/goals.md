# Goals

FoQLens goals in order of execution. Each next goal opens only if the previous one passed. Details, measurements and reasoning are in the [plan](plan.md); every experiment is in [experiments/](../experiments/_index.md), every hypothesis in [hypotheses.md](hypotheses.md). Updated 2026-09-12.

## Main goal

**A universal mechanism that lets a model work more economically and react to its environment and context: a precision regulator.** It sets how finely the model works right now - not an optimizer on top of the system but its constant state, made up of three inputs: the difficulty of the task, the resources of the machine (load, heat, memory) and the value of the query. The idea and its background are in the author's article [«Квантование - всё, что вам нужно»](https://sawking.tech/blog/kvantovaniie-vsio-chto-vam-nuzhno).

Mixture of Experts is a special case of it: experts are zones with hard edges fixed by the architecture, behind opaque glass, opened by a router. The regulator makes the same thing continuous - the zones emerge from the query, their edges fall off smoothly, the rest of the weights is coarse rather than absent, and how sharp the zones are follows the context and the machine. One set of weights then serves every device and every load, lean where little is needed and at full precision where the query needs it.

**The main hypothesis:** a model behind the filter is not only cheaper but better where it counts - it does not sink into detail the task does not need. An agent that reasons over many steps on it should be both more efficient and more accurate than on the same model read in full ([H4](hypotheses.md)). It is tested once the FoQLens model exists (step 7). A stronger one: the lens model should be more accurate and more efficient than a Mixture of Experts built from the same model ([H5](hypotheses.md)) - its bench part, the MoE end of the lens scale at the same memory, comes after the lenses work.

## What this bench tests

FoQLens tests the core of the regulator on the weights: **can precision follow the meaning of the query?** The whole network sits behind a glass - at any precision, down to nothing at all - and a lens is inserted into each expert zone of the query ([lens.md](lens.md)). It holds if, on the curve of quality against memory, the query's own lenses lie above the paired topic's lenses, the lenses of generic importance and the same memory without a mask - the same quality for less memory; random lenses are the floor. MoE-like layouts (an empty glass, hard edges) as one end of the same scale come after.

The input from the machine (a governor that lowers precision under load or heat) and from the value of the query come after this core works.

## Goals by step

**Step −1. Preregistration in git** - done.
Before every run the repo holds its hypotheses and predictions, fixed as directions; the main preregistration is in [prereg/](../prereg/), each experiment's addenda in its folder.

**Step 0. Topics separate in representations** - done on E2B: **yes** ([E001](../experiments/E001-run1-exploration/_index.md)).
At the middle layer k-means ARI 0.98 for biology-math ([`runs/E001-run1-exploration/step0/e2b/summary.json`](../runs/E001-run1-exploration/step0/e2b/summary.json)). The held-out pair (history-geography) barely separates even here (ARI 0.04).

**Step 1. Masks are separable and concentrated** - **not confirmed with the naive score** ([E001](../experiments/E001-run1-exploration/_index.md)): exploration passed weakly, the held-out confirmation failed. The gradient score ([E002](../experiments/E002-gradient-score/_index.md)) replaced it as the mask source but was never checked on its own. Load-bearing.

**Step 2. Zones overlap** - exploration passed, confirmation failed ([E001](../experiments/E001-run1-exploration/_index.md)). Read again when an instrument passes step 1.

**Step 2+. Mask geometry** - mixed in exploration; the hand-written mixed domain came out biology-like, so a better mixed domain is needed first.

**Step 3. Precision follows the meaning** - in progress.
- Generic block importance carries most of the budget ([E005](../experiments/E005-backbone/_index.md)); flat topic masks add no address ([E004](../experiments/E004-injection/_index.md), [E007](../experiments/E007-dilation/_index.md)).
- Expert zones from the gradient mask carry an address for far-apart topics, most of all when bits are scarce - on the legacy fixed-budget layout, checked along the way ([E008](../experiments/E008-zones-fixed-budget/_index.md), [E009](../experiments/E009-zones-matrix/_index.md)).
- Next: the lens layout (E010, ADDENDUM-11) - the test of the idea itself; then the address computed online from the first layers instead of a full pass.
Done when: the quality-against-memory curve of the query's lenses lies above the paired topic's lenses, generic importance and the same memory without a mask, on E2B for more than one mask score.

**Step 4. Learned score** - beyond solo work; only if the untrained scores give an effect.

**Step 5. Memory follows the lenses** - engineering. Done so far ([E006](../experiments/E006-read-depths/_index.md)): one residual-sliced copy read at 2 / 4 / 6 / 8 bits, the bf16 weights leave the GPU (-1.63 GiB on E2B, D8 as good as int8), and every block can store only the depth it is read to. Speed and energy savings need a kernel that reads only the bits it needs.

**Step 6. The regulator reacts to the machine** - after step 3: precision lowered under load or heat, the lenses of the query kept sharpest.

**Step 7. Agents on the lens model - the main hypothesis** - once the FoQLens model exists (lenses with the address taken online from the first layers). An agent solves a hard multi-step task - designing a software architecture, for example - on the lens model and on the same model in bf16; compared are the quality of the result and what it cost (memory, compute, tokens).
Done when: the agent on the lens model is at least as good as on the full model at a lower cost - and the hypothesis expects better.

## Order and boundaries

- Blind analysis: all runs of an experiment first, then everything is opened at once. The exception is step 0.
- A second model (E4B) only after a stable positive result on E2B with more than one score.
- Publication only if there is a result to publish; external review comes after it.
- Literature is read when the stage that needs it comes, with notes that cite the passage.
