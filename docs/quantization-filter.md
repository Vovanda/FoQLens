---
title: The quantization filter and its zones
---

# The quantization filter and its zones

How the quantization filter lays precision over the weights. This page is the single source of truth for the mechanism: experiments cite it by commit, other documents link here instead of retelling it.

The mechanism is **FoQZones** - focusable quantization zones. A zone is a region of the weights that
the regulator reads at a higher precision than the rest; it is focusable in two senses, its size and
how far it rises, and the zones of a query come from the query itself. **FoQLens** names the project
and the model that carries the regulator; the lens is its metaphor.

The metaphor of the mechanism - a multi-lens chosen for the query - is in [visual-metaphor.md](visual-metaphor.md).

**The distance `d` is an input of the mechanism.** The rules below need a distance between blocks and do not say which. In what space the regions of a query are defined, and what makes two weights close, is the first open hole of the [problem statement](problem-statement.md). Where zones grow from, along what, how far and how fast are strategies to be tested: [zone-strategies.md](zone-strategies.md).

## The mechanism in formulas

**Given**

| Symbol | What it is | E2B |
| --- | --- | --- |
| `ℓ_0 < ℓ_1 < … < ℓ_m` | the ladder: the levels a block can be read at, set by the model and its storage; `ℓ_0` = ZERO, nothing read | 0, 2, 4, 6, 8 bits (ZERO, D2, D4, D6, D8) |
| `w_b` | block `b`: its number of weights | |
| `d(a, b)` | the distance between blocks; the space it is taken in is open ([problem statement](problem-statement.md), hole 1) | not chosen ([zone-strategies.md](zone-strategies.md)) |
| `D` | the width of the network in `d` | |
| `c_i` | the centers of the query's expert zones ([where they come from](#where-the-zones-come-from)) | |

**Controls**

| Control | Symbol | Range | Default |
| --- | --- | --- | --- |
| base precision - the level everything outside the zones is read at; `floor` in the scripts and the runs | `G = ℓ_γ` | any rung | - |
| focus_area - the share of the network a zone covers | `f` | [0, 1] | - |
| focus_strength - how far the zone centers rise above the base | `g` | [0, 1] | - |
| profile - where each ring of a zone ends | stops `s_j` | s > 0, strictly growing; a stop above 1 is a ring past the edge | even, below |
| attention level - `q_proj`, `k_proj` outside the zones at ZERO base | `A = ℓ_α` | `α ≥ 1` | the lowest non-zero rung |
| combining overlapping zones | `⊕` | sum, max | sum |

**The rules**

```
1. zone radius       R = f · D                        f = 0: the center only;   f = 1: the whole network
2. ceiling           κ = γ + ⌊g · (m − γ)⌋            g = 0: κ = γ, the zone is the base;   g = 1: κ = m
3. profile           levels ℓ_κ … ℓ_(γ+1), stops s_κ < … < s_(γ+1)
     default         s_j = (κ − j + 1) / (κ − γ)        even, the last at the edge
4. lift of a zone    ρ_i(b) = d(b, c_i) / R,        L_i(b) = max(0, 1 − ρ_i(b) / s_last)       linear for now
5. combining         L(b) = min(1, Σ_i L_i(b))   (sum)      or      max_i L_i(b)   (max)
6. level of a block  outside every zone (ρ_i(b) > s_last for all i):  G;  q_proj, k_proj at ZERO base:     A
                     inside:  ρ* = s_last · (1 − L(b)),  the level ℓ_j with the smallest s_j ≥ ρ*
7. memory            bits = Σ_b w_b · bits(level of b) / Σ_b w_b
```

**Worked examples** on the E2B ladder:

| Base | g | Ceiling (2) | Profile (3) | Beyond the zone |
| --- | --- | --- | --- | --- |
| D4 (`γ = 2`) | 1 | `κ = 4`, D8 | `D8:0.5 D6:1` | D4 |
| D4 | 0.5 | `κ = 3`, D6 | `D6:1` | D4 |
| D2 (`γ = 1`) | 1 | `κ = 4`, D8 | `D8:0.33 D6:0.67 D4:1` | D2 |
| ZERO (`γ = 0`) | 1 | `κ = 4`, D8 | `D8:0.25 D6:0.5 D4:0.75 D2:1` | ZERO |
| ZERO | 0.5 | `κ = 2`, D4 | `D4:0.5 D2:1` | ZERO |
| any | 0 | `κ = γ` | none | the base everywhere |

**What the names mean.** `D` is depth, counted in 2-bit planes of the one stored copy, and the number
is the bits it comes to. The copy is a k-quant base after llama.cpp - Q2_K, one plane, or Q4_K, two
planes for the sensitive module classes - with residual slices over it, each quantizing what the planes
before it left with a step four times finer (`foqlens.kquant.KSlicedWeight`, E002). They are not
different quantizers - they are how deep the same copy is read. A module on a Q4_K base reads the same
at D2 and D4, so base precision D2 comes to about 3.3 bits per weight on E2B.

`D8` is the top rung. The model holds no bf16 weights: bf16 is the precision of the source model, the
reference the judge and the retention are measured against. A rung above D8 would take four more slices,
about 16.6 bits per weight - more than the bf16 weight itself.

No rung is fixed in the rules: every level follows from the ladder and the controls. On E2B naive
round-to-nearest at two bits breaks the model ([E001](../experiments/E001-uniform-quantization/results.md)); the
k-quant copy keeps 51.4% of bf16's knowledge at D2 ([E002](../experiments/E002-base-precision-d2/results.md)),
so D2 serves as a base precision and as the ring pushed past a zone edge.

## Where the zones come from

A source scores how much the query needs each block; the peaks of that score in the metric `d` are the centers
of the query's expert zones. Their number is not fixed: the query decides it. Which source and which metric -
the variants to test are in [zone-strategies.md](zone-strategies.md).

## The base precision

Everything outside the zones is read at the base level `G`, any rung of the ladder. In the metaphor this is the glass: a coarse base is frosted glass - the whole network still works, coarsely. A ZERO base is an opaque one: outside the zones there is nothing.

ZERO means emptiness - no knowledge, as an expert a mixture of experts did not choose. It is applied only where a zero row is emptiness:

| Module | A zero row | At ZERO base |
| --- | --- | --- |
| `gate_proj`, `up_proj`, `per_layer_input_gate` | the neuron or channel is off, act(0) = 0 | ZERO |
| `v_proj` | the head carries nothing along those rows | ZERO |
| `o_proj`, `down_proj`, `per_layer_projection` | the update to the residual stream is empty along those rows | ZERO |
| `q_proj`, `k_proj` | attention goes flat - a distortion, not emptiness | the attention level `A` |

The norms after each sub-block rescale the rows left - as the MoE block of Gemma 4 itself does with the experts it did not choose.

## The zones

Each **expert zone** of the query is read above the base ([problem statement](problem-statement.md)): the zones grow from the peaks of the query's score, and their number comes from the query itself.

- **focus_area** sets the size of all zones at once (rule 1), as a share of the network: 0 the center only, 1 the whole network, linear in between.
- **focus_strength** sets how far the centers rise above the base (rule 2), counted in rungs of the ladder above it: 0 the zone is the base, 1 the top rung. The scale follows the base: at a D4 base it runs D4 … D8, at ZERO base 0 … D8.

Memory is not set in advance: it is the result of the base, the size and the strength of the zones (rule 7).

## The profile of a zone

From the center outwards the precision falls off along **stops**, as gradient stops in a graphics editor: one stop per level, the outer edge of that level's ring as a share of the radius. Written as `D8:0.33 D6:0.67 D4:1`.

- The default is even (rule 3). A level may be skipped: at base D2, `D8:0.5 D6:1` - the step from D6 to D2 is then a jump, a choice of the profile.
- Every stop above 1 is a ring beyond the zone edge, as many rings as such stops: at ZERO base `D8:0.5 D6:1 D4:1.2 D2:1.5` gives two, D4 out to 1.2 radii and D2 out to 1.5. Such rings may soften the step from the zone into emptiness.
- How fast precision falls off from the center is a strategy of its own; today the fall is linear ([zone-strategies.md](zone-strategies.md)).

## Where zones overlap

Each zone lifts the blocks it covers, continuously, from 1 at its center to 0 at its last stop (rule 4). Overlapping lifts combine by a replaceable strategy (rule 5):

- **sum**, the default: the lifts of overlapping zones add up, capped at 1. The core of a deep overlap reaches the ceiling: a query on the border of two topics gets a sharp junction.
- **max**: the strongest zone only - no gain, and the junction does not fall back to the base.

Both are continuous: at the border of an overlap the second zone adds 0, so the junction rises smoothly. A single zone reads its stepped profile exactly, an overlap lowers no block, and neighbouring blocks differ by at most one rung unless the profile skips one (1D simulation of five layouts of two zones, both strategies).

## Why a quantization filter can make the model better

A zone restores at most the precision of the stored weights at its center; it cannot make a weight better. The claim is about an answer reached over several steps. On a hard question the first pass gives a draft read mostly at base precision; the draft goes back into the input with the refinement, the zones of the next step land more precisely, and the weights outside them, read at base precision, affect the answer less. That is the main hypothesis of the project ([H4](hypotheses.md)), tested once the model exists. It asks one thing of the rules above that they do not yet have: a strength of its own for each zone, so that on a question the model knows poorly the zones stay weak and the draft stays a draft. Today the ceiling (rule 2) is one for all zones.

## What is compared

At the same base, focus_area and focus_strength:

- the zones of the query's own topic;
- the zones of the paired topic - the address;
- the zones of the backbone, one set of zones for every query - the query against generic importance;
- no mask at the same memory;
- uniform quantization at the same memory - the baseline, what a deployment would otherwise do. Random zones are not a baseline: beating deliberate damage proves nothing (dropped 2026-09-13, [plan](plan.md)).

## Names used before

| Earlier | Now |
| --- | --- |
| precision_share - the share of a preset budget | focus_strength - how far the zone centers rise above the base; there is no preset budget |
| focus_area (earlier) | focus_area - the same word, now the size of the zones |
| coarse level / uniform background; frosted (D4) or opaque (ZERO) glass | the floor: any rung of the ladder |
| glass | floor - the same control, the word the runs and the scripts already use; glass stays its metaphor |
| - | the site shows it as **base precision**: what the network is read at before any zone. Not *baseline*, which in this bench means the control to beat - uniform quantization at the same memory |
