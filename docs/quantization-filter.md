---
title: The quantization filter and its zones
---

# The quantization filter and its zones

How the quantization filter lays precision over the weights. This page is the single source of truth for the mechanism: experiments cite it by commit (E010 and on), other documents link here instead of retelling it.

The mechanism is **FoQZones** - focusable quantization zones. A zone is a region of the weights that
the regulator reads at a higher precision than the rest; it is focusable in two senses, its size and
how far it rises, and the zones of a query come from the query itself. **FoQLens** names the project
and the model that carries the regulator; the lens is its metaphor.

The metaphor is a multi-lens objective: a glass over the whole network, and lenses inserted where the query needs to see. It is a metaphor, not the mechanism - what the mechanism does is allocate precision over blocks by the meaning of the query.

**The distance `d` is the one the bench uses today.** The rules below need a distance between blocks, and today it is the distance on the weight map. In what space the regions of a query are defined, and what makes two weights close, is the first open hole of the [problem statement](problem-statement.md); the shape of a zone is meant to grow from an oval to arbitrary shapes with bridges.
## The mechanism in formulas

**Given**

| Symbol | What it is | E2B |
| --- | --- | --- |
| `ℓ_0 < ℓ_1 < … < ℓ_m` | the ladder: the levels a block can be read at, set by the model and its storage; `ℓ_0` = ZERO, nothing read | 0, 2, 4, 6, 8, 16 bits (ZERO, D2, D4, D6, D8, bf16) |
| `w_b` | block `b`: its number of weights | |
| `d(a, b)` | the distance between blocks; the space it is taken in is open ([problem statement](problem-statement.md), hole 1) | today: Euclidean distance on a 2D PCA layout of the co-activation distance `sqrt(2(1 - corr))` ([weight_map.py](../src/foqlens/weight_map.py)) |
| `c_i`, `r_i` | the query's expert zones: centers and base radii ([zones.md](zones.md)) | |

**Controls**

| Control | Symbol | Range | Default |
| --- | --- | --- | --- |
| base precision - the level everything outside the zones is read at; `floor` in the scripts and the runs | `G = ℓ_γ` | any rung | - |
| focus_area - the size of the zones | `f` | [0, 1) | - |
| focus_strength - how far the zone centers rise above the base | `g` | [0, 1] | - |
| profile - where each ring of a zone ends | stops `s_j` | 0 < s ≤ 1.5, strictly growing | even, below |
| attention level - `q_proj`, `k_proj` outside the zones at ZERO base | `A = ℓ_α` | `α ≥ 1` | the lowest non-zero rung |
| combining overlapping zones | `⊕` | sum, max | sum |

**The rules**

```
1. zone radius       R_i = r_i · f / (1 − f)
2. ceiling           κ = γ + ⌊g · (m − γ)⌋            g = 0: κ = γ, the zone is the base;   g = 1: κ = m
3. profile           levels ℓ_κ … ℓ_(γ+1), stops s_κ < … < s_(γ+1)
     default         s_j = (κ − j + 1) / (κ − γ)        even, the last at the edge
     G = ZERO        even over ℓ_κ … ℓ_2, and s_1 = 1.5  the lowest non-zero rung is pushed past the edge
4. lift of a zone    ρ_i(b) = d(b, c_i) / R_i,      L_i(b) = max(0, 1 − ρ_i(b) / s_last)
5. combining         L(b) = min(1, Σ_i L_i(b))   (sum)      or      max_i L_i(b)   (max)
6. level of a block  outside every zone (ρ_i(b) > s_last for all i):  G;  q_proj, k_proj at ZERO base:     A
                     inside:  ρ* = s_last · (1 − L(b)),  the level ℓ_j with the smallest s_j ≥ ρ*
7. memory            bits = Σ_b w_b · bits(level of b) / Σ_b w_b
```

**Worked examples** on the E2B ladder:

| Base | g | Ceiling (2) | Profile (3) | Beyond the zone |
| --- | --- | --- | --- | --- |
| D4 (`γ = 2`) | 1 | `κ = 5`, bf16 | `bf16:0.33 D8:0.67 D6:1` | D4 |
| D4 | 0.5 | `κ = 3`, D6 | `D6:1` | D4 |
| D2 (`γ = 1`) | 1 | `κ = 5`, bf16 | `bf16:0.25 D8:0.5 D6:0.75 D4:1` | D2 |
| ZERO (`γ = 0`) | 1 | `κ = 5`, bf16 | `bf16:0.25 D8:0.5 D6:0.75 D4:1 D2:1.5` | ZERO |
| ZERO | 0.5 | `κ = 2`, D4 | `D4:1 D2:1.5` | ZERO |
| any | 0 | `κ = γ` | none | the base everywhere |

**What the names mean.** `D` is depth, counted in slices of the one stored copy, and the number is
the bits it comes to: the first slice quantizes the weight to 2 bits, the second quantizes what the
first left with a step four times finer, and so on, so reading the first k slices gives a 2k-bit
weight (`foqlens.quant.SlicedWeight`, 4 slices on E2B). They are not different quantizers - they are
how deep the same copy is read.

`bf16` is the top rung, not a yardstick outside the ladder: it is the weight as stored, read without
slices at all. There is no `D16` because eight slices would come to the same 16 bits the stored
weight already occupies, while costing an unpacking the stored weight does not need. What is not on
the ladder and could be - D10 and D12, the fifth and sixth slice - has not been measured.

No rung is fixed in the rules: every level follows from the ladder and the controls. What E2B needs is configuration of a run, not a rule: uniform D2 by round-to-nearest, without calibration, breaks E2B ([E006](../experiments/E006-read-depths/_index.md)), and calibrating the first slice within this storage was tried and dropped ([reading notes](reading-notes.md)), so D2 serves only as a base to test against and as the ring pushed past a zone edge.

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

Each **expert zone** of the query is read above the base ([problem statement](problem-statement.md)): the zones are the peaks of the query's mask on the weight map, their number and base radii come from the query itself.

- **focus_area** sets the size of all zones at once (rule 1): 0 no zones, 1 one zone over the whole map; 0.5 the zones as found.
- **focus_strength** sets how far the centers rise above the base (rule 2), counted in rungs of the ladder above it: 0 the zone is the base, 1 the top rung. The scale follows the base: at a D4 base it runs D4 … bf16, at ZERO base 0 … bf16.

Memory is not set in advance: it is the result of the base, the size and the strength of the zones (rule 7).

## The profile of a zone

From the center outwards the precision falls off along **stops**, as gradient stops in a graphics editor: one stop per level, the outer edge of that level's ring as a share of the radius. Written as `D8:0.33 D6:0.67 D4:1`.

- The default is even (rule 3). A level may be skipped: `D8:0.33 D6:1 D2:1.5` - the step from D6 to D2 is then a jump, a choice of the profile.
- A stop above 1 puts a ring beyond the zone edge. Not recommended, except at ZERO base: there the lowest non-zero rung is pushed past the edge by default, as a ring that may soften the step from the zone into emptiness. On E2B that rung is D2, which alone often gives nonsense; the hope is only less nonsense at the junction than with a hard cut.
- Stops are bounded by 1.5, a parameter of the profile: a ring larger than that would be a second zone, and the size belongs to focus_area. The bound was set on the 2D map and is revisited together with the space of the zones.

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
- uniform quantization at the same memory - the baseline, what a deployment would otherwise do. Random zones of the same number and sizes are not a baseline: they overlap less, so they cost more (dropped 2026-09-13, [plan](plan.md)).

## Names used before

| Earlier | Now |
| --- | --- |
| precision_share (ADDENDUM-10) - the share of a preset budget | focus_strength - how far the zone centers rise above the base; there is no preset budget |
| focus_area (ADDENDUM-10) | focus_area - the same word, now the size of the zones |
| coarse level / uniform background; frosted (D4) or opaque (ZERO) glass | the floor: any rung of the ladder |
| glass (ADDENDUM-11, E010 - the control) | floor - the same control, the word the runs and the scripts already use; glass stays its metaphor |
| - | the site shows it as **base precision**: what the network is read at before any zone. Not *baseline*, which in this bench means the control to beat - uniform quantization at the same memory |
