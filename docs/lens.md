---
title: The filter and its lenses
---

# The filter and its lenses

How the FoQLens filter lays precision over the weights. This page is the single source of truth for the mechanism: experiments cite it by commit (E010 and on), other documents link here instead of retelling it.

The mechanism is **FoQZones** - focusable quantization zones. A zone is a region of the weights that
the regulator reads at a higher precision than the rest; it is focusable in two senses, its size and
how far it rises, and the zones of a query come from the query itself. **FoQLens** names the project
and the model that carries the regulator; the lens is the picture of it.

The picture is a multi-lens objective: a glass over the whole network, and lenses inserted where the query needs to see. It is a picture, not the mechanism - what the mechanism does is allocate precision over blocks by the meaning of the query.

**The metric is an input, and it is not settled.** A zone is a ball of radius `R` in a distance `d` on blocks; the rules below hold in any `d`. Which `d` the bench should use is [issue #4](https://github.com/Vovanda/FoQLens/issues/4).

A ball here is a set of blocks, not a round island. It is round only in the metric used today, the 2D map, and that roundness is a property of the picture, not a finding about the model. In a metric taken on the full co-activation profile, or on a graph of blocks, the same ball is whatever the blocks near its center turn out to be - possibly a strand running through the layers. What shape the zones really have is one of the things the bench has to measure, not an assumption it starts from.

## The mechanism in formulas

**Given**

| Symbol | What it is | E2B |
| --- | --- | --- |
| `ℓ_0 < ℓ_1 < … < ℓ_m` | the ladder: the levels a block can be read at, set by the model and its storage; `ℓ_0` = ZERO, nothing read | 0, 2, 4, 6, 8, 16 bits (ZERO, D2, D4, D6, D8, bf16) |
| `w_b` | block `b`: its number of weights | |
| `d(a, b)` | **the metric on blocks** - an input of the mechanism, not a fixed choice. A zone is a ball in it | today: Euclidean distance on a 2D PCA layout of the co-activation distance `sqrt(2(1 - corr))` ([weight_map.py](../src/foqlens/weight_map.py)) |
| `c_i`, `r_i` | the query's expert zones: centers and base radii ([zones.md](zones.md)) | |

**Controls**

| Control | Symbol | Range | Default |
| --- | --- | --- | --- |
| base precision - the level everything outside the lenses is read at; `floor` in the scripts and the runs | `G = ℓ_γ` | any rung | - |
| focus_area - the size of the lenses | `f` | [0, 1) | - |
| focus_strength - how far the lens centers rise above the base | `g` | [0, 1] | - |
| profile - where each ring of a lens ends | stops `s_j` | 0 < s ≤ 1.5, strictly growing | even, below |
| attention level - `q_proj`, `k_proj` outside the lenses at ZERO base | `A = ℓ_α` | `α ≥ 1` | the lowest non-zero rung |
| combining overlapping lenses | `⊕` | sum, max | sum |

**The rules**

```
1. lens radius       R_i = r_i · f / (1 − f)
2. ceiling           κ = γ + ⌊g · (m − γ)⌋            g = 0: κ = γ, the lens is the base;   g = 1: κ = m
3. profile           levels ℓ_κ … ℓ_(γ+1), stops s_κ < … < s_(γ+1)
     default         s_j = (κ − j + 1) / (κ − γ)        even, the last at the edge
     G = ZERO        even over ℓ_κ … ℓ_2, and s_1 = 1.5  the lowest non-zero rung is pushed past the edge
4. lift of a lens    ρ_i(b) = d(b, c_i) / R_i,      L_i(b) = max(0, 1 − ρ_i(b) / s_last)
5. combining         L(b) = min(1, Σ_i L_i(b))   (sum)      or      max_i L_i(b)   (max)
6. level of a block  outside every lens (ρ_i(b) > s_last for all i):  G;  q_proj, k_proj at ZERO base:     A
                     inside:  ρ* = s_last · (1 − L(b)),  the level ℓ_j with the smallest s_j ≥ ρ*
7. memory            bits = Σ_b w_b · bits(level of b) / Σ_b w_b
```

**Worked examples** on the E2B ladder:

| Base | g | Ceiling (2) | Profile (3) | Beyond the lens |
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

No rung is fixed in the rules: every level follows from the ladder and the controls. What E2B needs is configuration of a run, not a rule: D2 is garbage on this model ([E006](../experiments/E006-read-depths/_index.md)), and calibrating it was measured and dropped ([reading notes](reading-notes.md)), so D2 serves only as a base to test against and as the ring pushed past a lens edge.

## The base precision

Everything outside the lenses is read at the base level `G`, any rung of the ladder. This is the glass of the picture: a coarse base is frosted glass - the whole network still works, coarsely. A ZERO base is an opaque one: outside the lenses there is nothing.

ZERO means emptiness - no knowledge, as an expert a mixture of experts did not choose. It is applied only where a zero row is emptiness:

| Module | A zero row | At ZERO base |
| --- | --- | --- |
| `gate_proj`, `up_proj`, `per_layer_input_gate` | the neuron or channel is off, act(0) = 0 | ZERO |
| `v_proj` | the head carries nothing along those rows | ZERO |
| `o_proj`, `down_proj`, `per_layer_projection` | the update to the residual stream is empty along those rows | ZERO |
| `q_proj`, `k_proj` | attention goes flat - a distortion, not emptiness | the attention level `A` |

The norms after each sub-block rescale the rows left - as the MoE block of Gemma 4 itself does with the experts it did not choose.

## The lenses

A lens is inserted into each **expert zone** of the query ([problem statement](problem-statement.md)): the zones are the peaks of the query's mask on the weight map, their number and base radii come from the query itself.

- **focus_area** sets the size of all lenses at once (rule 1): 0 no lenses, 1 one lens over the whole map; 0.5 the zones as found.
- **focus_strength** sets how far the centers rise above the base (rule 2), counted in rungs of the ladder above it: 0 the lens is the base, 1 the top rung. The scale follows the base: at a D4 base it runs D4 … bf16, at ZERO base 0 … bf16.

Memory is not set in advance: it is the result of the base, the size and the strength of the lenses (rule 7).

## The profile of a lens

From the center outwards the precision falls off along **stops**, as gradient stops in a graphics editor: one stop per level, the outer edge of that level's ring as a share of the radius. Written as `D8:0.33 D6:0.67 D4:1`.

- The default is even (rule 3). A level may be skipped: `D8:0.33 D6:1 D2:1.5` - the step from D6 to D2 is then a jump, a choice of the profile.
- A stop above 1 puts a ring beyond the lens edge. Not recommended, except at ZERO base: there the lowest non-zero rung is pushed past the edge by default, as a ring that may soften the step from the lens into emptiness. On E2B that rung is D2, which alone often gives nonsense; the hope is only less nonsense at the junction than with a hard cut.
- Stops are bounded by 1.5. The bound was argued on the 2D map, where a ring from 1 to 1.5 covers 1.25 of the area of the zone; in `k` dimensions the same ring covers `1.5^k − 1` - 2.38 at `k = 3`, 6.59 at `k = 5`, 56.7 at `k = 10`. **The number does not carry over to another metric** and has to be restated as a share of the weight mass covered, which is what makes it comparable at all ([issue #4](https://github.com/Vovanda/FoQLens/issues/4)). A ring larger than the bound would be a second zone, and the size belongs to focus_area.

## Where lenses overlap

Each lens lifts the blocks it covers, continuously, from 1 at its center to 0 at its last stop (rule 4). Overlapping lifts combine by a replaceable strategy (rule 5):

- **sum**, the default: lenses stacked on each other add their power, as thin lenses in contact (`1/f = 1/f_1 + 1/f_2`). The core of a deep overlap reaches the ceiling: a query on the border of two topics gets a sharp junction.
- **max**: the strongest lens only - no gain, and the junction does not fall back to the base.

Both are continuous: at the border of an overlap the second lens adds 0, so the junction rises smoothly. A single lens reads its stepped profile exactly, an overlap lowers no block, and neighbouring blocks differ by at most one rung unless the profile skips one (1D simulation of five layouts of two lenses, both strategies).

## Why a filter can make the model better

A lens restores at most the precision of the original weights at its center - it cannot make a weight better. But a model is not its weights; it is the computation over them. Behind the glass the associations that do not belong to the query are damped, and where two topics meet the isthmus is sharpened: the model keeps to the task instead of sinking into detail. For a single answer this may be close to neutral; for an agent that reasons over many steps it should add up. That is the main hypothesis of the project ([H4](hypotheses.md)), tested once the model exists.

## What is compared

At the same base, focus_area and focus_strength:

- the lenses of the query's own topic;
- the lenses of the paired topic - the address;
- the lenses of the backbone, one set of zones for every query - the query against generic importance;
- no mask at the same memory;
- random lenses of the same number and sizes - the noise to beat: a meaningful mask is expected to do better.

## Names used before

| Earlier | Now |
| --- | --- |
| precision_share (ADDENDUM-10) - the share of a preset budget | focus_strength - how far the lens centers rise above the base; there is no preset budget |
| focus_area (ADDENDUM-10) | focus_area - the same word, now the size of the lenses |
| coarse level / uniform background; frosted (D4) or opaque (ZERO) glass | the floor: any rung of the ladder |
| glass (ADDENDUM-11, E010 - the control) | floor - the same control, the word the runs and the scripts already use; glass stays the picture of it |
| - | the site shows it as **base precision**: what the network is read at before any lens. Not *baseline*, which in this bench means the control to beat - uniform quantization at the same memory |
