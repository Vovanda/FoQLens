---
title: The precision regulator
---

# The precision regulator

What decides the depth every weight is read at for a given query. This page is the single source of truth about it:
the experiments cite it by commit, and other documents link here instead of retelling it.

| What | It is |
| --- | --- |
| **The precision regulator** | what decides the depth of the reading for a query. It builds the map |
| **The map of the quantization filter** | which weights to read finely and which coarsely. In short, the map |
| **The quantization filter** | the reading of the network by that map. There is no module of that name in the code: `foqlens/regulator.py` sets the levels and the kernel reads by them |
| **A layout** | the same map spread over the blocks: what the model reads on a pass |
| **A zone** | an area of weights the regulator reads more finely than the rest. It focuses by size and by how high it lifts; the zones come from the query itself |
| **A rung** | the depth a block is read to: D2, D4, D6, D8 - two, four, six, eight bits a weight |
| **The base precision** | the rung everything outside the zones is read at |

The mechanism of the zones is **FoQZones** - focusable quantization zones. **FoQLens** is the name of the project and
of the model that carries the regulator; the lens is its metaphor, drawn in
[visual-metaphor.md](visual-metaphor.md). The rest of the project's words are in the [glossary](glossary.md).

**Two variants.** A - the map is built whole before the pass, from the address of the first layers carried onto the
rest. B - the decision is taken layer by layer as the pass runs, from what enters each layer. Both read the network
once and differ in where the decision is made; the rules of the level are common to both. **Variant B is the one to
build**: it needs no map predicted in advance and carries one rule for any network, instead of a matrix trained for
each.

**The distance $d$ is an input of the mechanism.** The rules below need a distance between blocks and do not say
which. In what space a query's areas are set, and what makes two weights close, is the first open hole of the
[problem statement](problem-statement.md). Where zones grow from, along what, how far and how fast are strategies to
be tested: [zone-strategies.md](zone-strategies.md).

## Variant A: the map is built before the pass

### Where the zones come from

A source scores how much the query needs each block; the peaks of that score in the metric $d$ are the centers
of the query's expert zones. Their number is not fixed: the query decides it. Which source and which metric -
the variants to test are in [zone-strategies.md](zone-strategies.md).

### The base precision

Everything outside the zones is read at the base level $G$, any rung of the ladder. In the metaphor this is the glass: a coarse base is frosted glass - the whole network still works, coarsely. A ZERO base is an opaque one: outside the zones there is nothing.

ZERO means emptiness - no knowledge, as an expert a mixture of experts did not choose. It is applied only where a zero row is emptiness:

| Module | A zero row | At ZERO base |
| --- | --- | --- |
| `gate_proj`, `up_proj`, `per_layer_input_gate` | the neuron or channel is off, act(0) = 0 | ZERO |
| `v_proj` | the head carries nothing along those rows | ZERO |
| `o_proj`, `down_proj`, `per_layer_projection` | the update to the residual stream is empty along those rows | ZERO |
| `q_proj`, `k_proj` | attention goes flat - a distortion, not emptiness | the attention level $A$ |

The norms after each sub-block rescale the rows left - as the MoE block of Gemma 4 itself does with the experts it did not choose.

### The zones

Each **expert zone** of the query is read above the base ([problem statement](problem-statement.md)): the zones grow from the peaks of the query's score, and their number comes from the query itself.

- **focus_area** sets the reach of all zones at once (rule 1), as a share of the width of the network: 0 lifts nothing at all, 1 reaches everything. What share of the network that covers is measured, not set.
- **focus_strength** sets how far the centers rise above the base (rule 2), counted in rungs of the ladder above it: 0 the zone is the base, 1 the top rung. The scale follows the base: at a D4 base it runs D4 … D8, at ZERO base 0 … D8.

Memory is not set in advance: it is the result of the base, the size and the strength of the zones (rule 7).

### The profile of a zone

From the center outwards the precision falls off along **stops**, as gradient stops in a graphics editor: one stop per level, the outer edge of that level's ring as a share of the radius. Written as `D8:0.33 D6:0.67 D4:1`.

- The default is even (rule 3). A level may be skipped: at base D2, `D8:0.5 D6:1` - the step from D6 to D2 is then a jump, a choice of the profile.
- Every stop above 1 is a ring beyond the zone edge, as many rings as such stops: at ZERO base `D8:0.5 D6:1 D4:1.2 D2:1.5` gives two, D4 out to 1.2 radii and D2 out to 1.5. Such rings may soften the step from the zone into emptiness.
- How fast precision falls off from the center is a strategy of its own; today the fall is linear ([zone-strategies.md](zone-strategies.md)).

### Where zones overlap

Each zone lifts the blocks it covers, continuously, from 1 at its center to 0 at its last stop (rule 4). Overlapping lifts combine by a replaceable strategy (rule 5):

- **sum**, the default: the lifts of overlapping zones add up, capped at 1. The core of a deep overlap reaches the ceiling: a query on the border of two topics gets a sharp junction.
- **max**: the strongest zone only - no gain, and the junction does not fall back to the base.

Both are continuous: at the border of an overlap the second zone adds 0, so the junction rises smoothly. A single zone reads its stepped profile exactly, an overlap lowers no block, and neighbouring blocks differ by at most one rung unless the profile skips one (1D simulation of five layouts of two zones, both strategies).

### The mechanism in formulas

**Given**

| Symbol | What it is | E2B |
| --- | --- | --- |
| $\ell_0 < \ell_1 < \dots < \ell_m$ | the ladder: the levels a block can be read at, set by the model and its storage; $\ell_0$ = ZERO, nothing read | 0, 2, 4, 6, 8 bits (ZERO, D2, D4, D6, D8) |
| $w_b$ | block $b$: its number of weights | |
| $d(a, b)$ | the distance between blocks; the space it is taken in is open ([problem statement](problem-statement.md), hole 1) | not chosen ([zone-strategies.md](zone-strategies.md)) |
| $D$ | the width of the network in $d$ | |
| $c_i$ | the centers of the query's expert zones ([where they come from](#where-the-zones-come-from)) | |

**Controls**

| Control | Symbol | Range | Default |
| --- | --- | --- | --- |
| base precision - the level everything outside the zones is read at; `floor` in the scripts and the runs | $G = \ell_\gamma$ | any rung | - |
| focus_area - the reach of a zone, as a share of the width of the network | $f$ | $[0, 1]$ | - |
| focus_strength - how far the zone centers rise above the base | $g$ | $[0, 1]$ | - |
| profile - where each ring of a zone ends | stops $s_j$ | $s > 0$, strictly growing; a stop above 1 is a ring past the edge | even, below |
| attention level - `q_proj`, `k_proj` outside the zones at ZERO base | $A = \ell_\alpha$ | $\alpha \ge 1$ | the lowest non-zero rung |
| combining overlapping zones | $\oplus$ | sum, max | sum |

**The rules**

1. Zone radius: $R = f \cdot D$. At $f = 0$ the zone is empty - nothing rises above the base, its center included - at $f = 1$ it reaches the whole network. The share of the network a zone covers is not $f$ and does not grow with it linearly: a ball on the block graph gains its whole boundary at every hop ([math, section 8](bench-math.md#8-the-shape-of-a-map-what-is-measured)), and the measured shares are in [zone-strategies.md](zone-strategies.md).
2. Ceiling: $\kappa = \gamma + \lfloor g\,(m - \gamma) \rfloor$. At $g = 0$, $\kappa = \gamma$ and the zone is the base; at $g = 1$, $\kappa = m$.
3. Profile: the levels $\ell_\kappa, \dots, \ell_{\gamma+1}$ end at stops $s_\kappa < \dots < s_{\gamma+1}$; by default they are even, the last at the edge:
   $$s_j = \frac{\kappa - j + 1}{\kappa - \gamma}.$$
4. Lift of a zone, linear for now:
   $$\rho_i(b) = \frac{d(b, c_i)}{R}, \qquad L_i(b) = \max\Big(0,\ 1 - \frac{\rho_i(b)}{s_\text{last}}\Big).$$
5. Combining: $L(b) = \min\big(1, \sum_i L_i(b)\big)$ (sum) or $L(b) = \max_i L_i(b)$ (max).
6. Level of a block: outside every zone ($\rho_i(b) > s_\text{last}$ for all $i$) it is $G$, and `q_proj`, `k_proj` at a ZERO base are at $A$; inside, with $\rho^* = s_\text{last}\,(1 - L(b))$, it is the level $\ell_j$ with the smallest $s_j \ge \rho^*$.
7. Memory:
   $$\text{bits} = \frac{\sum_b w_b \cdot \text{bits}(\text{level of } b)}{\sum_b w_b}.$$

The derivations behind these rules - where they come from, what they guarantee and how close they are to the best use of the memory - are in [the mathematics of the filter](bench-math.md).

**Worked examples** on the E2B ladder:

| Base | g | Ceiling (2) | Profile (3) | Beyond the zone |
| --- | --- | --- | --- | --- |
| D4 ($\gamma = 2$) | 1 | $\kappa = 4$, D8 | `D8:0.5 D6:1` | D4 |
| D4 | 0.5 | $\kappa = 3$, D6 | `D6:1` | D4 |
| D2 ($\gamma = 1$) | 1 | $\kappa = 4$, D8 | `D8:0.33 D6:0.67 D4:1` | D2 |
| ZERO ($\gamma = 0$) | 1 | $\kappa = 4$, D8 | `D8:0.25 D6:0.5 D4:0.75 D2:1` | ZERO |
| ZERO | 0.5 | $\kappa = 2$, D4 | `D4:0.5 D2:1` | ZERO |
| any | 0 | $\kappa = \gamma$ | none | the base everywhere |

**What the names mean.** `D` is depth, counted in 2-bit steps of the one stored copy, and the number
is the bits it comes to. The copy is a k-quant base after llama.cpp - Q2_K, one step, or Q4_K, two
steps for the sensitive module classes - with refinements over it, each quantizing what the steps
before it left with a step four times finer (`foqlens.refinements.KRefinedWeight`, E002). Every level is
the same copy, read to a different depth. A module on a Q4_K base reads the same
at D2 and D4, so base precision D2 comes to about 3.3 bits per weight on E2B.

The base may also be the blocks of a published k-quant file as they lie in it; since E003 the bench's base is
bartowski's Q2_K, calibrated with an imatrix.

`D8` is the top rung of the zones: on the frozen corpus it keeps 98.8% of bf16's knowledge over the bench's own base
at 8.57 bits per controlled weight (E002), and 97.5% over bartowski's (E003). The model's file may hold above the refinements an exact tail to the source weights in
their own type - bf16, fp16 or fp32 ([the model format](refocustensors.md)). The model then reads back the source
bit for bit, the bench runs without the checkpoint, and the format does not depend on the type a model is
published in; the tail also gives the judge its bf16 reference. A zone can be raised to the source as well, but on
E2B-it that is 15.05 bits per weight against 8.57 at D8 for 1.2 points of knowledge.

No rung is fixed in the rules: every level follows from the ladder and the controls. On E2B naive
round-to-nearest at two bits breaks the model ([E001](../experiments/E001-uniform-quantization/results.md)); the
k-quant copy keeps 51.4% of bf16's knowledge at D2 ([E002](../experiments/E002-base-precision-d2/results.md)), over
bartowski's calibrated base 76.7% ([E003](../experiments/E003-calibrated-base/results.md)),
so D2 serves as a base precision and as the ring pushed past a zone edge.

## Variant B: deciding as the pass runs (the one to build)

It is built and runs on the card: the scores, the levels and the count of what has been lifted stay on the device,
and the step is captured into a graph whole, decision included. The speed holds - 39.6 ms at a batch of 32 against
31.5 for a uniform mixed layout. Inference and engineering are done here; what is open is the formula itself - the
one by which a layer decides whom to lift.

The regulator decides a layer's precision during the pass itself, from what enters that layer, instead of
building the whole map before the first one. The pass is single: the network is read once and the decision
travels with it. A second pass is the agent's business ([H4](hypotheses.md)); the regulator has none, or
inference doubles.

The map it produces is the same object as in variant A: a level a block, the base everywhere else. What
changes is where it comes from.

### The rule

At layer $k$ the pass holds the state entering it. Three sources of a score for what lies ahead, each computable in
the pass:

1. **The activity** of the units of the layer itself - the norm of what enters them ([the address](bench-math.md#4-the-address-signal-background-excess)).
2. **The votes**: a unit votes for the units it feeds, the weight of a vote is the coupling read from the weights
   ([the signal's path](bench-math.md#7-the-distance-between-blocks), built once per model and kept), the
   strength of a vote is the voter's activity now. A voter's vote is given to the share `focus_area` of its own edges,
   the strongest first: at 0 it does not vote at all, at 1 it votes for everything it feeds.
3. **The draft**: the next layer read at the base first, its output telling which of its blocks carry the question;
   those blocks are read again at a higher level and the layer is recomputed for them alone.

The score is turned into levels by the mapper of the filter: the price of memory (or a share) decides how many rise,
the ceiling decides how high.

### The mechanics in formulas

At layer $k$ the pass holds $h_k$, the state entering it. A unit $c$ of what comes next gets a score $s_c$, and the
score is read into a level by the rule of [section 9](bench-math.md#9-the-best-allocation-knapsack-and-lagrangian-new):
with the reduced sensitivity $\hat s_c = s_c \Delta_c^2 / w_c$ ($\Delta_c$ the block's quantization step, known offline,
$w_c$ its weights),

$$\text{level}(c) = \gamma + \#\{\, j \ge 1 : \hat s_c \ge \lambda\, 16^{\,j} \,\},$$

capped at the ceiling $\kappa = \gamma + \lfloor g (m - \gamma) \rfloor$. With a share $f$ in place of a price, the
threshold is the quantile: the top $f$ of the layer by $\hat s$ rise, the next band of width `rim` is held at
$\gamma + 1$, the rest stays at the base.

The three sources differ only in $s_c$:

$$s^{\text{activity}}_c = \lVert x_c \rVert^2, \qquad
s^{\text{votes}}_c = \sum_{b \,\to\, c} a_b\, \kappa(b \to c), \qquad
s^{\text{draft}}_c = \lVert y_c^{\text{base}} \rVert^2 \Delta_c^2 .$$

$x_c$ is what enters the unit, $a_b$ the activity of a voter, $\kappa(b \to c)$ the coupling of the signal's path
(built once per model), $y_c^{\text{base}}$ the unit's output in the draft at the base. The votes are given to the
share `focus_area` of a voter's edges, the strongest first.

*What the input of a layer can and cannot tell.* Inside a layer `gate_proj`, `up_proj` and `q_proj`, `k_proj`,
`v_proj` are all given the same vector - the state that entered the layer. Their blocks therefore differ in
$\lVert x_c \rVert$ by nothing at all; what is left between them is the norms of their own rows, which is a property of
the model and not of the question. A score from the input of a layer separates the blocks of `down_proj` (its input is
the neurons already computed) and of `o_proj` (the heads' outputs), and no others. The activity is therefore the floor
of the three sources; the votes are the only one that looks ahead per question, and the draft is the only other one
that sees a block's own output, at the price of reading the layer twice.

*What carries over from the filter's mathematics.* The forward half of the sensitivity of
[section 3](bench-math.md#3-what-a-coarse-block-costs-the-output-new) is exactly what these three
measure; the backward half - how much the loss listens to the unit - is not in the pass. Variant A carries it by a
projection fitted on calibration questions; a layer-wise regulator either goes without it, or carries a fixed per-class
prior measured once per model, which costs no pass.

*Where the guarantee breaks.* A price is optimal over the whole network when the scores of every layer are on one
scale; a score read layer by layer is not, so either the price is calibrated per layer once on calibration questions,
or a share per layer is used, which fixes the memory but not the gain. What no layer-wise rule has is a look-ahead:
the decision at $k$ cannot know what the layers after it will need.

### The controls

| Control | What it sets | The ends |
| --- | --- | --- |
| price of memory $\lambda$, or the share $f$ | how much of the layer rises | $f = 0$: everything at the base; $f = 1$: the whole layer rises |
| ceiling $g$ | how high it rises, in rungs over the base | $g = 0$: the base; $g = 1$: the top rung |
| the rim | the band under the risen part, held one rung above the base | 0: the risen part ends on the base; wide: the whole layer is held above the base |

The profile of the zones - the stops of [rule 3](precision-regulator.md#the-mechanism-in-formulas) - has no place
here: with no centre and no distance, the level follows from the score against the price, and the rungs of the ladder
already lie a factor of 16 apart in sensitivity
([section 9](bench-math.md#9-the-best-allocation-knapsack-and-lagrangian-new)).

**The rim** matters where the base is ZERO. A block read sharp whose input arrives through blocks that are not read at
all is given noise, and its precision is spent on nothing; the rim is what carries the signal into the sharp part and
out of it. Its width is measured, not set: two layouts of the same bytes, with the rim and without, answered by
generation.
