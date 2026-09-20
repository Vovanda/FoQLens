---
title: The layer-wise regulator
---

# The layer-wise regulator

A regulator that decides the precision of a layer while the pass is already running, from what that layer is given,
instead of deciding the whole map before the first layer. One pass: the network is read once, and the decision travels
with it. The second pass belongs to the agent ([H4](hypotheses.md)), never to the regulator - it would double the
inference.

The map it produces is the same object the [filter](quantization-filter.md) produces: a level per block, the base
everywhere else. What changes is where the map comes from.

## The rule

At layer $k$ the pass holds the state entering it. Three sources of a score for what lies ahead, each computable in
the pass:

1. **The activity** of the units of the layer itself - the norm of what enters them ([the address](quantization-filter-math.md#4-the-address-signal-background-excess)).
2. **The votes**: a unit votes for the units it feeds, the weight of a vote is the coupling read from the weights
   ([the signal's path](quantization-filter-math.md#7-the-distance-between-blocks), built once per model and kept), the
   strength of a vote is the voter's activity now. A voter's vote is given to the share `focus_area` of its own edges,
   the strongest first: at 0 it does not vote at all, at 1 it votes for everything it feeds.
3. **The draft**: the next layer read at the base first, its output telling which of its blocks carry the question;
   those blocks are read again at a higher level and the layer is recomputed for them alone.

The score is turned into levels by the mapper of the filter: the price of memory (or a share) decides how many rise,
the ceiling decides how high.

## The mechanics in formulas

At layer $k$ the pass holds $h_k$, the state entering it. A unit $c$ of what comes next gets a score $s_c$, and the
score is read into a level by the rule of [section 9](quantization-filter-math.md#9-the-best-allocation-knapsack-and-lagrangian-new):
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
[section 3](quantization-filter-math.md#3-what-a-coarse-block-costs-the-output-new) is exactly what these three
measure; the backward half - how much the loss listens to the unit - is not in the pass. Variant A carries it by a
projection fitted on calibration questions; a layer-wise regulator either goes without it, or carries a fixed per-class
prior measured once per model, which costs no pass.

*Where the guarantee breaks.* A price is optimal over the whole network when the scores of every layer are on one
scale; a score read layer by layer is not, so either the price is calibrated per layer once on calibration questions,
or a share per layer is used, which fixes the memory but not the gain. What no layer-wise rule has is a look-ahead:
the decision at $k$ cannot know what the layers after it will need.

## The variants

- **The source**: activity, votes, draft, or a sum of them.
- **The granularity**: a block, a module of a layer, a whole layer.
- **The moment**: one layer ahead, a window of several layers, a few checkpoints along the depth.
- **The threshold**: one price for the network, a price per layer, a share per layer.
- **Revising**: raising only, or lowering again what was raised earlier when the signal fades.
- **Where it is computed**: on the card inside the captured graph, or on the host between its segments.

## The controls

| Control | What it sets | The ends |
| --- | --- | --- |
| price of memory $\lambda$, or the share $f$ | how much of the layer rises | $f = 0$: everything at the base; $f = 1$: the whole layer rises |
| ceiling $g$ | how high it rises, in rungs over the base | $g = 0$: the base; $g = 1$: the top rung |
| the rim | the band under the risen part, held one rung above the base | 0: the risen part ends on the base; wide: the whole layer is held above the base |

The profile of the zones - the stops of [rule 3](quantization-filter.md#the-mechanism-in-formulas) - has no place
here: with no centre and no distance, the level follows from the score against the price, and the rungs of the ladder
already lie a factor of 16 apart in sensitivity
([section 9](quantization-filter-math.md#9-the-best-allocation-knapsack-and-lagrangian-new)).

**The rim** matters where the base is ZERO. A block read sharp whose input arrives through blocks that are not read at
all is given noise, and its precision is spent on nothing; the rim is what carries the signal into the sharp part and
out of it. Its width is measured, not set: two layouts of the same bytes, with the rim and without, answered by
generation.

## What it costs

- The score has to be computed where the state is, on the card, or the pass pays a synchronization per layer.
- The codes of the levels have to be written into a tensor the kernel reads, inside the graph the pass is captured in.
  The bench writes a layout once per question today; per layer inside a graph is not done yet.
- The draft (source 3) reads a layer twice and recomputes a part of it. On a decoding step the bytes read are the
  narrow place, so a draft at the base plus a refinement of a small share can still read fewer bytes than the whole
  layer at the top rung.

## What decides between the sources

Measured on the oracles' maps, without generation: does the score of a source pick the groups the map raises. Measured
by generation: the answer at the layout against the answer of the whole network at the top rung, and the bytes, on the
questions the address was not fitted on.

*Measured 2026-09-20, the small corpus's first part, at the granularity of groups:* the activity inside a layer does
not tell which blocks of that layer the oracles raise (rank -0.09 to +0.05 by depth band), while the choice of layers
does carry. The votes and the draft are not measured yet.
