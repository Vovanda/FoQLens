---
title: The mathematics of the quantization filter
---

# The mathematics of the quantization filter

The whole mechanism in one derivation, from a block of weights to the zones of a query: what the object is, what precision costs, where the address comes from, how far two blocks are from each other, how zones are built, why they approximate the best use of the memory, and what a batch pays. Each step rests on the one before it. The rules this page derives are in [quantization-filter.md](quantization-filter.md); the strategies of source, metric and reach are in [zone-strategies.md](zone-strategies.md); the format the levels are read from is in [refocustensors.md](refocustensors.md). Sections marked *new* are derived here first; the rest is gathered from those pages, the code of the bench and the measurements of 2026-09-19. *Proved* means derived from the definitions or the code; *measured* is a number from a run; *hypothesis* is not checked.

Order: [0 the task](#0-the-task-before-any-mechanism) → [1 the object](#1-the-object) → [2 the error of a level](#2-the-error-of-a-level) → [3 what a coarse block costs](#3-what-a-coarse-block-costs-the-output-new) → [4 the address](#4-the-address-signal-background-excess) → [5 the working address](#5-the-working-address-from-the-first-layers-to-the-rest) → [6 the depth of the reading](#6-the-depth-of-the-reading-closed) → [7 distance](#7-the-distance-between-blocks) → [8 the shape of a map](#9-the-shape-of-a-map) → [10 the best allocation](#10-the-best-allocation-knapsack-and-lagrangian-new) → [11 the batch](#11-the-batch) → [12 known and open](#12-what-is-known-and-what-is-being-worked-out).

## 0. The task, before any mechanism

Knowledge is spread over the whole network, not kept in a region of it. A question does not live in one place, and a
layout that reads blocks from the first layer to the last is what a question needs, not a defect to remove.

The ends are known: every block at D2 answers badly and incompletely (E003: 76.7% of bf16's knowledge over bartowski's
base), every block at D8 answers fully (97.5%). Between them lie layouts where some blocks read D8, some D4, the rest
D2. The filter has one job: for a question, decide - deterministically, from the question - which blocks are worth
reading higher, so that the answer gains at a given memory. Nothing in it is random.

What decides a block's level is how much the answer needs it: its sensitivity for this question ([section 3](#3-what-a-coarse-block-costs-the-output-new)). With the sensitivity known, the best layout at a price of memory is thresholds on it, a rung for every 16 times ([section 10](#10-the-best-allocation-knapsack-and-lagrangian-new)). So the task splits into two questions, each a measurement:

1. **Existence.** Does the layout of the gradient's Taylor score - the best estimate of the sensitivity the bench has, the oracle - answer better than uniform quantization at the same memory? If it does, such layouts exist. If it does not, that refutes the estimate and its assumptions (a second-order loss with the Fisher for the Hessian, blocks whose errors add up independently, the 16 times per rung of the noise model, the loss of the prompt rather than the answer), not the existence of a better layout.
2. **Prediction.** How much of the oracle's gain does the working address, read in a forward pass of the first layers, recover? That is how well the address stands for the sensitivity.

Zones ([section 9](#9-the-shape-of-a-map)) are one way to write a layout: a smoothness prior on the block graph, worth it only where the address is noisy per block and the sensitivity smooth. Their shape - a ball, a branch, a layer - is secondary; what is compared is the quality of answers at the same bytes.

## 1. The object

- The model has $L$ layers (E2B: 35). Its controlled weights are cut into blocks of 64 output rows of a module: block $b$, its layer $\ell(b)$, its number of weights $w_b$. E2B has $|B| \approx 14\,708$ blocks.
- The ladder $\ell_0 < \ell_1 < \dots < \ell_m$: ZERO, D2, D4, D6, D8 (0, 2, 4, 6, 8 bits); above it, the exact tail to the source. A level is how deep the one copy is read.
- A block at level $\ell$ costs $c_b(\ell) = w_b \cdot \text{bits}(\ell) / 8$ bytes. The memory of a layout ([rule 7](quantization-filter.md#the-mechanism-in-formulas)) is $\sum_b w_b\, \text{bits}(\ell(b)) / \sum_b w_b$ bits per weight.
- The base $G = \ell_\gamma$ is the level of everything outside the zones; today D2 over bartowski's Q2_K ([E003](../experiments/E003-calibrated-base/results.md)).

The task of the filter: for a query $q$, choose every block's level so that the answer is as close as possible to the source model's at a given memory.

## 2. The error of a level

**Refinements** ([refinements.py](../src/foqlens/refinements.py), format: [refocustensors.md](refocustensors.md)). The k-quant base gives a block the step $\Delta_b = d \cdot \text{scale}$. Refinement $k$ quantizes what the steps before it left with a 2-bit symmetric code of step $\Delta_b / 4^k$.

*Proved (an invariant of the code, with its test):* a weight whose base error is within $\Delta_b / 2$ is within $\Delta_b / (2 \cdot 4^k)$ of the source after $k$ refinements.

**Quantization noise.** In the standard model (Widrow) the error of rounding with step $\Delta$ is uniform noise on $[-\Delta/2, \Delta/2]$, independent of the weight, of variance $\Delta^2/12$. For refinement $k$:

$$\sigma_k^2 = \frac{\Delta_b^2}{12 \cdot 16^k}.$$

Every rung of the ladder (2 bits) divides the variance of the error by 16 - the same 6 dB per bit as any uniform quantizer. The model is an approximation: on an imatrix base the error is not uniform; for the refinements, which quantize an already small remainder, it holds better.

## 3. What a coarse block costs the output (new)

Block $b$ is the rows $r$ of a module with input $x$. An error $\Delta W$ of its weights gives the output error $\delta y_r = \sum_j \Delta W_{rj}\, x_j$. The loss $\mathcal{L}$ to second order in the block's outputs:

$$\Delta \mathcal{L} \approx g^\top \delta y + \tfrac12\, \delta y^\top H\, \delta y, \qquad g = \frac{\partial \mathcal{L}}{\partial y},\quad H = \frac{\partial^2 \mathcal{L}}{\partial y^2}.$$

Quantization noise has zero mean and does not depend on $x$, so the first term vanishes in expectation:

$$\mathbb{E}[\Delta \mathcal{L}] \approx \tfrac12 \operatorname{tr}(H\, \Sigma_\delta), \qquad \Sigma_\delta = \mathbb{E}[\delta y\, \delta y^\top].$$

With independent weight errors of variance $\sigma^2$, every row of the block gets $\mathbb{E}[\delta y_r^2] = \sigma^2 \lVert x \rVert^2$. Replacing the Hessian by the Fisher information ($H \approx \mathbb{E}[g g^\top]$, as in OBS and HAWQ) gives the block's share:

$$\mathbb{E}[\Delta \mathcal{L}_b(\ell)] \approx \tfrac12\, \sigma_\ell^2(b)\, s_b, \qquad s_b = \mathbb{E}_t\Big[\sum_{r \in b} g_r^2\, \lVert x_t \rVert^2\Big].$$

$s_b$ is the **sensitivity** of the block: how much the loss listens to its outputs ($g^2$) times how strong a signal goes through it ($\lVert x \rVert^2$). It depends on the query, and it is what the filter has to know.

**The gain of a lift.** Lifting a block from $\ell$ to $\ell'$ removes

$$u_b(\ell \to \ell') = \tfrac12\, s_b\, (\sigma_\ell^2 - \sigma_{\ell'}^2).$$

By [section 2](#2-the-error-of-a-level) the variance falls 16 times per rung: the first rung above the base removes $15/16$ of the block's error, the second $15/256$, the third $15/4096$.

*Proved from the noise model:* the gain per byte of every next rung of the same block is 16 times smaller than the one before (a rung costs the same 2 bits per weight).

*Consequence (a hypothesis for the answers to check):* at one memory, lifting a second block by one rung beats lifting the first block by a second rung while their sensitivities differ less than 16 times. Wide low zones should beat narrow high ones; high ceilings pay only in blocks 16 times more sensitive than the background.

**What our signals see of it.** $s_b$ needs the gradient - a backward pass. The forward pass gives the second factor, $\lVert x \rVert^2$: neuron activity (the norm of the `down_proj` input per group) and head energy (the norm of a head's output) are the forward halves of the Taylor score (GRIFFIN; Michel et al. 2019). The gradient half is not visible to the working address; the projection of [section 5](#5-the-working-address-from-the-first-layers-to-the-rest), fitted on calibration questions, carries it. The signals are in [activity.py](../src/foqlens/activity.py), the sources by name in `pipeline.ADDRESS_SOURCES`. *Measured 2026-09-19:* the full gradient signal is not a stable address (identification across wrappers 0.47-0.49); the forward halves are (0.96-1.00).

## 4. The address: signal, background, excess

A source gives a mask $a(q) \in \mathbb{R}^B$. What every question shares - the template, frequent tokens, blocks loud on any input - is removed with the mean of the same reading:

$$\tilde a(q) = a(q) - \bar a, \qquad \bar a = \text{the mean of } a \text{ over the questions of the same reading}.$$

The measures of this section and the next two are in [address.py](../src/foqlens/address.py).

**Identification** measures that the excess names the question: for two readings of the same $n$ questions, the share whose nearest excess by cosine in the other reading is its own, both ways; chance is $1/n$.

**The hybrid.** Signals over different blocks are summed after each is scaled to length 1 per question:

$$a_\text{hybrid} = \frac{\hat a_\text{neuron}}{\lVert \hat a_\text{neuron} \rVert} + \frac{\hat a_\text{head}}{\lVert \hat a_\text{head} \rVert}.$$

The signals measure different things in different units, and a sum of raw values would be the loudest signal alone. A part with no signal on a question adds 0.

**Agreement** of two signals over different blocks is the correlation of their question-by-question cosine matrices off the diagonal (representational similarity).

*Measured 2026-09-19*, base bartowski, 206/50/306 questions of TriviaQA/NQ/SQuAD:

| signal | across wrappers | paraphrase (words alone: 0.717) |
| --- | --- | --- |
| hybrid | 0.997-1.000 | 0.917 |
| neuron_activity | 0.995-1.000 | 0.892 |
| head_energy | 0.964-0.993 | 0.692 |
| pooled | 0.820-0.886 | 0.308 |
| gradient | 0.473-0.490 | - |

Agreement of neuron and head activity is 0.73, of the hybrid and neuron activity 0.96.

**What the literature says of the sources.**
- A block's Taylor score in the form $\lvert \sum \rvert$ (the magnitude of a sum over 64 rows and the tokens) ranks importance poorly: the terms of a group cancel (Molchanov et al. 2019). The bench therefore computes it in two forms from one backward pass, $\lvert \sum \rvert$ and $\sum \lvert \cdot \rvert$ (`gradient`, `gradient_magnitude`).
- Neuron activity is the forward half of Taylor for `up_proj`: with $h = \varphi(\text{gate}) \odot \text{up}$, the Taylor score of an up block is $\lvert \sum \partial \mathcal{L}/\partial h \cdot h \rvert$, and the norm of $h$ is its factor without the gradient. GRIFFIN (arXiv 2404.01365) selects neurons per query by this quantity and keeps quality at 50% MLP sparsity.
- Head energy is the forward half of a head's importance $I_h = \mathbb{E}\lvert \text{Att}_h^\top\, \partial \mathcal{L}/\partial \text{Att}_h \rvert$ (Michel et al. 2019). The norm of a head's output is a stable sign of its activity; metrics on the attention weights catch fewer than half of the dormant heads (arXiv 2504.03889).
- Head entropy is a weak source: low entropy and high gradient importance do not coincide (HIES, arXiv 2510.13832); the attention sink takes more than half of the attention from the second layer on whatever the query; entropy depends on the length.
- The Fisher information of a block on one query is the square of the signed Taylor score, the same order of blocks; the trace of the Hessian adds 0.03 Spearman for two backward passes and more (Molchanov et al. 2019).
- Carrying an early signal onto every block by regression: one predictor on the first layer recovers the Taylor score of every head and neuron of all layers at Spearman about 0.85 (ShadowLLM, arXiv 2406.16635) - the ground of the projection of [section 5](#5-the-working-address-from-the-first-layers-to-the-rest).

## 5. The working address: from the first layers to the rest

The zones act in the layers after the working ones, so the address is needed there, and it is read from the first layers at the base.

- The working reading of depth $n$: $x_n(q)$, the source's mask over the layers $\ell < n$ at the base precision.
- The target: the address $a_T(q)$ on the blocks $T = \{b : \ell(b) \ge m\}$ at bf16.

**The projection** (ridge, [projection.py](../src/foqlens/projection.py)), over centred calibration readings $X_n$ and targets $A_T$:

$$\hat a^{(n)}(q) = \mu_T + (x_n(q) - \mu_x)\, P_n, \qquad P_n = (X_n^\top X_n + \alpha I)^{-1} X_n^\top A_T,$$

$\alpha = \rho \cdot (\text{the mean variance of a read block}) \cdot (\text{the number of calibration questions})$; $\rho$ is the relative ridge, one value for every depth.

*Proved (the invariants of [projection.py](../src/foqlens/projection.py)):* at $\alpha = 0$ and full rank an exact linear relation is recovered; a larger $\alpha$ never gives a larger $P$.

**The price of depth.** In one pass the layers $\ell < n$ are read at the base with no zones, and the zones act on the share of weights $W(n) = \sum_{\ell(b) \ge n} w_b / \sum_b w_b$. In two passes the zones act everywhere, and the extra prefill at the base is $n/L$ of a pass, once per question.

*Measured 2026-09-19*, the hybrid, $\rho = 0.1$, identification of the real deep address by the predicted one (NQ: 100 questions, 487 for calibration):

| $n$ | TriviaQA | SQuAD | NQ | $W(n)$ |
| --- | --- | --- | --- | --- |
| 1 | 0.680 | 0.824 | 0.740 | 0.98 |
| 4 | 0.830 | 0.877 | 0.875 | 0.92 |
| 6 | 0.893 | 0.869 | 0.860 | 0.88 |
| 8 | 0.893 | 0.877 | 0.860 | 0.84 |
| 12 | 0.905 | 0.891 | 0.855 | 0.76 |

A window of layers $[a, n)$ in place of $[0, n)$ does not raise identification: $[5, 6)$ alone gives 0.65-0.78 against 0.86-0.89 for $[0, 6)$. The early layers carry the address.

## 6. The depth of the reading (closed)

The idea was to read layer by layer and stop once the predicted address had stopped moving, so that the depth would
follow the question instead of the length of the network.

*The measurement of 2026-09-19 closed it.* What is visible at a shallow depth does not say whether a question needs a
deeper one: the area under the shift's curve is 0.42-0.66, and even the oracle's margin, which takes the true deep
address, gives 0.53-0.70. A silhouette rule - stop once the zones hold over several depths - is no better by
identification than a fixed depth of the same average cost. Thirty trivial questions stop at layer 6.70 on average and
twenty-five hard ones at 6.48: the hard ones are read shallower.

One number is left of the section: an address needs the first 4-8 layers, and that is a fixed
depth ([section 5](#5-the-working-address-from-the-first-layers-to-the-rest)).

## 7. The distance between blocks

A zone needs "near". The rules look only at distances from a center, so the metric is a replaceable part:
[metric.py](../src/foqlens/metric.py), the signal's path in [coupling.py](../src/foqlens/coupling.py).

**Why a ball over the graph jumps.** Let the graph have vertex expansion $h > 0$: every set $A$ of at most half the
blocks has at least $h|A|$ neighbours outside it. The ball $B(c, r)$ takes its whole outer boundary at the next step,
so while it holds at most half the blocks,

$$|B(c, r+1)| \ge (1 + h)\, |B(c, r)|, \qquad |B(c, r)| \ge (1 + h)^r.$$

*Proved* by induction over $r$. The ball reaches half the network in $r_{1/2} \le \log(|B|/2) / \log(1 + h)$ steps: up
to some radius it covers little, and a few steps later most of the network. *Measured* (the pooled address, a
mutual-NICDM graph over 16 neighbours, a TriviaQA smoke over the base D2): the median raised share is 0.005 at
$f = 0.05$, 0.038 at $0.1$ and 0.497 at $0.2$ - the coverage jumps thirteenfold between two neighbouring values of one
knob. The expansion $h$ itself is not measured.

## 8. The oracles: a field of importance and how it is read into a map

An oracle builds the map of one question knowing its answer. It is impossible at inference - each one costs from tens
to thousands of passes over the network - and it is here as a ceiling: it shows which layout exists for this question
at all and what it costs. The code: [group_oracle.py](../src/foqlens/group_oracle.py),
[scoring.py](../src/foqlens/scoring.py), [error_energy.py](../src/foqlens/error_energy.py), and the reading of a field
in [precision_field.py](../src/foqlens/precision_field.py).

The unit of a map here is a **group** $g$: a layer's attention or its MLP, $70$ of them on E2B. A block takes the level
of its group.

**The target.** Every oracle measures the same thing - how much a question needs a group, with the model's own answer
at full precision as the target. Let $A(q)$ be that answer and $\mathrm{NLL}(A \mid \Lambda)$ its negative
log-likelihood under a layout $\Lambda$. Write $\Lambda_\top$ for the whole network at the top rung and $\Lambda_\bot$
for the whole network at the base.

**The sweep by trying: the lift.** The whole network at the base, one group raised:

$$s^{\text{lift}}_g = \mathrm{NLL}(A \mid \Lambda_\bot) - \mathrm{NLL}(A \mid \Lambda_\bot \text{ with } g \text{ at the top}).$$

How much likelihood the raised group gave the answer back. One pass a group plus the two ends: $|G| + 2$ passes a
question.

**The sweep by trying: the drop.** The whole network at the top, one group dropped:

$$s^{\text{drop}}_g = \mathrm{NLL}(A \mid \Lambda_\top \text{ with } g \text{ at the base}) - \mathrm{NLL}(A \mid \Lambda_\top).$$

How much the dropped group spoiled the answer. The lift and the drop measure different things: the first what a group
gives on its own over the base, the second what is missed when every other group is at the top.

**The gradient of the answer.** A backward pass over $\mathrm{NLL}(A)$ under $\Lambda_\top$ gives the Taylor score of
[section 3](#3-what-a-coarse-block-costs-the-output-new), summed over the blocks of a group:

$$s^{\text{gradient}}_g = \sum_{b \in g} \mathbb{E}_t\Big[\sum_{r \in b} g_r^2\, \lVert x_t \rVert^2\Big].$$

One forward and one backward pass a question. The **quantization gap** variant takes the real difference of the weights
between the base and the top instead of the modelled noise: $\delta W = W_\top - W_\bot$, and a block contributes
$\lVert g \odot (\delta W x) \rVert^2$.

**The error energy.** Without a gradient: how much a block's output changes when it is read at the base instead of the
top, over one forward pass:

$$s^{\text{energy}}_g = \sum_{b \in g} \mathbb{E}_t \big\lVert (W_\top - W_\bot)_b\, x_t \big\rVert^2.$$

The D4 variant measures what is left over an already raised network.

**The pooled field.** Every field is brought to unit mass a question and averaged:

$$s^{\text{pooled}}_g = \frac{1}{|O|} \sum_{o \in O} \frac{\max(s^o_g, 0)}{\sum_{g'} \max(s^o_{g'}, 0)}.$$

An oracle that holds no field for a question does not vote on it.

**The reference.** Not an estimate but a measurement rung by rung: with every other group at the top, the group $g$ is
put at each rung $r$, and the coarsest one that holds the answer is taken:

$$\ell^{\text{alone}}_g = \min\{\, r : \mathrm{NLL}(A \mid \Lambda_\top \text{ with } g \text{ at } r) \le \mathrm{NLL}(A \mid \Lambda_\top) + \varepsilon \,\}.$$

Groups that hold alone need not hold together, so the map is raised by whole rungs until it does:

$$\ell^{\text{reference}} = \text{raised}(\ell^{\text{alone}}, k), \qquad k = \min\{\, k : \mathrm{NLL}(A \mid \text{raised}(\ell^{\text{alone}}, k)) \le \mathrm{NLL}(A \mid \Lambda_\top) + \varepsilon \,\}.$$

The cost: $1 + |R| \cdot |G|$ passes a question against $|G| + 2$ for the sweep.

**Reading a field into a map.** Any field of importance becomes levels by one rule. The cost of a group at a rung $r$:

$$c_g(r) = \max(s_g, 0) \cdot e_g(r),$$

where $e_g(r)$ is the share of the base's error the rung leaves, measured from the energies
([section 3](#3-what-a-coarse-block-costs-the-output-new)); $e_g(\text{base}) = 1$. At a threshold $\varepsilon_q$ a
group reads the coarsest rung whose cost fits into it, and the top where none does:

$$\ell_g(\varepsilon_q) = \min\{\, r : c_g(r) \le \varepsilon_q \,\}, \qquad \ell_g = \top \text{ if there is none}.$$

*Proved (the invariant of `precision_field`):* $\ell_g$ is monotone in $\varepsilon_q$ - a larger threshold never reads
a group finer.

**The question's threshold.** One $\varepsilon_q$ a question, searched by the answer itself: the thresholds at which
the map changes (the distinct values of $c_g(r)$) are swept, and the largest whose map holds the answer is taken:

$$\varepsilon_q = \max\{\, \varepsilon : \mathrm{NLL}(A \mid \ell(\varepsilon)) \le \mathrm{NLL}(A \mid \Lambda_\top) + \tau_q \,\}.$$

The search is a bisection over the sorted thresholds, in batches of $w$ layouts beside $\Lambda_\top$ in one batch, so
that the comparison runs under the same conditions. The question's tolerance is
$\tau_q = \max(\tau, \rho \cdot \mathrm{NLL}(A \mid \Lambda_\top))$: flat nats are unreachable where the whole network
is itself unsure, and the oracle then raises everything.

**The shape of the result.** It comes out in one of three shapes, and two of them are not zones: the whole network at
the base - the question holds without precision; the whole network at the top - no threshold was found; and a map
proper. Averages over them are taken apart ([field_analysis.shapes](../src/foqlens/field_analysis.py)).

## 9. The shape of a map

A query's map is the level of every group or block. How it is to be outlined by rules - what it grows along, how far,
with what profile - **is being worked out**: rules with a radius and a profile do not fit the maps that were read, and
the measurements are in [E005](../experiments/E005-precision-map/results.md).

What of the shape enters the formulas today:

- the base $G$ is the level of everything the map does not raise ([section 1](#1-the-object));
- a group's level is read from the field directly ([section 10](#10-the-best-allocation-knapsack-and-lagrangian-new)),
  with no geometry in it;
- the rung's coefficient $e_g(r)$ - how much the rung $r$ cuts a group's error against the base - is measured on the
  energies and enters the cost of a group ([section 3](#3-what-a-coarse-block-costs-the-output-new)).

## 10. The best allocation: knapsack and Lagrangian (new)

**The problem.** Choose every block's level:

$$\min \sum_b \tfrac12\, s_b\, \sigma^2_{\ell(b)}(b) \quad \text{subject to} \quad \sum_b c_b(\ell(b)) \le M.$$

It is a multiple-choice knapsack: every block takes exactly one level.

**The Lagrangian.** A price of memory $\lambda \ge 0$ moves the constraint into the objective:

$$\Lambda(\lambda) = \sum_b \min_\ell \big[\tfrac12\, s_b\, \sigma^2_\ell(b) + \lambda\, c_b(\ell)\big] - \lambda M.$$

The problem splits into independent decisions per block: a block goes up a rung when the gain pays for it,

$$u_b(\ell \to \ell^+) \ge \lambda \big(c_b(\ell^+) - c_b(\ell)\big).$$

*Proved (standard duality):* the solution at every $\lambda$ is optimal for the memory $M(\lambda)$ it spends; $M(\lambda)$ does not increase with $\lambda$. The Lagrangian bound equals the optimum of the linear relaxation; in a multiple-choice knapsack at most one class is fractional at that optimum (Sinha & Zoltners 1979), so the gap to the integer optimum is at most the gain of one block.

**Thresholds per rung.** By [section 3](#3-what-a-coarse-block-costs-the-output-new) the gain of rung $k$ above the base is $\tfrac12 s_b \Delta_b^2 \cdot 15 / (12 \cdot 16^k)$ and its cost $w_b \cdot 2/8$ bytes. With the reduced sensitivity $\hat s_b = s_b \Delta_b^2 / w_b$ the condition becomes $\hat s_b \ge \lambda' \cdot 16^k$, $\lambda' = \frac{24}{15} \cdot \frac28 \cdot \lambda = \frac{2\lambda}{5}$, one constant for every block. The levels are superlevel sets of the reduced sensitivity:

$$\text{level}(b) = \gamma + \bigl|\{\, k \ge 1 : \hat s_b \ge \lambda'\, 16^k \,\}\bigr|.$$

**The link to zones.** If $\hat s_b$ is smooth on the block graph - neighbours alike in sensitivity - the sets $\{b : \hat s_b \ge t\}$ are hills around peaks, and the nested thresholds $t, 16t, 256t$ are nested rings. The zones of [section 9](#9-the-shape-of-a-map) are a parametric approximation of this optimum: peaks are the centers of the superlevel sets, the radius their size, the profile their nesting. $f$ plays the price of memory: a larger $f$ is a smaller $\lambda'$ and wider sets.

Consequences to check:
1. The rings should lie 16 times apart in reduced sensitivity per rung; the even profile of rule 3 does not know this. A profile from the measured $\hat s_b$ (or its proxy) is a direct candidate.
2. Wide low zones against narrow high ones - [section 3](#3-what-a-coarse-block-costs-the-output-new).
3. The per-block threshold (the per-block control) is the Lagrangian solution without smoothness; zones beat it only if the address is noisy per block and smoothness on the graph averages the noise out.

**The per-block quantile regulator is a preset budget.** The per-block control lifts the blocks above a quantile: $P_f = \mathrm{Quantile}(\sigma, 1 - f)$, $L_b = \max(0, \sigma_b - P_f) / \max_b(\sigma_b - P_f)$. *Proved:* the number of blocks above the base, $\bigl|\{b : \sigma_b > P_f\}\bigr| = f |B|$ (up to ties), whatever $\sigma$ is, so its memory is set in advance, while the zones' memory follows from the query ([rule 7](quantization-filter.md#the-mechanism-in-formulas)). The share counts blocks, not weights, and blocks differ in their number of inputs, so its $f$ is not even a share of memory. Normalizing by the maximum lifts the strongest block of any query to the ceiling, so it has no strength of its own per zone (H4). Its role on the bench is the control: what the connectedness of zones adds over a per-block score.

## 11. The batch

At a decoding step a batch reads every block at the highest level among its questions:

$$C(S) = \sum_b w_b \cdot \text{bits}\big(\max_{q \in S} \ell_q(b)\big) / 8.$$

*Proved:* $C(S)$ is monotone in $S$ and submodular - a question adds no more to a larger batch than to a smaller one (a maximum over a set is a coverage function per block, and a sum of submodular functions is submodular). Consequence: questions with close zones are cheaper in one batch; choosing a batch is a covering problem. *Measured, smoke of 2026-09-19* at $f = 0.2$, bytes per question against per step of the batch: per block 760 / 937 MB, static zones 1126 / 1590, the signal's path 946 / 1508; uniform D2 677, D8 1978.

Sharing a batch's cost between its questions (the Shapley value) and a budget between the requests of a pool are game theory, for the stage of serving many requests.

## 12. What is known and what is being worked out

The numbers of the runs are in the results of the experiments: the address in
[E004](../experiments/E004-question-address/results.md), the query's map in
[E005](../experiments/E005-precision-map/results.md).

**Enters the formulas:**

- the rungs' coefficients are measured rather than taken from the noise model, which would give 1/16 a rung
  ([section 3](#3-what-a-coarse-block-costs-the-output-new));
- the working address is the forward half of the sensitivity; the backward half is carried by the projection of the
  first layers ([section 5](#5-the-working-address-from-the-first-layers-to-the-rest));
- the level at a given price of memory is a set of thresholds on the sensitivity
  ([section 10](#10-the-best-allocation-knapsack-and-lagrangian-new));
- the cost of a batch is submodular over the set of queries ([section 11](#11-the-batch)).

**Being worked out, no exact description yet:**

- the outline of a map by rules: what an area grows along, how far, with what profile ([section 9](#9-the-shape-of-a-map));
- a cheap predictor of the map: how much of it the address of the first layers returns, and how much a decision taken
  as the pass runs;
- the statement over a budget instead of a tolerance: today the threshold is searched by the answer rather than by a
  given memory;
- the bounds of the scale of the rungs and the tolerance - set, not derived;
- the smoothness of the sensitivity over the block graph - the condition under which connected areas are near the
  optimum.

**Implemented and not checked by answers** (`foqlens.strategies`, `foqlens.graph_zones`, `foqlens.zones`): the reaches
`EqualReach`, `ProportionalReach`, `LogReach`; the medium and the signal's path as the metric an area grows along; the
strength of a zone through `zone_ceilings`; the profile through `even_stops`.
