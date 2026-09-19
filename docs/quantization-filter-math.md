---
title: The mathematics of the quantization filter
---

# The mathematics of the quantization filter

The whole mechanism in one derivation, from a block of weights to the zones of a query: what the object is, what precision costs, where the address comes from, how far two blocks are from each other, how zones are built, why they approximate the best use of the memory, and what a batch pays. Each step rests on the one before it. The rules this page derives are in [quantization-filter.md](quantization-filter.md); the strategies of source, metric and reach are in [zone-strategies.md](zone-strategies.md); the format the levels are read from is in [refocustensors.md](refocustensors.md). Sections marked *new* are derived here first; the rest is gathered from those pages, the code of the bench and the measurements of 2026-09-19. *Proved* means derived from the definitions or the code; *measured* is a number from a run; *hypothesis* is not checked.

Order: [1 the object](#1-the-object) → [2 the error of a level](#2-the-error-of-a-level) → [3 what a coarse block costs](#3-what-a-coarse-block-costs-the-output-new) → [4 the address](#4-the-address-signal-background-excess) → [5 the working address](#5-the-working-address-from-the-first-layers-to-the-rest) → [6 adaptive depth](#6-adaptive-depth) → [7 distance](#7-the-distance-between-blocks) → [8 zones](#8-zones) → [9 the best allocation](#9-the-best-allocation-knapsack-and-lagrangian-new) → [10 the batch](#10-the-batch) → [11 open](#11-what-is-measured-and-what-is-open).

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

## 6. Adaptive depth

A fixed $n$ is fitted to the length of one network. A rule that decides by what it reads is one for any length.

**One target.** So that depths are compared by what they read, all of them predict one address: $T = \{b : \ell(b) \ge n_\text{max}\}$.

**Two criteria.**
- The oracle margin $\delta_n(q) = \cos(\hat a^{(n)}, \tilde a_T(q)) - \max_{q' \ne q} \cos(\hat a^{(n)}, \tilde a_T(q'))$ needs the real deep address, that is the full pass. It bounds what any rule can reach, and is not available at inference.
- Convergence is available at inference; the most naive one - the prediction stopped moving:
  $$\Delta_n(q) = 1 - \cos\big(\hat a^{(n)}(q), \hat a^{(n-1)}(q)\big).$$
  Its form on the zones: with $Z_n$ the top $k$ blocks of $\hat a^{(n)}$, $J_n = |Z_n \cap Z_{n-1}| / |Z_n \cup Z_{n-1}|$. The zones care whether the chosen blocks moved, not how the whole mask trembles.

Drift and silhouette are read on the excess (the prediction minus its mean over the questions): the mean address is shared and far louder than any question, and on raw predictions every step looks like $10^{-4}$ - the measure cannot tell settled from moving.

**The silhouette rule with patience and roll-back** (Volodya, 2026-09-19). The silhouette at depth $n$ is $Z_n$, the future zones. Reading goes layer by layer; once the silhouette holds for $p$ depths running, reading stops and rolls back to where that run began:

$$n^*(q) = \min\{\, n \ge n_\text{min} : J_{n+1}(q), \dots, J_{n+p}(q) \ge 1 - \varepsilon \,\}.$$

On an easy question the silhouette at layers 7 and 8 is the one at 6: $n^* = 6$, and everything after 6 goes under zones. On a hard one it changes at 7 and 8 and hardly at 9 and 10: $n^* = 8$. No layer number enters the rule, only how the silhouette holds. Its knobs are $k$ (the silhouette's size), $\varepsilon$ (the tolerance), $p$ (the patience).

**The cap is a share of the network** (Volodya, 2026-09-19). $n_\text{max}$ is set by a share $\rho_\text{max} \in (0, 1)$ of the number of layers, not by a layer number; reading, patience included, goes no deeper than $\lfloor \rho_\text{max} L \rfloor$ layers, and a question whose silhouette has not settled by then stops there:

$$n^*(q) + p \le \lfloor \rho_\text{max} L \rfloor.$$

*Proved by construction:* whatever the question, the zones act on at least $W(\lfloor \rho_\text{max} L \rfloor)$ of the weights - the budget holds, and the same share carries to a model of another length.

The default is $\rho_\text{max} = 0.3$: on E2B, 10 of 35 layers with the patience, so at $p = 2$ the deepest stop is layer 8 and the zones act on about 80% of the weights ($W(8) = 0.84$, $W(12) = 0.76$). The address needs those layers: on the target from layer 10 on, a fixed depth of 8 identifies 0.908 / 0.880 / 0.892 of TriviaQA / NQ / SQuAD against 0.845 / 0.770 / 0.866 at depth 5, the deepest stop a cap of 0.2 allows. Why not more: at a cap of 0.5, in one pass half the network is read at the base without zones for the whole answer, which may hurt the model, and in two passes every question pays an extra pass over half the network.

Every question decides its own depth. The cap does not follow a statistic of past questions: questions come at random and of different difficulty, and past ones say nothing of the next; such a cap would cut a hard question by the measure of the easy ones before it.

The price: $n^* + p$ layers read at the base, the zones from $n^*$ on. In one pass the $p$ layers after $n^*$ are read again with the zones - the price of a stop confirmed rather than guessed from one step.

**Why convergence says something about what is left.** Let $\mathcal{F}_n$ be everything computable from the first $n$ layers; $\mathcal{F}_1 \subset \mathcal{F}_2 \subset \dots$ is a filtration. The best prediction of the target from $\mathcal{F}_n$ in squared error is $M_n = \mathbb{E}[a_T \mid \mathcal{F}_n]$, an orthogonal projection.

*Proved.* (1) By the tower property $\mathbb{E}[M_{n+1} \mid \mathcal{F}_n] = M_n$: the predictions are a Doob martingale. (2) The steps $D_k = M_{k+1} - M_k$ are orthogonal: for $j < k$ the step $D_j$ is known at $k$, so $\mathbb{E}[D_j D_k] = \mathbb{E}[D_j\, \mathbb{E}[D_k \mid \mathcal{F}_k]] = 0$; the rest $R = a_T - M_L$ is orthogonal to every step. (3) $a_T - M_n = R + \sum_{k \ge n} D_k$, so by Pythagoras

$$\mathbb{E}\lVert a_T - M_n \rVert^2 = \mathbb{E}\lVert R \rVert^2 + \sum_{k \ge n} \mathbb{E}\lVert D_k \rVert^2.$$

The error at depth $n$ is the sum of the corrections still to come plus what the whole network does not know.

**What the identity does not give.** It speaks of the steps to come; $\Delta_n$ sees the step just made. The rule rests on a hypothesis: the steps shrink with depth on average. Further, ridge on a finite calibration is not the exact conditional mean and $P_n$ differs per depth, so the martingale is approximate; the identity is an average over questions; the cosine is a relative step, the identity is about the squared norm.

*Measured 2026-09-19*, the hybrid, the target from layer 10 on, $k = 147$ (1% of the blocks), $\varepsilon = 0.2$, $p = 2$: at the cap of 0.2 almost every question reaches the cap (205 of 206, 100 of 100, 257 of 306) and the rule falls back to the fixed depth 5 (identification 0.840 / 0.770 / 0.853 against 0.845 / 0.770 / 0.866); at 0.3 the stops spread over layers 6-8 and identification matches the fixed depth of the same mean cost or exceeds it by up to 1.5 points (0.898 / 0.850 / 0.863). 49 SQuAD questions settle at layers 1-3. A cap of 0.2 stops the reading before the layers the address needs.

## 7. The distance between blocks

A zone needs "near". The rules ([section 8](#8-zones)) look only at distances from a center, so the metric is a replaceable part: [metric.py](../src/foqlens/metric.py), the signal's path in [coupling.py](../src/foqlens/coupling.py). Which metric is best is a question of [zone-strategies.md](zone-strategies.md#along-what-zones-grow).

**M1, co-activation.** The profile of a block is its masks over $n$ calibration questions, standardized: $\tilde x_b$. The correlation $\rho_{ab} = \tilde x_a^\top \tilde x_b / n$ and the distance $d_1(a, b) = \sqrt{2(1 - \rho_{ab})}$.

*Proved:* $d_1(a, b) = \lVert \tilde x_a - \tilde x_b \rVert / \sqrt n$ (expand the square with $\lVert \tilde x \rVert^2 = n$) - a Euclidean distance, the triangle inequality holds; shifting or scaling a block's profile does not move it.

**The graph.** The surface of the zones is a graph of blocks. A union of $k$ nearest lists grows hubs: a block near the mean profile enters hundreds of lists. Mutual neighbours on the rescaled distance NICDM (Schnitzer et al. 2012), $d'(a, b) = d(a, b) / \sqrt{\mu_a \mu_b}$ with $\mu$ the mean distance to the $k$ nearest: a hub has a small $\mu$, and distances to it grow. Components are joined along the minimum spanning tree of the union graph on $d'$, what is left by Borůvka rounds. *Proved (an invariant):* the graph is connected. *Measured 2026-09-16* on E001's masks: degree at most 17, one component, the skew of k-occurrence 3.38 → 0.81.

**The distance along the graph** is the shortest path (the geodesic); on a connected graph with positive lengths it is a metric.

**M4, the query's medium.** An edge conducts by the query's activity at its ends, $\hat a_b = a_b / \bar a_b$ (the ratio to the background). An edge is two halves in series (Kirchhoff):

$$c_{ab} = \frac{2}{1/\hat a_a + 1/\hat a_b}, \qquad \text{length } \ell_{ab} = d_\text{base}(a, b) / c_{ab}.$$

*Proved:* $\min(\hat a_a, \hat a_b) \le c_{ab} < 2 \min(\hat a_a, \hat a_b)$ - a quiet end closes the edge, a loud one cannot open it more than twice; at $\hat a \equiv 1$ it is the shortest path of the base graph.

**M4b, the border by the jump** (Perona-Malik): $c_{ab} = \exp\!\big(-(\hat a_a - \hat a_b)^2 / K^2\big)$, $K = 1.4826 \cdot \text{MAD}$ of the jumps; 1 at equal activity, less at a jump.

**M3, the signal's path through the weights.** An edge from block $a$ of module $A$ to block $b$ of its reader $B$ conducts

$$\kappa(a \to b) = \lVert A[\rho_a, :] \rVert_F \cdot \lVert B[\sigma_b, C(\rho_a)] \rVert_F.$$

*Proved* (the Frobenius norm is submultiplicative): it bounds the transfer through $x \to A \to B$ at unit input; the graph is built from the weights once per model.

**The width of the network** $D$ is the largest distance found by sweeps; $f$ is a share of $D$.

## 8. Zones

The code: [graph_zones.py](../src/foqlens/graph_zones.py) and [zones.py](../src/foqlens/zones.py); rules 1-7 in [quantization-filter.md](quantization-filter.md#the-mechanism-in-formulas).

**Centers and hills.** The mask is smoothed by the mean over graph neighbours. A peak is a block no lower than any neighbour and above the quantile `PEAK_QUANTILE`. The hill of a peak is the connected region of the graph above half its height over the median; a weaker top inside a stronger hill is not a zone. The query decides how many zones there are.

**The own radius** $r_i$ is the smallest radius at which the blocks around the center hold the weight of the hill - the width at half height. *Proved (an invariant):* the blocks within $r_i$ hold at least the hill's weight, and no smaller radius does.

**Reach.**
- Rule 1, `EqualReach`: $R_i = f \cdot D$ for every zone; $f = 0$ is the center only, $f = 1$ the whole network.
- `ProportionalReach`: $R_i = f \cdot D \cdot r_i / \bar r$, $\bar r$ the mean own radius of the query's zones.
  *Proved:* the mean reach $\frac1n \sum_i R_i = f D$, so $f$ keeps its meaning as a share of the network; $R_i / R_j = r_i / r_j$ at any $f > 0$; $\partial R_i / \partial f \ge 0$; with one zone or equal $r_i$ it is rule 1.

**Lift and level.** The lift of a zone is $L_i(b) = \max\big(0, 1 - d(b, c_i) / (R_i\, s_\text{last})\big)$ (rule 4); combined by rule 5 into $L = \min(1, \sum_i L_i)$ or $\max_i L_i$; the ceiling $\kappa = \gamma + \lfloor g (m - \gamma) \rfloor$ (rule 2).
*Proved:* with the even profile and $L > 0$ the level is $\min\big(\kappa, \gamma + 1 + \lfloor L (\kappa - \gamma) \rfloor\big)$; the level does not fall as $f$ or $g$ grows; a kernel with $L > 0$ everywhere lifts the whole network by a rung, so the profile must reach 0 at a finite distance.

**The strength of a zone.** A ceiling of its own, $\kappa_i = \gamma + \lfloor g\, s_i (m - \gamma) \rfloor$, $s_i \in [0, 1]$ the strength of the heads pointing at its center, through the projection. Zones with different ceilings combine in rungs: $r(b) = \sum_i L_i(b)(\kappa_i - \gamma)$, level $= \min\big(\max_i \kappa_i, \gamma + 1 + \lfloor r \rfloor\big)$ at $r > 0$. *Proved:* with equal $\kappa_i$ this is rule 5's sum; with $s_i \equiv 1$ it is rule 2.

**Decay comes down to the front.** The lift could fall off by decay, $L_i(b) = K(d/R)$ with $K(u) = \frac{e^{-u} - e^{-s_\text{last}}}{1 - e^{-s_\text{last}}}$ for $u \le s_\text{last}$ and 0 beyond. *Proved:* the center is lifted to 1, and the level depends on $L$ only through a threshold on $u = d/R$; both $1 - u/s_\text{last}$ and $K(u)$ strictly decrease, so any split into rings that decay gives, the linear front gives too, with stops $s'_j = s_\text{last}(1 - K(s_j))$. Front or decay is a choice of the profile (rule 3), not a knob of its own.

**What the rules guarantee.** Each property is proved by substitution into rules 1-6.
- The reach does not saturate: $R = f D$ grows over the whole range $f \in [0, 1]$. A stretch $R \propto f/(1-f)$ would saturate the threshold from $f \ge \tfrac12$ on and send the radius to infinity.
- Inside a zone a level is never below the base: rule 6 gives $\gamma + 1 + \lfloor L(\kappa - \gamma) \rfloor \ge \gamma$. A level of the form $\lfloor \kappa L \rfloor$ would drop blocks with $L < \gamma/\kappa$ below the base.
- At $g = 0$ a zone is the base: $\kappa = \gamma$, and no block rises or falls.
- A level never leaves the ladder: $L \le 1$ after combining, so the level is at most $\kappa \le m$.
- A negative signal gives no negative lift: the excess is clipped at zero before the projection.

## 9. The best allocation: knapsack and Lagrangian (new)

**The problem.** Choose every block's level:

$$\min \sum_b \tfrac12\, s_b\, \sigma^2_{\ell(b)}(b) \quad \text{subject to} \quad \sum_b c_b(\ell(b)) \le M.$$

It is a multiple-choice knapsack: every block takes exactly one level.

**The Lagrangian.** A price of memory $\lambda \ge 0$ moves the constraint into the objective:

$$\Lambda(\lambda) = \sum_b \min_\ell \big[\tfrac12\, s_b\, \sigma^2_\ell(b) + \lambda\, c_b(\ell)\big] - \lambda M.$$

The problem splits into independent decisions per block: a block goes up a rung when the gain pays for it,

$$u_b(\ell \to \ell^+) \ge \lambda \big(c_b(\ell^+) - c_b(\ell)\big).$$

*Proved (standard duality):* the solution at every $\lambda$ is optimal for the memory $M(\lambda)$ it spends; $M(\lambda)$ does not increase with $\lambda$. The Lagrangian bound equals the optimum of the linear relaxation; in a multiple-choice knapsack at most one class is fractional at that optimum (Sinha & Zoltners 1979), so the gap to the integer optimum is at most the gain of one block.

**Thresholds per rung.** By [section 3](#3-what-a-coarse-block-costs-the-output-new) the gain of rung $k$ above the base is $\tfrac12 s_b \Delta_b^2 \cdot 15 / (12 \cdot 16^k)$ and its cost $w_b \cdot 2/8$ bytes. With the reduced sensitivity $\hat s_b = s_b \Delta_b^2 / w_b$ the condition becomes $\hat s_b \ge \lambda' \cdot 16^k$, $\lambda' = \frac{24}{15} \cdot \frac28 \cdot \lambda = \frac{2\lambda}{5}$, one constant for every block. The levels are superlevel sets of the reduced sensitivity:

$$\text{level}(b) = \gamma + \#\{\, k \ge 1 : \hat s_b \ge \lambda'\, 16^k \,\}.$$

**The link to zones.** If $\hat s_b$ is smooth on the block graph - neighbours alike in sensitivity - the sets $\{b : \hat s_b \ge t\}$ are hills around peaks, and the nested thresholds $t, 16t, 256t$ are nested rings. The zones of [section 8](#8-zones) are a parametric approximation of this optimum: peaks are the centers of the superlevel sets, the radius their size, the profile their nesting. $f$ plays the price of memory: a larger $f$ is a smaller $\lambda'$ and wider sets.

Consequences to check:
1. The rings should lie 16 times apart in reduced sensitivity per rung; the even profile of rule 3 does not know this. A profile from the measured $\hat s_b$ (or its proxy) is a direct candidate.
2. Wide low zones against narrow high ones - [section 3](#3-what-a-coarse-block-costs-the-output-new).
3. The per-block threshold (the per-block control) is the Lagrangian solution without smoothness; zones beat it only if the address is noisy per block and smoothness on the graph averages the noise out.

**The per-block quantile regulator is a preset budget.** The per-block control lifts the blocks above a quantile: $P_f = \operatorname{Quantile}(\sigma, 1 - f)$, $L_b = \max(0, \sigma_b - P_f) / \max_b(\sigma_b - P_f)$. *Proved:* the number of blocks above the base, $\#\{b : \sigma_b > P_f\} = f |B|$ (up to ties), whatever $\sigma$ is, so its memory is set in advance, while the zones' memory follows from the query ([rule 7](quantization-filter.md#the-mechanism-in-formulas)). The share counts blocks, not weights, and blocks differ in their number of inputs, so its $f$ is not even a share of memory. Normalizing by the maximum lifts the strongest block of any query to the ceiling, so it has no strength of its own per zone (H4). Its role on the bench is the control: what the connectedness of zones adds over a per-block score.

## 10. The batch

At a decoding step a batch reads every block at the highest level among its questions:

$$C(S) = \sum_b w_b \cdot \text{bits}\big(\max_{q \in S} \ell_q(b)\big) / 8.$$

*Proved:* $C(S)$ is monotone in $S$ and submodular - a question adds no more to a larger batch than to a smaller one (a maximum over a set is a coverage function per block, and a sum of submodular functions is submodular). Consequence: questions with close zones are cheaper in one batch; choosing a batch is a covering problem. *Measured, smoke of 2026-09-19* at $f = 0.2$, bytes per question against per step of the batch: per block 760 / 937 MB, static zones 1126 / 1590, the signal's path 946 / 1508; uniform D2 677, D8 1978.

Sharing a batch's cost between its questions (the Shapley value) and a budget between the requests of a pool are game theory, for the stage of serving many requests.

## 11. What is measured and what is open

Measured (2026-09-19): identification of the address and of paraphrases by source; the depth of the working address and windows; the hybrid is the best address; the silhouette rule on three corpora.

Open:
- the gain of zones on answers at the same memory against the ladder of [E003](../experiments/E003-calibrated-base/results.md) - the main measurement, what everything else is for;
- the noise model of [sections 2-3](#2-the-error-of-a-level) on real weights: does the share of the error fall 16 times per rung;
- wide low zones against narrow high ones;
- adaptive depth: do the steps shrink, and which tolerance and silhouette size let the rule work within a 0.3 cap;
- smoothness of the sensitivity on the block graph - the condition for zones to be near the optimum;
- heads: leave out the sink heads, the share of attention on the question's tokens;
- the strength of a zone and how well the model knows the question ([H4](hypotheses.md)).

**What a zone must give over a per-block score.** If none of the four holds, the honest conclusion is that a zone is a picture and the mechanism a per-block allocation.
1. The cost of the description: a zone is a center and a radius, a shape is a dozen numbers, not a mask over 14 708 blocks.
2. Stability across queries of one topic: zones repeat where the raw score does not. The measure, with a bound: for every zone $d(c_i, c'_i)/r_i$ between two texts (zones matched by overlap) and the share of the weight both layouts cover.
3. Smoothness: neighbouring blocks at very different precision may hurt more than the mean bits show.
4. Carrying into the kernel: connected groups of blocks are cheaper to read than a scattered mask.

**The null check.** A metric in which the zones cannot be told from what the same procedure builds on data with the topic labels shuffled has found no zones - whatever quality they hold.
