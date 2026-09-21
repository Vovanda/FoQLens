---
title: Reading notes
---

# Reading notes

What was taken from the literature, read when the stage that needed it came. Quotes are verbatim, with the section, paragraph or table they come from, so that they can be cited as they are; what FoQLens takes from each work follows the quotes. The list of related work is in [prior-art.md](prior-art.md).

## MoBiQuant

Wang, Kim, Han, Gudovskiy, Nakata, Okuno, Peong, Jeon, Ko, Chen, Yang. *MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM.* arXiv 2602.20191v2, 25 May 2026. Read in full 2026-09-12 from the arXiv HTML version, for the speed of reading residual slices and for the architecture of the bench. Formulas and some numbers were lost in the HTML-to-text conversion; where a quote contains a dropped symbol, it is marked [...].

### What it does

- Abstract: "we propose a many-in-one recursive residual quantization that can iteratively reconstruct higher-precision weights at runtime and mitigates outlier migration with a token-aware router to dynamically select the optimal inference precision of each token."
- §1, par. 3, the phenomenon behind it: "the specific subset of tokens that are responsible for high quantization errors are not static for each precision."
- §3, last par.: "Counterintuitively, it shows that inferring tokens at a lower precision can yield higher overall performance (pink bar)."

### Slices (§4.1, Appendix B)

- §4.1, par. 2: "Our MoBiSlice decomposes the weight matrix [...] of each linear layer in the LLM into [...] slices [...], each containing a slice of [...] bits from the quantized weight. This decomposition is implemented by recursively quantizing the residuals".
- §4.1, par. 2: "Moreover, dequantization can be performed via efficient shift-and-add operations (Sec. 4.3)."
- §4.1, par. 3: "This design enables any-precision inference using a fixed 2-bit kernel, thereby avoiding kernel relaunch overhead."
- Appendix B, "Quantizer design": "we adopt a floor-aligned mapping following the truncation-ready quantization principle [14], where a lower precision code is obtained by simply dropping least significant bits (LSB) rather than re-rounding."
- Appendix B, same paragraph block: the residual slices fix their zero point, "placing the midpoint code at the center of the integer range so that positive and negative residual corrections are represented symmetrically, which avoids systematic drift during slice accumulation."
- Appendix B, "Bias and Error Bounds", last par.: "activating residual slices performs a true residual refinement: it only fills in finer bit slices without altering the coarser representation".
- Appendix C.1: "We use weight-only quantization in all reported runs in the main paper, with abits=16 and group_size=128, while the base bit slice uses wbits=2." and "our default configuration uses four bit slices with slice_bits_list = 2 2 2 2."

### Routing and the budget (§4.2)

- §4.2, "Challenge 1", last par. (Eq. 6): "Using the learned mask, the input token [...] is routed through the selected slices, and the corresponding outputs are aggregated to produce the output token".
- §4.2, "Joint optimization": "We also treat [the first slice] as a shared-expert slice such that tokens always pass through for stable training."
- §4.2, "Efficient runtime precision switching": "Increasing [the threshold] reduces the number of activated slices per token, thereby lowering the effective precision, and vice versa."

### Kernel (§4.3)

- Par. 1: "Conventional static low-bit kernels typically load all slices regardless of the runtime precision, leading to unnecessary memory bandwidth and limiting the inference speedups." and "only the required slices are fetched that enables on-demand memory access with proportional speedups."
- Par. 2: "our kernel performs Binary Matrix Multiplication (BMMA) directly on packed bit-planes." and "the lower-bit slice is first shifted and added to the higher-bit slice at the bit-plane level, then multiplied by the shared scaling factor as shown in Fig. 3."
- Par. 4 ("Non-contiguous Memory Access and Load Imbalance"): "we apply token permutation after routing. Then, the tokens that are assigned to the same bit slice are stored contiguously, thereby improving memory bandwidth utilization."
- Par. 5: "we employ a parallel execution strategy using independent CUDA streams to overlap the computation of the first slice with subsequent ones."

### Results

- Table 1, LLaMA2-7B, WikiText2 perplexity at 2 bits: AnyPrecisionLLM 2e3, AnyBCQ 15.38, MoBiQuant 10.91; decoding throughput at 2 / 3 / 4 bits: 404 / 311 / 256 tokens per second (AnyBCQ 312 / 268 / 225).
- §5.1: "All kernel results are measured on NVIDIA A100 GPUs with CUDA 12.9."
- §5.3, "Token- and block-wise assignments": "We also observe that the average precision varies across different linear blocks, even though each layer is trained for the same target bit-width." and "the first layers tend to have deviations in bit-width assignments."
- §5.4: MoBiQuant "achieves [...] end-to-end latency reduction for different decoding lengths when compared to FP16" - the factor is in Fig. 7 (left), dropped from the text.
- Table 9, GSM8K, LLaMA3.2-1B at 4 bits: FP16 46.47%, OmniQuant 30.86%, MoBiQuant (elastic) 32.83% (flexible extract).
- Appendix F: "our evaluation mainly focuses on single-request or small-batch decoding scenarios [...] The behavior of token-wise adaptive precision under large-scale cloud serving workloads with heterogeneous batching strategies remains an important direction for future investigation." No code release is mentioned.

### Where FoQLens goes further

- **The unit of precision.** MoBiQuant decides how many slices a *token* reads, the same for every weight of a layer ("dynamically activates the optimal number of MoBiSlice residual components for each token", §1). FoQLens decides the depth of every *block of weights* for a query - an address in the weights, not only a number of bits.
- **No trained router.** MoBiRoute is "a learnable 2-layer MLP" per layer, trained jointly with the quantizer (§4.2, Eq. 4, Eq. 9). FoQLens takes the address from the model's own activations and gradients.
- **A structure, not a local switch.** A MoBiQuant decision is local to a token and a layer. FoQLens lays precision out as expert zones over the whole network, with the zones, their profile and the isthmus where two zones meet ([precision-regulator.md](precision-regulator.md)).
- **What it is for.** MoBiQuant fixes the generalization of post-training quantization across precisions ("outlier migration", §3) under a budget. FoQLens aims at a regulator that follows the task, the machine and the value of the query, with MoE as its special case ([goals.md](goals.md)).
- **Mixed layouts in one batch.** MoBiQuant leaves "heterogeneous batching strategies" open (Appendix F); the FoQLens bench already evaluates a batch in which every question has its own layout.
- **Method.** MoBiQuant reports perplexity, zero-shot accuracy and throughput; FoQLens compares against uniform quantization at the same memory, and against the other topic's zones for the address, with every prediction preregistered.

### Does the FoQLens idea fit their scheme

Yes - they are two axes of the same storage.

- **Same slices.** The FoQLens `RefinedWeight` is already a MoBiSlice: four 2-bit slices, each the residual of the previous one, the scale refined by 4 per slice, symmetric residual codes (compare §4.1 and Appendix B with `foqlens/quant.py`). A lens layout reads the same planes their kernel fetches.
- **Two axes compose.** Their router picks a depth per token; a zone picks a depth per block of output rows. Together the depth of a weight read for a token is a function of both - for example, the zone sets the ceiling of a block (focus_strength) and a token router chooses within it.
- **In a kernel.** A tiled GEMM loops over the output rows in tiles; a depth per tile - how many planes the tile fetches - is the zone layout itself, and the FoQLens block of 64 rows fits a tile. Their token permutation (§4.3) groups the other axis. What is new for their kernel is a different tile depth per question when a batch mixes queries.
- **What FoQLens would take as is.** Their calibration of the base slice (Appendix B, C.1) for a base at 2 bits, and their shared always-on first slice (§4.2) as the base under the zones.

### Calibrating the base slice (Appendix B, C.1, Algorithm 1) - read 2026-09-12 for the D2 floor

- Appendix B, "Quantizer design": "we adopt a floor-aligned mapping following the truncation-ready quantization principle [14], where a lower precision code is obtained by simply dropping least significant bits (LSB) rather than re-rounding." and "The floor operator enforces hierarchical nesting of integer codes, so switching bit width corresponds to adding or dropping bit slices without changing previously formed higher order bits."
- Appendix B, same section: "Let [the calibrated parameters] be the calibrated parameters of the first slice. After assigning [b] bits to slice [e], the next slice refines the resolution as [s / 2^b]. Finally, while the first slice uses the calibrated zero point [z], slices fix [the zero point] for [e > 1], placing the midpoint code at the center of the integer range".
- Algorithm 1, Stage 1: "First slice (FS) stabilization ... First slice-only forward pass ... match FP reference output ... Update [the quantization parameters]". Stage 2 then trains slices and router jointly.
- Appendix C.1: "the base bit slice uses wbits=2 ... our default configuration uses four bit slices with slice_bits_list = 2 2 2 2. Training proceeds for epochs=20 and nsamples=128, with batch_size=1 for all models."
- Appendix C.1: "For each layer, we optimize three parameter groups with AdamW: learnable weight clipping parameters (LWC), learnable equivalent transformation parameters (LET), and MoBiQuant parameters ... We typically use lwc_lr in the range 1e-3 to 1e-2".
- §4.2, "Joint optimization": "We freeze all weights from the pretrained LLM and calibrate only [the quantizer and router parameters]." and §4.3 of the main text: "we adopt a layer-wise calibration strategy from [25]" - OmniQuant.
- Appendix B, "Bias and Error Bounds": truncation noise is zero-mean and "strictly smaller than one half step of the coarser quantizer ... cannot flip any bit of the coarse code".

**What this means for the FoQLens floor at D2.** The part FoQLens needs is Stage 1 alone: no router, no token routing - a layer-wise fit of the first slice's clipping and zero point against the layer's full-precision output, with the pretrained weights frozen. The rest of the ladder follows: slices 2 to 4 are the residuals of the calibrated first slice, so D4, D6 and D8 are rebuilt too and will not be bit for bit with the current ones. The test after calibration is therefore "D2 works and the deeper reads are no worse", not "the deeper reads are unchanged".

## OmniQuant

Shao, Chen, Zhang, Xu, Zhao, Li, Zhang, Gao, Qiao, Luo. *OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models.* arXiv 2308.13137, ICLR 2024. Read 2026-09-12 from the arXiv HTML for the calibration MoBiQuant builds on - it is the source of the learnable weight clipping the D2 floor needs.

- **Learnable weight clipping (LWC).** The quantizer is the usual affine one, `Wq = clamp(round(W / h) + z, 0, 2^N - 1)`, but its range is learned through two factors in [0, 1]: `h = (gamma max(W) - beta min(W)) / (2^N - 1)` and `z = -round(beta min(W) / h)`. Clipping the range is what a 2-bit weight needs: with the full range each step is so coarse that the bulk of the distribution collapses into one or two codes.
- **Block-wise objective.** `arg min ||F(W, X) - F(Qw(W; T1, T2), Qa(X, T2))||` over one transformer block at a time, sequentially - the same layer-wise strategy MoBiQuant cites as "[25]".
- **Calibration setup.** 128 segments of 2048 tokens from WikiText2; the full-precision weights stay frozen and only the clipping strengths and the transformation factors are trained; AdamW with no weight decay, learning rate 5e-3 for LWC; 20 epochs, 40 for 2-bit; one A100-40G.
- **What it buys at 2 bits.** LLaMA-2-7B at W2A16 with groups of 128: perplexity 11.06, against 36.77 for GPTQ - the two-bit weight becomes usable, which is exactly what the FoQLens floor at D2 is missing (ours diverges at 9.4e6).

**What FoQLens takes.** LWC alone, on the first slice: learn `gamma` and `beta` per group of the base slice against the block's full-precision output, weights frozen. LET is for activation quantization and is not needed - the bench quantizes weights only. The scale refinement of the deeper slices follows from the calibrated first one, so the whole ladder is rebuilt with it.

### What FoQLens takes

1. **The bench now.** Slices never change after quantization, so the weight of every read depth can be built once and kept (a bench mode behind a flag, bit for bit with the current reads), instead of unpacking four slices of 275 modules on every batch. See issue "Bench: build the depth weights once".
2. **The kernel later.** The output is linear in the slices (Eq. 3 and 6): the output of each slice can be computed once and summed with a mask per output row - in FoQLens the mask is the zone layout, per question and per block of rows. With a kernel that fetches only the required planes (§4.3, par. 1-2), memory, speed and energy follow the zones.
3. **A base at 2 bits - tried and dropped for this storage.** A 2-bit base works when it is calibrated (Table 1, Appendix B, C.1); ours diverges uncalibrated (perplexity 9.4e6). Measured 2026-09-12 on E2B, it cannot be fixed by calibrating the first slice while the ladder stays: the residual slices reach only half a step of the first one, so anything that sharpens D2 leaves a remainder they cannot cover.
   - clipping the range to 0.8 of the group's absmax: the D2 error falls from 0.396 to 0.355 of the weights' rms, and the D8 error rises from 0.006 to 0.081 - thirteen times worse;
   - levels fitted by Lloyd-Max to the model's own weights: D2 0.350, D8 0.059;
   - a group of 8 weights instead of 64: D2 0.302, but the scales then cost 4 bits per weight, more than the slices;
   - overlapping slices (each refining by 2 instead of 4): D8 recovers to 0.040 at the price of three bits of depth.
   It is not a few bad layers either: the whole net at D2 reads perplexity 2.5e6, and holding the last four layers at D4 only brings it to 3.8e4, against 11.1 for D4 everywhere. Two bits over four even levels are simply too coarse for Gemma 4 E2B; the floor of a layout stays D4. What is left untried is fitting the clipping against a block's output rather than its weights - the gain on the weights is 10-12%, and a working D2 needs multiples of that.
4. **The shared first slice is the base.** Their always-on first slice (§4.2) is our base precision: the minimum every block is read at, with the zones adding slices on top.
5. **Precision varies by block** (§5.3) even under a per-token router - support for addressing precision by block, as FoQLens does.
6. **Mixed depths in a big batch are open** (Appendix F) - exactly the case of the FoQLens bench, where every question in a batch has its own layout.
7. **Lower precision can score higher** (§3) - a small observation in their setting, in the direction of H4.

## Dynamic precision by the input: DQT, DynaQuant, InfoQ

Three papers read in full on 2026-09-21 from their arXiv HTML versions, with the code where there is any, to answer
one question: does published work already allocate precision by the input the way FoQLens does. Formulas are quoted
from the LaTeX of the HTML; a dropped symbol is marked [...].

In short: all three pick a bit width per **layer** of a convolutional network. The dynamic two do it with a small
network trained jointly with the model; InfoQ does it once per network, for every input alike. None works on a
language model, none looks inside a layer, none reads the meaning of a query.

### DQT

Shalby, Pittorino, Palermo, Trojaniello, Roveri. *DQT: Dynamic Quantization Training via Dequantization-Free Nested
Integer Arithmetic.* arXiv 2508.09176v1, 7 Aug 2025. The abstract page names no venue. No code is released.

**What it does.**

- Abstract: "Dynamic, instance-based mixed-precision quantization promises a superior accuracy-efficiency trade-off by
  allocating higher precision only when needed. However, a critical bottleneck remains: existing methods require a
  costly dequantize-to-float and requantize-to-integer cycle to change precision".
- §1: "In the DQT architecture, a lightweight controller adaptively selects per-layer bit-widths for each input. At
  inference, the model loads high-precision weights once and uses bit-shifting to instantiate the desired precision
  on-the-fly."
- §4, the nesting: "The quantization scale for any other bit-width, $b<n$, is defined by a power-of-two relationship:
  $\Delta_{b}=\Delta_{n}\cdot 2^{n-b}$" (Eq. 3), so that
  $Q^{b}_{\text{DQT}}(x)=\text{clip}\left(\left\lfloor\frac{Q^{n}(x)}{2^{n-b}}\right\rceil,0,2^{b}-1\right)=Q^{n}(x)\gg(n-b)+\epsilon$ (Eq. 4).
- §5, the controller: "It outputs logits for each of these layers, producing a probability distribution
  $\mathbf{p}_{i}=\text{softmax}(\mathcal{C}(\mathbf{x})_{i})$ over $K$ candidate bit-widths" and "At inference,
  bit-widths $b_{i}$ are chosen via a non-differentiable argmax". Appendix D: "The controller architecture is a
  lightweight two-layer MLP. The first layer has a hidden dimension of 64."
- §5, the loss: $J=J_{\mathrm{task}}+\alpha J_{\mathrm{consistency}}+\beta J_{\mathrm{cost}}$ (Eq. 11), where
  $J_{\mathrm{consistency}}=\sum_{b\in K^{\prime}}J^{(b)}_{\mathrm{task}}$ "is the task loss when the entire network is
  statically set to bit-width $b$" and $J_{\mathrm{cost}}=\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}p_{i,k}\cdot d_{k}$.
- Appendix F: with $\alpha=0$ "training is unstable for the 2-bit configuration ... as the controller favors
  higher-precision bit-widths during training."

**Results.**

- §6: "On ImageNet with ResNet-50, our 4-bit dynamic DQT model achieves 77.00% top-1 accuracy, outperforming the SotA
  static method LSQ (76.70%) and the leading dynamic method DQNet (76.94%) at a comparable BitOPs budget."
- Appendix D: "The cost regularization $\beta$ is set to 0.0 for the main results to prioritize accuracy." The
  controller of the headline numbers is not pushed to save anything.
- §6, memory: "The model memory footprint is determined by the master bit-width $n$, as the full $n$-bit weights must
  be stored. We use $n=8$". A lower bit width saves operations, not memory.
- §7: "Extending this framework to other domains, including large language models (LLMs) and object detection, is
  another important direction."

### DynaQuant

Bao et al. (Shenzhen University, Harbin Institute of Technology). *DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image
Compression.* arXiv 2511.07903, AAAI 2026. Code: [baoyu2020/DynaQuant](https://github.com/baoyu2020/DynaQuant), read
2026-09-21.

**What it does.**

- Abstract: "we introduce a data-driven, dynamic bit-width selector that learns to assign an optimal bit precision to
  each layer, dynamically reconfiguring the network's precision profile based on the input data."
- "Content-Aware Bit-Width Selector": "The selector takes the input activation tensor
  $A\in\mathbb{R}^{C\times H\times W}$" - adaptive pooling, a two-layer MLP,
  $\bm{p}=\text{Softmax}(\text{MLP}(\bm{h}_{\text{pool}}))$ (Eq. 7), Gumbel-Softmax in training and "we
  deterministically select the bit-width with the highest probability" at inference.
- "Joint Optimization Framework": $\mathcal{L}=R+\lambda D+\gamma\mathcal{L}_{\text{bits}}$ (Eq. 8) with
  $\mathcal{L}_{\text{bits}}=\frac{1}{L}\sum_{l=1}^{L}\sum_{k=1}^{M}(\bm{p}_{l})_{k}\cdot b_{k}$ (Eq. 9) - the same cost
  term as DQT's.
- The rest of the paper is a gradient for rounding, $g(x)=\frac{1}{2}\cdot\frac{\tanh(\beta(x - \lfloor x \rfloor) - 0.5)}{\tanh(0.5)}+0.5$
  (Eq. 6), for learnable scales and zero points - training, not selection.

**Results.**

- Table 1, Cheng2020, the average over three datasets: fixed 8 bits, BD-Rate loss 1.60% at 4.00x; dynamic at 6.19
  bits on average, 12.18% at 5.17x. The dynamic model is compared there with a fixed one at 8 bits, not at its own
  bits.
- Table 2, the comparison at equal bits: fixed "+DPA[INT6]" at 6 bits, PSNR 35.664, R-D loss 1.74; dynamic
  "+DPA-DQ" at 6.02 bits, PSNR 36.231, R-D loss 1.68. At the same bits the dynamic selector gains 0.57 dB.
- "Bit-width Selection Results": "Edge layers (e.g., ga-0, ga-6, gs-1) exhibit higher precision than intermediate
  layers" and an image with more texture "assigns a 10-bit width to its gs-1 layer, exceeding the 8-bit allocation in
  other images." Most of the allocation follows the layer; the input moves it in places.

**The code.** `DynamicQConv.forward` (`src/dynaquant/quantization/dynamic.py`) quantizes the weight three times, to
`nbits - d_bit`, `nbits` and `nbits + d_bit`, each with its own rounding from the full-precision weight, runs all three
convolutions and sums them under the one-hot mask. The three levels are not nested, and a lower level saves nothing in
this code. The repository's `docs/IMPLEMENTATION_STATUS.md`: "Paper RD reproduction: pending" for every model, and "The
speedups reported by the paper cannot be inferred from PyTorch simulated quantization alone".

### InfoQ

Akbulut, Shalby, Pittorino, Roveri. *InfoQ: Mixed-Precision Quantization via Global Information Flow.* arXiv
2508.04753, AAAI 2026 (40(24), 19598-19606). The same group as DQT. Static: one allocation per network.

**What it does.**

- Abstract: "InfoQ assesses layer importance by performing a single forward pass to measure the change in mutual
  information in the remaining part of the network, thus creating a global sensitivity score."
- §4: "Our central hypothesis is that the impact of quantizing a layer is not a local phenomenon but is best measured
  by its effect on the information propagated through subsequent layers."
- §4, the measure against a model at 8 bits everywhere:
  $\Delta\text{SMI}_{L,Y}^{(i,b,j)}=|\text{SMI}(L_{j,\text{8bit}};Y)-\text{SMI}(L_{j,b};Y)|$ (Eq. 3), and the same for
  $\text{SMI}(X_{\mathcal{E}};L_{j})$, where the input is first embedded by DINOv2. The score (Eq. 4):
  $S=\frac{1}{b}\frac{\langle\Delta\text{SMI}\rangle_{\mathcal{O}^{j>i}}}{\langle\text{SMI}\rangle^{\text{8bit}}_{\mathcal{O}^{j>i}}}$.
- Algorithm 1: "Perturb model by quantizing only layer $\ell$ to bit $b$" with all others at 8 bits; §5: "the process
  requires $L\times|\boldsymbol{B}|$ forward passes over a small, labeled calibration dataset".
- §4, the allocation: $\boldsymbol{s}^{*}=\arg\min\sum_{\ell}\left(S_{w}(\ell,b_{w}^{(\ell)})+\alpha S_{a}(\ell,b_{a}^{(\ell)})\right)$
  subject to $\text{Cost}(\boldsymbol{s})\leq C$ (Eq. 5), "solved efficiently using an off-the-shelf solver".
- §4, where to measure: "the change in task-relevant information, $\Delta\text{SMI}_{L,Y}$, measured at layers toward
  the end of the network ... exhibits the strongest correlation with accuracy degradation. Conversely, the change in
  input-relevant information, $\Delta\text{SMI}_{X,L}$, is most predictive when measured at intermediate layers."

**Results.** Table 3, ResNet50 at a 12.2x weight compression after QAT: 77.03% top-1, a drop of 0.34 from full
precision, against 0.60 for LIMPQ and 1.50 for HAWQv2. §5: "on ResNet18 at a 4.5MB constraint, InfoQ yields a
configuration that is 25% more accurate than its closest competitor" before any retraining.

### Where FoQLens differs

| | DQT | DynaQuant | InfoQ | FoQLens |
| --- | --- | --- | --- | --- |
| Networks | ResNet, MobileNetV2 | image-compression autoencoders | ResNet, MobileNetV2 | Gemma 4 E2B-it, generating answers |
| Changes with the input | yes | yes | no | yes |
| Unit of the decision | layer | layer | layer | block of 64 rows inside a module; the oracles' maps today by group - a layer's attention or its MLP |
| Who decides | a trained MLP | a trained MLP | a one-time sweep over a calibration set | a regulator without training, from what the model itself computes ([precision-regulator.md](precision-regulator.md)) |
| What it decides by | the input as a whole | pooled activations of the first module | nothing per input | the address of the query from the first layers ([E004](../experiments/E004-question-address/results.md)) and the state entering each layer |
| Is the model retrained | yes, with the controller | yes | yes, after the search | no |
| Storage | one int8 master, lower widths by a shift | in the code, three separate roundings | one width per layer | one stack: a base and refinements, read to a depth ([refocustensors.md](refocustensors.md)) |
| Memory at a lower width | the same int8 | the same | smaller | the aim: what the layout reads; today the copy lies in memory whole |

- **The unit.** All three decide per layer. FoQLens reads every block of 64 output rows at its own depth, and the
  kernel multiplies a mixed layout in one launch ([kernels.md](kernels.md)). The maps measured so far are by group -
  70 on E2B - which is close to per layer. Inside a layer the bench has the means and no measured gain yet.
- **The selector.** DQT and DynaQuant train a controller together with the network, with a cost term on the expected
  bits. FoQLens trains nothing: the cheap regulator is the open question, and the address the regulator would read is
  measured on its own - a paraphrase is recognized in 0.917 of the cases against 0.717 for the words alone (E004).
- **Knowing the input against a common allocation.** InfoQ is the common map of FoQLens computed well: one layout for
  every input. On E005 the common map, averaged over the oracles' fields, holds 0.544 of the answers at 0.513 of the
  memory, and a query's own map holds 0.816 at 0.523 - on 103 questions only the top rung answers, with maps built
  knowing the answer. The gap is what knowing the query is worth, and a static allocation cannot take it by
  construction. Neither dynamic paper measures this gap: DynaQuant compares its selector with uniform bits, DQT with
  static uniform and static mixed methods at a comparable BitOPs budget, and neither with its own best static layout
  at the same bits.
- **The measure.** InfoQ's sensitivity - one layer lowered while the rest stay high, the change read downstream - is
  the design of the FoQLens drop oracle and of the reference ([bench-math.md](bench-math.md#8-the-oracles-an-oracles-field-and-how-it-is-read-into-a-map)),
  with mutual information in place of the likelihood of the model's own answer, and one calibration set in place of
  one question. The allocation under a budget is the same problem as bench-math section 10: a multiple-choice
  knapsack; InfoQ solves it with an ILP solver, FoQLens by a price of memory.
- **Nesting.** DQT's power-of-two relation between the steps of the widths is the relation between FoQLens rungs:
  refinement $k$ has the step $\Delta_b / 4^k$. DQT gets the lower widths from the top one by a shift and stores the top
  one; FoQLens stores the base and refines upwards, so a coarse read needs no deeper plane. This belongs in the
  comparison of the format with its analogues, beside Matryoshka Quantization, Any-Precision LLM and MoBiQuant.

### What FoQLens takes

1. **The consistency term** of DQT, $\sum_{b\in K^{\prime}}J^{(b)}_{\mathrm{task}}$ over uniform configurations: if
   H5 trains a network for zones, every rung has to stay usable, and without the term their 2-bit path did not train
   (Appendix F).
2. **The expected-bits cost** $\sum_k p_{i,k} d_k$ with Gumbel-Softmax, as the form of a learned regulator, should the
   untrained ones give an effect (plan, step 4).
3. **Observers inside the network.** InfoQ finds that the change of task information is read best at the end and the
   change of input information in the middle. The FoQLens oracles read only the likelihood of the answer at the
   output; a change read at inner layers is a candidate source for a regulator that decides as the pass runs.
4. **An exact solver for the allocation.** An ILP over groups - 70 of them, four rungs - is solved in milliseconds,
   and gives the exact optimum where the price of memory leaves a gap of at most one block
   (bench-math, section 10).
5. **A control each experiment keeps.** A dynamic allocation is judged against the best static one at the same
   memory; E005 does this with the common map, and the two dynamic papers do not.
6. **Nothing for the kernel.** DQT's gain is avoiding a floating-point tensor between two widths. The FoQLens kernel
   builds each weight in registers from the copy's bytes and never writes an unpacked weight
   ([kernels.md](kernels.md)).
