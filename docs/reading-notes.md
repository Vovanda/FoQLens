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
- **A structure, not a local switch.** A MoBiQuant decision is local to a token and a layer. FoQLens lays precision out as expert zones on a weight map of the whole network, with lenses, their profile and the isthmus where two lenses meet ([lens.md](lens.md)).
- **What it is for.** MoBiQuant fixes the generalization of post-training quantization across precisions ("outlier migration", §3) under a budget. FoQLens aims at a regulator that follows the task, the machine and the value of the query, with MoE as its special case ([goals.md](goals.md)).
- **Mixed layouts in one batch.** MoBiQuant leaves "heterogeneous batching strategies" open (Appendix F); the FoQLens bench already evaluates a batch in which every question has its own layout.
- **Method.** MoBiQuant reports perplexity, zero-shot accuracy and throughput; FoQLens compares against random lenses of the same shape and the other topic at the same memory, with every prediction preregistered.

### Does the FoQLens idea fit their scheme

Yes - they are two axes of the same storage.

- **Same slices.** The FoQLens `SlicedWeight` is already a MoBiSlice: four 2-bit slices, each the residual of the previous one, the scale refined by 4 per slice, symmetric residual codes (compare §4.1 and Appendix B with `foqlens/quant.py`). A lens layout reads the same planes their kernel fetches.
- **Two axes compose.** Their router picks a depth per token; a lens picks a depth per block of output rows. Together the depth of a weight read for a token is a function of both - for example, the lens sets the ceiling of a block (focus_strength) and a token router chooses within it.
- **In a kernel.** A tiled GEMM loops over the output rows in tiles; a depth per tile - how many planes the tile fetches - is the lens layout itself, and the FoQLens block of 64 rows fits a tile. Their token permutation (§4.3) groups the other axis. What is new for their kernel is a different tile depth per question when a batch mixes queries.
- **What FoQLens would take as is.** Their calibration of the base slice (Appendix B, C.1) for a glass at 2 bits, and their shared always-on first slice (§4.2) as the glass under the lenses.

### What FoQLens takes

1. **The bench now.** Slices never change after quantization, so the weight of every read depth can be built once and kept (a bench mode behind a flag, bit for bit with the current reads), instead of unpacking four slices of 275 modules on every batch. See issue "Bench: build the depth weights once".
2. **The kernel later.** The output is linear in the slices (Eq. 3 and 6): the output of each slice can be computed once and summed with a mask per output row - in FoQLens the mask is the lens layout, per question and per block of rows. With a kernel that fetches only the required planes (§4.3, par. 1-2), memory, speed and energy follow the lenses.
3. **A glass at 2 bits.** A 2-bit base works when it is calibrated (Table 1, Appendix B, C.1); ours diverges uncalibrated (perplexity 9.4e6, [E006](../experiments/E006-read-depths/results.md)). Calibrating the first slice would let the frosted glass sit at D2 and halve its memory.
4. **The shared first slice is the glass.** Their always-on first slice (§4.2) is our glass: the minimum every block is read at, with the lenses adding slices on top.
5. **Precision varies by block** (§5.3) even under a per-token router - support for addressing precision by block, as FoQLens does.
6. **Mixed depths in a big batch are open** (Appendix F) - exactly the case of the FoQLens bench, where every question in a batch has its own layout.
7. **Lower precision can score higher** (§3) - a small observation in their setting, in the direction of H4.
