---
title: The kernel that reads the copy
---

# The kernel that reads the copy

The kernel `kquant_mma` (`src/foqlens/kernels/kquant_mma.cu`) multiplies the input by the k-quant copy straight from its bytes, on tensor cores: every block of 64 output rows is read to its own depth, and a block at ZERO is not read at all. A zone layout is exactly the depths of the blocks, so a module under a layout is computed by the kernel without unpacking the weight.

## When the bench goes through the kernel

`MixedPrecisionLinear` hands the multiplication to the kernel when all of these hold:

- the module's copy is k-quant (`KRefinedWeight`), on a Q2_K or Q4_K base;
- every level of the layout is a depth D2 ... D8 or ZERO;
- a block is 64 rows (`DEFAULT_BLOCK_ROWS`);
- there are no more than `KERNEL_MAX_TOKENS` = 256 tokens: the decoding steps and short inputs.

The bf16 level and a level baked into the weight (`bake`, the way the bench answers in the corpus runs) take the old path. A long input - a prefill - unpacks the weight once and multiplies it on cuBLAS. The flag `precision.KERNEL` turns the kernel off entirely.

A layout may be one for the batch or one per sample (`set_layout` with per-sample layouts): the kernel gets a depth per token, and the tokens of a sample read its layout.

## What the kernel reads

The kernel is built on the `mma.sync` instruction m16n8k16: 16 rows of weights times 8 tokens over 16 columns, bf16 in, fp32 out. A step of 16 columns is one Q2_K block or half a Q4_K block, so a row has one scale and one min in it.

A thread block covers 16 rows - inside one block of 64, so every token reads them at one depth - and 32 tokens. Its warps split the input: warp k takes super-blocks (256 columns) k, k + split, ..., where the split is the largest divisor of the super-blocks up to 8. A decoding step has few tokens, and without the split a module would give the card too few warps to hide its reads. A warp copies its rows of a super-block into shared memory as whole words - the base block (`block_q2_K` / `block_q4_K`) and one plane per refinement up to the deepest depth any token reads, none deeper - and takes the super-block in 16 steps.

In a step a thread builds the eight weights its part of the mma needs, rows g and g+8 at columns 2q, 2q+1, 2q+8, 2q+9 (g = lane / 4, q = lane % 4): two 16-bit reads of base codes and one word per refinement. The weights are refined depth by depth, and at every depth some token reads the mma runs over the token tiles; a token that reads another depth enters with a zero input and adds exact zeros. The tokens' inputs are read straight from global memory, where a decoding step's few rows stay in cache. At the end the warps' partial sums are added in the order of the warps.

The first kernel, `kquant_matmul.cu`, multiplies on CUDA cores - a warp per output row - and stays as the reference the tensor cores are held to (`kernels.kquant.CUDA_CORES`).

## Accuracy

A weight is built with the operations of the torch path: the base `d*scale*code - dmin*min`, then `+ step/4^k * (code - 1.5)` per refinement, every operation rounded on its own with no fused multiply-add, and the weight is rounded to bf16 before the multiplication, as `F.linear` on the bf16 weight does. The weights come out bit for bit, and a product of two bf16 is exact in fp32; what differs from cuBLAS is how the products are summed. Against the same sum in fp64 the kernel is off by at most 2.9e-6 of the largest output: an mma adds its products with truncation (Fasi et al. 2021, "Numerical behavior of NVIDIA tensor cores"). That is a thousandth of the bf16 step the module rounds its output to. On CUDA cores the bound is 1.6e-7.

A token's output does not depend on the other tokens of the batch: a token's column of an mma depends on its own inputs only, and the order of every sum is fixed by the shapes.

## Speed

A 12288×1536 module (E2B's widest), Q2_K base, RTX 3090 Ti, ms in a CUDA graph (`scripts/kernel_speed.py`):

| Tokens | bf16, cuBLAS | Unpacking to D8 and GEMM | CUDA cores, D2 / D8 | Tensor cores, D2 / D8 |
| --- | --- | --- | --- | --- |
| 1 | 0.045 | 3.81 | 0.078 / 0.093 | 0.056 / 0.100 |
| 8 | 0.045 | 3.80 | 0.091 / 0.106 | 0.060 / 0.103 |
| 16 | 0.056 | 3.81 | 0.172 / 0.200 | 0.075 / 0.116 |
| 32 | 0.067 | 3.83 | 0.333 / 0.402 | 0.101 / 0.137 |
| 64 | 0.068 | 3.85 | 0.664 / 0.806 | 0.209 / 0.275 |
| 128 | 0.109 | 3.88 | 1.327 / 1.610 | 0.386 / 0.522 |
| 256 | 0.210 | 3.98 | 2.646 / 3.265 | 0.749 / 1.018 |

Timed from the host, a small kernel measures its launch instead: 0.12 ms for either kernel at any depth.

The tensor cores read D2 faster than the CUDA cores at every token count and D8 from 16 tokens on, 3 times faster at 32. At one token they are 1.2 times slower than bf16 at D2 and 2.2 times at D8: the three refinements of D8 add 0.044 ms of reading and decoding to the base.

A decoding step of E2B-it, one CUDA graph, ms (`scripts/decode_step_speed.py`; the mixed layout draws D2 ... D8 per block):

| Reading | Batch 1 | Batch 8 | Batch 32 |
| --- | --- | --- | --- |
| bf16 | 11.5 | 15.3 | 20.4 |
| D4, unpacked | 156.3 | 160.1 | 165.7 |
| D4, kernel | 15.6 | 17.3 | 27.0 |
| mixed, unpacked | 401.7 | 413.7 | 422.6 |
| mixed, kernel | 17.8 | 19.4 | 28.8 |
| mixed, CUDA cores | 17.7 | 20.4 | 51.0 |

A zone layout through the kernel costs about as much as a uniform depth and is 15-23 times faster than unpacking; against bf16 it is 1.5 times slower at one row and 1.4 times at 32. Read by blocks through the kernel, the ladder of `test_kquant_ladder_gpu` answers its 102 questions at four depths in 3.0 minutes, against 3.7 on CUDA cores and 13.9 unpacked.

## In the decoding graph

`graph_decode` captures a decoding step as a CUDA graph again for every batch, at the batch's layout. The kernel is launched through the driver API (`cuLaunchKernel`) on the current stream and lands in the graph; the blocks' depths live in the module while the layout stands. The first step runs without the graph, so the kernel is compiled and loaded before the capture.

## Checks

`tests/test_kquant_kernel_gpu.py`, for both kernels: every depth of both formats against the torch path; input widths of 256, 1536, 2048 and 12288 (a split of 1, 6 and 8 warps); a mixed layout with ZERO; a layout per token, and each token as it reads alone; 70 tokens at their own layouts; a copy cut short by depth. For the bench's kernel: a module through the kernel against unpacking, per-sample layouts and `read_samples`; a long input goes to unpacking; bf16 and a baked level bypass the kernel; the kernel is far faster than unpacking and a shallower read is not slower.

`tests/test_graph_decode_gpu.py`: the graph captures layouts of depths per block and per sample read by the kernel, and writes what the same step writes eagerly, token for token. `tests/test_kquant_ladder_gpu.py`: the model read by blocks through the kernel answers the committed questions of E002 as its ladder did.

## Next

Reading blocks from the file by the layout, and ZERO without memory, are in the [model format](refocustensors.md).
