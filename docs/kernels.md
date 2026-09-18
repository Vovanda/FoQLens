---
title: The kernel that reads the copy
---

# The kernel that reads the copy

The kernel `kquant_matmul` (`src/foqlens/kernels/kquant_matmul.cu`) multiplies the input by the k-quant copy straight from its bytes: every block of 64 output rows is read to its own depth, and a block at ZERO is not read at all. A zone layout is exactly the depths of the blocks, so a module under a layout is computed by the kernel without unpacking the weight.

## When the bench goes through the kernel

`MixedPrecisionLinear` hands the multiplication to the kernel when all of these hold:

- the module's copy is k-quant (`KRefinedWeight`), on a Q2_K or Q4_K base;
- every level of the layout is a depth D2 ... D8 or ZERO;
- a block is 64 rows (`DEFAULT_BLOCK_ROWS`);
- there are no more than `KERNEL_MAX_TOKENS` = 256 tokens: the decoding steps and short inputs.

The bf16 level and a level baked into the weight (`bake`, the way the bench answers in the corpus runs) take the old path. A long input - a prefill - unpacks the weight once and multiplies it on cuBLAS. The flag `precision.KERNEL` turns the kernel off entirely.

A layout may be one for the batch or one per sample (`set_layout` with per-sample layouts): the kernel gets a depth per token, and the tokens of a sample read its layout.

## What the kernel reads

One warp works on one output row; its 32 lanes split the row into format blocks - 16 weights for Q2_K, 32 for Q4_K. A lane reads whole words:

- the bytes of the base block (`block_q2_K` / `block_q4_K`): the codes, the block's scale and min, the super-block's multipliers;
- one word per refinement up to its depth - and none deeper;
- the tokens' inputs.

The 8 warps of a thread block are 8 neighbouring rows of one block of 64, so they share one depth. They share the tokens' inputs too: these are loaded into shared memory a chunk of 2048 columns at a time. A warp carries 8 tokens at once.

## Accuracy

A weight is built with the operations of the torch path: the base `d*scale*code - dmin*min`, then `+ step/4^k * (code - 1.5)` per refinement, every operation rounded on its own with no fused multiply-add, and the weight is rounded to bf16 before the multiplication, as `F.linear` on the bf16 weight does. The weights come out bit for bit; only the order of the sum over the input differs. Against the same sum in fp64 the kernel is off by at most 1.6e-7 of the largest output.

A token's output does not depend on the other tokens of the batch: every token gets the same order of operations in any batch.

## Speed

A 12288×1536 module (E2B's widest), Q2_K base, RTX 3090 Ti, ms:

| Tokens | bf16, cuBLAS | Unpacking to D8 and GEMM | Kernel, D2 | Kernel, D8 |
| --- | --- | --- | --- | --- |
| 1 | 0.084 | 4.03 | 0.202 | 0.200 |
| 8 | 0.080 | 3.92 | 0.204 | 0.208 |
| 16 | 0.077 | 3.94 | 0.201 | 0.213 |
| 32 | 0.077 | 3.96 | 0.342 | 0.399 |
| 64 | 0.084 | 3.96 | 0.668 | 0.816 |
| 128 | 0.115 | 3.99 | 1.350 | 1.601 |

Against unpacking the kernel is 20 times faster on a module. Dense bf16 on tensor cores is still 2.5 times faster than the kernel up to 16 tokens and 10 times at 64: the kernel computes on CUDA cores and decodes the weight again for every 8 tokens. Its time hardly depends on the depth: arithmetic sets it, and the bytes it reads hardly count.

A decoding step of E2B-it, one CUDA graph, ms (`scripts/decode_step_speed.py`; the mixed layout draws D2 ... D8 per block):

| Reading | Batch 1 | Batch 8 | Batch 32 |
| --- | --- | --- | --- |
| bf16 | 11.6 | 15.5 | 20.7 |
| D4, unpacked | 158.0 | 161.4 | 166.9 |
| D4, kernel | 17.3 | 19.9 | 49.5 |
| mixed, unpacked | 401.5 | 414.1 | 425.2 |
| mixed, kernel | 17.7 | 20.4 | 51.0 |

A zone layout through the kernel costs as much as a uniform depth and is 23 times faster than unpacking; it is 1.5 times slower than bf16 at one row and 2.5 times at 32. Read by blocks through the kernel, the ladder of `test_kquant_ladder_gpu` answers its 102 questions at four depths in 3.7 minutes against 13.9 unpacked.

## In the decoding graph

`graph_decode` captures a decoding step as a CUDA graph again for every batch, at the batch's layout. The kernel is launched through the driver API (`cuLaunchKernel`) on the current stream and lands in the graph; the blocks' depths live in the module while the layout stands. The first step runs without the graph, so the kernel is compiled and loaded before the capture.

## Checks

`tests/test_kquant_kernel_gpu.py`: every depth of both formats against the torch path; a mixed layout with ZERO; a layout per token; a copy cut short by depth; a module through the kernel against unpacking, per-sample layouts and `read_samples`; a long input goes to unpacking; bf16 and a baked level bypass the kernel; the kernel is faster than unpacking.

`tests/test_graph_decode_gpu.py`: the graph captures layouts of depths per block and per sample read by the kernel, and writes what the same step writes eagerly, token for token. `tests/test_kquant_ladder_gpu.py`: the model read by blocks through the kernel answers the committed questions of E002 as its ladder did.

## Next

Decode a tile of weights into shared memory as bf16 and multiply it on tensor cores (mma) - this is how the kernel catches up with dense bf16 on batches from 16 tokens. Reading blocks from the file by the layout, and ZERO without memory, are in the [model format](refocustensors.md).
