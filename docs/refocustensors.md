---
title: The .refocustensors model format
---

# The .refocustensors model format

A model is stored as one stack of refinements: any depth is read from one file, from the coarse base up to the top the model was cut to, and nothing is stored twice. A file cut in full ends in an `exact` tail and reads back the source weights bit for bit, in their own type - bf16, fp16 or fp32. So a folder with a `.refocustensors` file runs without the Hugging Face checkpoint, whatever type the model was published in.

## The stack

Every controlled weight matrix is stored as a stack. The base is at the bottom, the refinements above it, each refining everything under it. Depth is counted in 2-bit steps.

| Layer | Bits per weight | Read up to it | Level |
| --- | --- | --- | --- |
| base Q2_K | 2.625 | the base | D2 |
| refinement 1 | 2 | + refinement 1 | D4 |
| refinement 2 | 2 | + refinement 2 | D6 |
| refinement 3 | 2 | + refinement 3 | D8 |
| `exact` | 6.5 (3.8 under an entropy coder) | everything | the source bit for bit |

Example: a 2×256 bf16 matrix, the first weight of row 0.

```
source weight                    -0.001534
D2  base                         -0.008173
D4  + refinement 1               +0.000138
D6  + refinement 2               -0.001940
D8  + refinement 3               -0.001421
exact: round D8 to bf16 and move it by -15 neighbouring bf16 values  ->  -0.001534
```

## Cutting

The source is the most precise version of the model there is; everything is cut from it.

The base is k-quant from llama.cpp: Q2_K, and Q4_K for the sensitive classes (`k_proj`, `v_proj`, `o_proj`, `down_proj` and the per-layer modules). Their base takes two steps, so their D2 reads as D4. Refinement k stores a 2-bit correction code with step `base block step / 4^k` and has no scale of its own. `exact` stores by how many ulps of the source type to move the prediction - the weight read to the last refinement and rounded to the source type.

The base and the refinements are computed in fp32. bf16, fp16 and fp32 convert to fp32 without loss, so the cut is the same for any source type. Only `exact` depends on the type: the distance is counted in that type's ulps.

The bottom of the stack is a parameter, D2 (Q2_K) now. There may be any number of refinements: `refinement.<k>` are read in order while they last, and the format fixes no count; the cut lays them to D8 now.

Tensors the regulator does not read (embeddings, norms, the vision and audio towers) lie in the file as in the source. On E2B-it they are 6.04 of the checkpoint's 9.54 GiB, 4.7 GiB of it the per-layer embedding table.

## The file

The container is safetensors: a JSON header, then raw tensors. A tensor is read at its offset from the header, without reading the rest and without mapping the file into memory. What is the format's own is the tensor names and the metadata. For every controlled weight `<key>` - its name in the source:

| Tensor | Type and shape | What it holds |
| --- | --- | --- |
| `<key>.base` | uint8 [out, super-blocks, bytes] | the base as ggml's `block_q2_K` / `block_q4_K` bytes |
| `<key>.refinement.<k>` | uint8 [out, in / 4] | refinement k, four 2-bit codes to a byte |
| `<key>.exact.widths` | uint8 [out, in / 16] | the width of the distances in each group of 16 weights of a row |
| `<key>.exact` | uint8 [bytes] | the distances, group by group, row by row |

The base in ggml bytes takes 2.625 and 4.5 bits per weight: exactly the codes, the block scales and the super-block pair. A test checks these bytes against gguf-py's dequantizer. The `exact` distances are mapped to non-negative numbers (zigzag) and written in each group at the width of the largest of them. The bytes of a block of rows lie back to back, so a zone's block is read alone.

The file's metadata: the format's name and version, the source with its revision, and for every controlled weight its base format, shape and source type.

The prediction `exact` is counted from is the same for everyone who reads it. A test checks it bit for bit on the CPU and on the GPU.

## How to get it

```
uv run python scripts/download_models.py e2b-it
uv run python scripts/cut_model.py e2b-it
```

The first command downloads the checkpoint at its pinned revision, the second cuts it into `~/.cache/foqlens/models/gemma-4-E2B-it@3e22461f/`: the config, the tokenizer and `model.refocustensors`. `FOQLENS_HOME` moves that folder. The weights never go into the repository.

`--depth D8` (or D2, D4, D6) stops the stack at that level, with no `exact`: E2B-it cut to D8 is 7.92 GiB and goes to the folder next to it, `...@3e22461f-D8/`, never over the full model. Such a file is read resident only.

## Loading

The bench (`Bench.load`) takes its model from one of three sources:

| Source | What is read | Levels | GPU, E2B-it |
| --- | --- | --- | --- |
| `FILE`, the default | the source weights from the file bit for bit, every module's copy from the file | bf16, D2 ... D8 | 8.6 GiB and the copies as they are read |
| `RESIDENT` | the copies alone; no source weight of a controlled module is loaded | D2 ... D8 | 7.06 GiB, of it 1.86 the copies |
| `CHECKPOINT` | the Hugging Face checkpoint, the copy quantized from bf16 | bf16, D2 ... D8 | 8.6 GiB and the copies |

A model that was never cut is not loaded, and the error names the command that cuts it. The weights become the model's parameters without a copy: 13.5 GiB of Windows commit charge at the peak of a load. The bench computes masks at bf16, so a resident bench serves runs without masks. In the resident bench 4.7 GiB are the per-layer embedding table in bf16; the copies hold the base as ggml blocks and the refinements packed, as the file does.

The three sources give the same logits bit for bit at every depth, and `FILE` and `CHECKPOINT` also at bf16 - the logits of `from_pretrained`.

## Size

E2B-it: the file is 9.34 GiB, the bf16 checkpoint 9.54 GiB. 6.04 GiB of it are tensors the regulator does not read, lying as in the source.

The file cut at a level (`--depth`), and the share of bf16's knowledge kept on the frozen corpus (E002):

| Top of the stack | Controlled weights, bits per weight | Controlled weights, GiB | File, GiB | Kept |
| --- | --- | --- | --- | --- |
| D2 | 3.33 | 0.73 | 6.77 | 51.4% |
| D4 | 4.57 | 1.00 | 7.04 | 91.0% |
| D6 | 6.57 | 1.44 | 7.48 | 96.8% |
| D8 | 8.57 | 1.88 | 7.92 | 98.8% |
| the bf16 source (`exact`) | 15.05 | 3.30 | 9.34 | 100% |
| bf16 checkpoint | 16 | 3.51 | 9.54 | 100% |
| GGUF UD-Q2_K_XL / Q4_K_M | - | - | 2.24 / 2.89 | 80.0% / 95.0% |

D8 and the full file are measured by cutting, D2-D6 are summed from the sizes of the parts. GGUF files quantize the tensors the regulator does not read as well, so they are smaller; our controlled weights at D4 and D6 take 1.00 and 1.44 GiB.

The controlled weights (1.88 billion), bits per weight: the base and the refinements to D8 plus the tail to bf16 under different codings of the tail (`scripts/exact_tail_cost.py`):

| Coding of the tail | Total bits per weight |
| --- | --- |
| width per group of 256 weights | 20.13 |
| width per group of 32 | 15.92 |
| width per group of 16 - in the file | 15.05 |
| entropy coder | 12.39 |
| entropy coder with the prediction's exponent as context | 11.66 |
| bf16 alone, no base, entropy | 10.55 |

An entropy coder would cut the file by another ~0.6 GiB. The total hardly depends on the depth the tail starts after: 12.4 bits by entropy after D4, D6 and D8. The base is a function of the source, and each refinement takes about its own 2 bits off the tail.

## Sections and checkpoints

The stack can be cut into sections, each section a file of its own. Files are joined into one and split again by moving tensors, with nothing recomputed.

- The stack is always one chain: one base at the bottom, only refinements above it.
- Extending up (to bf16 or fp32) adds a refining section.
- Extending down cuts again: a new base and refinements, and the old base becomes an `exact` tail over them and reads back bit for bit.

A checkpoint is a level the stack must pass through bit for bit: a finished model, our own or someone else's. It lies either on a boundary of our refinements or on a format that has a block step (Q2_K ... Q6_K, Q8_0): the refinements above it take that step. IQ formats have no block step and cannot be checkpoints.

A level with no section downloaded for it is refused when the layout is set, before any computation.

## Variants the format supports

| Source | What is needed | Files | Bits per weight | What to build for it |
| --- | --- | --- | --- | --- |
| bf16 | the bench, zones to D8, the source for the judge | base ... D8, `exact` | 15.05 | built |
| bf16 | the bench at D2 ... D8 only, no source | base ... D8 | 8.57 | built (`--depth D8`) |
| bf16 | a small model, the full one on demand | base, D4 / D6, D8, `exact` | 4.6 / ~10.5 | sections |
| fp32 | zones to fp32, bf16 as a checkpoint on the way | base ... D8, `exact` to bf16, `exact` to fp32 | not measured | a chain of tails |
| fp32 | a ceiling of 16 bits: the model read to bf16 at most | base ... D8, `exact` to bf16; the fp32 section is not downloaded | not measured | a chain of tails |
| someone else's Q8_0, nothing else | base precision below 8 bits | base ... D6, `exact` to Q8_0 | ~8.5, not measured | a Q8_0 port, a block-quantized source |
| fp8 with block scales | the same | base ... D6, `exact` to fp8 | not measured | a block-quantized source |
| bf16 and someone else's Q2_K | D2 exactly as published, zones to bf16 | the foreign Q2_K as the base, D4 ... D8, `exact` | not measured | nothing if every tensor is Q2_K or Q4_K; otherwise Q3_K, Q5_K, Q6_K ports |
| someone else's Q2_K only | running it | base | 2.6 | nothing; there is no information above D2 |

## Memory for the layout, and ZERO

The aim is to hold on the GPU only what the layout reads. Base precision D2 and D8 zones on 10% of the blocks: 0.9 · 3.33 + 0.1 · 8.57 ≈ 3.9 bits per weight. With base precision ZERO outside the zones ≈ 0.9 bits.

ZERO saves computation now: the kernel reads nothing for a block at ZERO and its output is zero ([kernels](kernels.md)), while the module's copy lies in memory whole. ZERO saves memory once two things exist: a depth cap per block for the k-quant copy (a block capped at 0 stores nothing) and reading the file by blocks for the layout, loading and unloading them as it changes. The file is ready for it: the rows of a block lie back to back in every tensor, and a tensor is read at its offset.

## What exists now

Built: the Q2_K / Q4_K base after llama.cpp, refinements to D8, the `exact` tail for bf16, fp16 and fp32, writing a model folder, cutting it at a level, and the bench's three sources; a layout of depths is read by a CUDA kernel straight from the copy's bytes ([kernels](kernels.md)). The model from the file matches the checkpoint bit for bit, every module's copy matches the copy quantized from bf16, and the ladder on the corpus matches E002. Sections, extending down, reading by blocks for the layout, ports of other formats and block-quantized sources come later.
