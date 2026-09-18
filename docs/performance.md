---
title: Performance of the bench
---

# Performance of the bench

What the bench costs on its one card - an RTX 3090 Ti, 400 W, the limits in [the station](station.md) - and how that changed. Every number names its source: the commit that measured it or the run it comes from. The current tables of a module are in [the kernel](kernels.md); the time, load and temperature of every run and GPU test session are in the log of [the station](station.md).

Measured by the bench's own scripts:

- `scripts/decode_step_speed.py` - a graphed decoding step of the model at each way of reading its weights;
- `scripts/kernel_speed.py` - one 12288×1536 module, bf16, unpacking and both kernels, 1-256 tokens;
- `GpuMonitor` in every run summary and GPU test session - time, utilization, temperature, power (`scripts/station_log.py` builds the log).

## A decoding step

E2B-it, one step, ms:

| Date | What changed | bf16 | Quantized | Source |
| --- | --- | --- | --- | --- |
| 2026-09-14 | our own loop over HF's cache, 128 short prompts | 251 | - | d6f40a5 |
| 2026-09-14 | a step is one CUDA graph over a static cache, 128 short prompts | 41 | - | d6f40a5; GPU 21% -> 100% |
| 2026-09-15 | a uniform level is baked into the weights once | - | the step of bf16 | 2e6ed30 |
| 2026-09-16 | decode steps on the memory-efficient sdpa kernel, 29 / 102 rows | 16.6 / 23.3 against 35.0 / 132.8 on math | - | d4e14b8 |
| 2026-09-18 | a mixed layout D2-D8, batch 1 / 8 / 32, unpacked | 11.6 / 15.5 / 20.7 | 401.5 / 414.1 / 425.2 | 87e7fff |
| 2026-09-18 | the same, the k-quant kernel on CUDA cores | 11.6 / 15.5 / 20.7 | 17.7 / 20.4 / 51.0 | 87e7fff |
| 2026-09-18 | the same, on tensor cores | 11.5 / 15.3 / 20.4 | 17.8 / 19.4 / 28.8 | 97e655a |

A uniform level in the corpus runs is baked and costs the step of bf16. A zone layout cannot be baked: it went from 425 ms at batch 32 to 28.8, 1.4 times the step of bf16.

## A module

12288×1536, 8 tokens, the copy read to D8, ms:

| Date | Reading | Time | Source |
| --- | --- | --- | --- |
| 2026-09-18 | unpacking for a GEMM | 3.8-4.0 | 87e7fff, 97e655a |
| 2026-09-18 | the kernel on CUDA cores, timed from the host | 0.21 | 87e7fff |
| 2026-09-18 | the kernel on CUDA cores, in a CUDA graph | 0.106 | 97e655a |
| 2026-09-18 | the kernel on tensor cores, in a CUDA graph | 0.103 | 97e655a |
| 2026-09-18 | bf16 on cuBLAS, in a CUDA graph | 0.045 | 97e655a |

Timed from the host a small kernel measures its launch, 0.12 ms for either kernel at any depth, so the times before 97e655a overstate the kernel. At 32 tokens the tensor cores take 0.137 ms against 0.402 on CUDA cores ([the kernel](kernels.md)).

## The corpus

A pass over the frozen corpus, 20,640 questions of E2B-it:

| Date | Run | Time | Source |
| --- | --- | --- | --- |
| 2026-09-14 | a round of stage 1, dynamic loop | ~11 min | d6f40a5 |
| 2026-09-14 | the same round, graphed | 208 s | d6f40a5 |
| 2026-09-15 | a smoke round at D4, unpacked at every call | ~20 min, against ~4.5 at bf16 | 2e6ed30 |
| 2026-09-16 | a bf16 round of E001, all attention on math | 8.65 min | d4e14b8 |
| 2026-09-16 | the same round, decode steps on the memory-efficient kernel | 3.66 min | d4e14b8 |
| 2026-09-16 | E001, one level answered: D6 / D4 / D2 | 52 min / 54 min / 1 h 40 min | station log |
| 2026-09-16 | E001, one level judged | 16 min | station log |
| 2026-09-17 | E002, the k-quant ladder, one level answered: D4 / D6 / D8 | 54 / 55 / 55 min | station log |

D2 of E001 took 1 h 40 min: it stopped on its own in 13.9% of the answers and ran the rest to the limit of 512 tokens ([E001](../experiments/E001-uniform-quantization/results.md)). The corpus runs keep the card at 78-85% on average.

## Memory and loading

| Date | What | Value | Source |
| --- | --- | --- | --- |
| 2026-09-12 | one sliced copy in place of the bf16 weights | -1.63 GiB on E2B | the plan, *Updated 2026-09-12* |
| 2026-09-18 | a load at its peak of Windows commit charge: assigning into a model on meta instead of from_pretrained | 13.5 GiB instead of 22 | fe4ba05 |
| 2026-09-18 | the resident bench of E2B-it: load, peak of commit charge | 13.1 s, 12.5 GiB | fb545c4 |
| 2026-09-18 | the resident bench of E2B-it on the GPU | 8.3 GiB, then 7.06 with the base held as ggml blocks; bf16 8.6 | fb545c4, b3eaa11 |
| 2026-09-18 | one 12288×1536 module's copy | 34.0 -> 19.4 MiB | b3eaa11 |
| 2026-09-18 | the file of E2B-it | 9.34 GiB with the tail to bf16, 7.92 cut to D8, against 9.54 of bf16 | fe4ba05, [the format](refocustensors.md) |

## The GPU tests

The two longest test files, minutes a session (from `runs/station/tests`):

| Date | test_graph_decode_gpu | test_kquant_ladder_gpu | What changed |
| --- | --- | --- | --- |
| 2026-09-14 | 16 | - | the graph decoder |
| 2026-09-16 | 18-19 | - | the prefill on the math kernel |
| 2026-09-17 | 20 | 15 | the k-quant copy; the ladder unpacks at every step |
| 2026-09-18 | 19.5 | 13.9 | the model read from its .refocustensors file |
| 2026-09-18 | 18.4 | 3.7 | the kernel on CUDA cores |
| 2026-09-18 | 19.5 | 3.0 | tensor cores; the graph test reads two more layouts through the kernel |

The graph test barely moves with the kernel: 6 of its 25 cases read through it, the rest run at bf16 or unpack a layout with bf16 blocks, and much of the time is the prefill and the eager step each case is held to. The ladder test reads the model by blocks at four depths, the path of the zones, and runs 4.6 times faster.
