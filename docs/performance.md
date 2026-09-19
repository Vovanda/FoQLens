---
title: Performance of the bench
---

# Performance of the bench

**The bench is ready for zones.** A zone layout - every block of 64 rows of every module at its own depth, D2 ... D8 or ZERO - is multiplied by one kernel straight from the model's copy, on tensor cores ([the kernel](kernels.md)). A decoding step of E2B-it at such a layout takes 28.8 ms at a batch of 32: as much as a uniform depth (27.0), 1.4 times bf16 (20.4), and 15 times faster than unpacking the weights (422.6).

The kernel speeds up what could not be baked. A uniform level - every corpus run of the ladder - is baked into the
weights once and multiplied by cuBLAS at the speed of bf16 (2e6ed30), where the kernel would be slower (0.045 ms a
module against 0.056-0.10). On the corpus a round of D4 (1 029 questions) took 157 s baked and 199 s read by blocks
through the kernel (E003, 38757ad). The GPU tests show the same split: the ladder test, which reads the model by
blocks, went from 13.9 to 3.0 minutes, the graph test, mostly
at bf16, stays at 18-20.

A decoding step of E2B-it, one CUDA graph, ms (`scripts/decode_step_speed.py`, a6c655b; the mixed layout draws D2 ... D8 per block):

| Reading | Batch 1 | Batch 8 | Batch 32 |
| --- | --- | --- | --- |
| bf16 | 11.5 | 15.3 | 20.4 |
| uniform D4, kernel | 15.6 | 17.3 | 27.0 |
| **mixed layout, kernel** | **17.8** | **19.4** | **28.8** |
| mixed layout, unpacked | 401.7 | 413.7 | 422.6 |

The model read by blocks at four depths answers the 102 questions of `test_kquant_ladder_gpu` in 3.0 minutes against 13.9 unpacked, and as E002's ladder did.

All figures: RTX 3090 Ti at 400 W ([the station](station.md)); every number names the commit that measured it.

## How it got here

| Date | Change | Effect | Commit |
| --- | --- | --- | --- |
| 2026-09-14 | a decoding step is one CUDA graph over a static cache | 251 -> 41 ms a step at bf16, 128 prompts; a corpus round 11 min -> 208 s | d6f40a5 |
| 2026-09-15 | a uniform level is baked into the weights once | the corpus runs decode at the step of bf16 | 2e6ed30 |
| 2026-09-16 | decode steps on the memory-efficient attention kernel | a bf16 round of E001 8.65 -> 3.66 min | d4e14b8 |
| 2026-09-18 | a mixed layout read by a kernel on CUDA cores | 425.2 -> 51.0 ms a step at batch 32 | 28b0e10 |
| 2026-09-18 | the kernel on tensor cores | 51.0 -> 28.8 ms | a6c655b |

## Memory

| What | Value | Commit |
| --- | --- | --- |
| the file of E2B-it | 9.34 GiB with the tail to bf16, 7.92 cut to D8, against 9.54 of bf16 | fe4ba05 |
| the resident bench on the GPU: D2 ... D8, no bf16 | 7.06 GiB against 8.6 at bf16 | c90f244 |
| a load at its peak of Windows commit charge | 13.5 GiB instead of 22 | fe4ba05 |

The copy lies in memory whole: reading only what the layout reads is not built yet ([the model format](refocustensors.md)).

## A corpus level

A level answered on the frozen corpus of 20,640 questions takes 54-55 min (E002's ladder) and is judged in 16 min (E001), by [the station's log](station.md); the corpus runs keep the card at 78-85% on average.
