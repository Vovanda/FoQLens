---
title: The station
---

# The station

The one machine every run of the bench has been made on, the limits it runs under, and a log of what the
runs cost it.

## Hardware

| Part | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090 Ti, 24 GB; driver 591.86 |
| CPU | AMD Ryzen 7 2700, 8 cores / 16 threads |
| Memory | 32 GB, 2 modules at 2400 MT/s |
| Board | ASRock B450 Pro4 |
| Case | DeepCool, three case fans on the board |
| Disks | Kingston KC3000 1 TB (NVMe), Kingston A400 960 GB, WD Green 120 GB |
| OS | Windows 11 Pro 10.0.22631 |

## Limits

| Limit | Value | Set by |
| --- | --- | --- |
| GPU share of a run | 0.8: that share of the VRAM, and a rest after every batch so the card is busy 80% of the time | `--gpu-share`, `src/foqlens/gpu_share.py` |
| Temperature ceiling | 80 °C: from 75 °C the rest after a batch grows, at 80 °C the run pauses until 72 °C | `ThermalGuard`, same file |
| Cooling break | 5 minutes after every hour of running | `Cooldown`, same file |
| Power limit | 400 W of the default 450 W (the card allows 100-480 W); 330 W until 2026-09-15 | `nvidia-smi -pl 400`, reapplied at startup and at logon by the scheduled task `Station GPU limits` - the driver forgets it on reboot |
| Driver target temperature | 78 °C (the default is 83 °C; the card slows at 94 °C and shuts down at 97 °C) | `nvidia-smi -gtt 78`, the same task |
| Fans | case fans and the two GPU fans on one curve of the GPU temperature: 40 °C 20%, 50 °C 40%, 65 °C 70%, 75 °C 90%, 80 °C 100%; the GPU fans do not go below 30%; the CPU fan stays with the board | Fan Control |

The ceiling of 80 °C is the owner's for this machine, below the card's own target.

## How the log is kept

Every run summary records the GPU through `GpuMonitor` - when the run started, how long it took, mean
utilization, mean and peak temperature, peak power - and the pacing through `"pacer"`, with the cooling
breaks taken. A session of the GPU tests leaves the same record in `runs/station/tests/`. A run stopped
before it wrote its summary is entered by hand in `runs/station/`, and its note says so.

`scripts/station_log.py` rebuilds the log and the peaks below from all of them. A time marked ~ is an
estimate: a summary written before the monitor kept a clock is counted by its samples, one a second.

<!-- station-log:begin -->
## Log

| Date | Run | Time | Utilization, mean | Temperature, mean / peak | Power, peak | Cooling breaks | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-13 | reference/address-stability/e2b | ~6 min | 58% | - / - | - | - |  |
| 2026-09-13 | reference/shuffle-answers-canonical/e2b | ~6 min | 56% | - / - | - | - |  |
| 2026-09-13 | reference/shuffle-answers-fixed-address/e2b | ~4 min | 68% | - / - | - | - |  |
| 2026-09-13 | reference/shuffle-answers-question-only-matched/e2b | ~5 min | 60% | - / - | - | - |  |
| 2026-09-13 | reference/shuffle-answers-question-only/e2b | ~5 min | 56% | - / - | - | - |  |
| 2026-09-13 | reference/shuffle-answers/e2b | ~6 min | 60% | - / - | - | - |  |
| 2026-09-14 | GPU tests | 2 min | 14% | 47 / 54 °C | 265 W | - | 2 passed; the injection and filter smokes, entered by hand from their report |
| 2026-09-14 | GPU tests | 8 min | 16% | 47 / 57 °C | 277 W | - | 61 passed; the first session under the thermal guard, entered by hand from its report |
| 2026-09-14 | GPU tests | 1 min | 5% | 45 / 47 °C | 173 W | - | 1 passed |
| 2026-09-14 | GPU tests | 5 min | 33% | 52 / 59 °C | 301 W | - | 6 passed, 1 failed |
| 2026-09-14 | GPU tests | 1 min | 22% | 50 / 56 °C | 245 W | - | 4 passed |
| 2026-09-14 | GPU tests | 1 min | 18% | 49 / 55 °C | 292 W | - | 4 passed |
| 2026-09-14 | GPU tests | 1 min | 8% | 46 / 48 °C | 151 W | - | 3 passed, 1 failed |
| 2026-09-14 | GPU tests | 1 min | 6% | 47 / 50 °C | 152 W | - | 4 passed |
| 2026-09-14 | GPU tests | 1 min | 37% | 49 / 57 °C | 330 W | - | 1 passed, 3 failed |
| 2026-09-14 | GPU tests | 2 min | 49% | 53 / 60 °C | 334 W | - | 2 passed, 2 failed |
| 2026-09-14 | GPU tests | 7 min | 22% | 50 / 60 °C | 334 W | - | 21 passed, 1 failed |
| 2026-09-14 | GPU tests | 1 min | 24% | 49 / 57 °C | 330 W | - | 1 passed |
| 2026-09-14 | GPU tests | 16 min | 53% | 56 / 65 °C | 335 W | - | 22 passed, 1 failed |
| 2026-09-14 | calibration of the letter-choice corpora, share 0.8, before the thermal guard | ~25 min | 100% | - / 83 °C | 403 W | - | entered by hand from nvidia-smi, the run was stopped before its summary: a smell of burnt dust; the dust was blown out, the limits were set |
| 2026-09-15 | 2026-09-15-e001-answer-d8-collapsed | 47 min | 83% | 63 / 68 °C | 413 W | 0 | E001 before issue #14: the memory-efficient sdpa kernel collapsed whole batches; answers moved out of the repo, run again |
| 2026-09-15 | 2026-09-15-e001-judge-bf16-collapsed | 12 min | 84% | 67 / 71 °C | 405 W | 0 | E001 before issue #14: the memory-efficient sdpa kernel collapsed whole batches; answers moved out of the repo, run again |
| 2026-09-15 | 2026-09-15-e001-judge-d8-collapsed | 16 min | 84% | 67 / 70 °C | 406 W | 0 | E001 before issue #14: the memory-efficient sdpa kernel collapsed whole batches; answers moved out of the repo, run again |
| 2026-09-15 | E001-uniform-quantization/e2b-it/summary-bf16 | 9 min | 70% | 60 / 67 °C | 402 W | 0 |  |
| 2026-09-15 | E001-uniform-quantization/e2b-it/summary-d8 | 5 min | 88% | 65 / 69 °C | 399 W | 0 |  |
| 2026-09-15 | GPU tests | 1 min | 10% | 51 / 55 °C | 229 W | - | 4 passed |
| 2026-09-15 | GPU tests | 0 min | 0% | 51 / 51 °C | 107 W | - | 8 passed |
| 2026-09-15 | GPU tests | 0 min | 0% | 50 / 50 °C | 107 W | - | 2 passed |
| 2026-09-15 | GPU tests | 1 min | 14% | 50 / 56 °C | 261 W | - | 2 passed |
| 2026-09-15 | GPU tests | 0 min | 0% | 48 / 49 °C | 106 W | - | 20 passed |
| 2026-09-15 | GPU tests | 1 min | 7% | 48 / 48 °C | 120 W | - | 7 passed |
| 2026-09-15 | GPU tests | 0 min | 7% | 47 / 51 °C | 225 W | - | 1 passed |
| 2026-09-15 | GPU tests | 3 min | 15% | 49 / 56 °C | 273 W | - | 4 passed |
| 2026-09-15 | GPU tests | 1 min | 10% | 49 / 54 °C | 275 W | - | 9 passed |
| 2026-09-15 | GPU tests | 1 min | 7% | 44 / 49 °C | 282 W | - | 4 passed |
| 2026-09-15 | GPU tests | 15 min | 53% | 57 / 72 °C | 400 W | - | 23 passed |
| 2026-09-15 | GPU tests | 4 min | 13% | 51 / 60 °C | 328 W | - | 4 passed |
| 2026-09-15 | GPU tests | 1 min | 11% | 46 / 50 °C | 198 W | - | 4 passed |
| 2026-09-15 | GPU tests | 1 min | 11% | 47 / 53 °C | 258 W | - | 4 passed, 1 failed |
| 2026-09-15 | GPU tests | 1 min | 13% | 47 / 54 °C | 266 W | - | 5 passed |
| 2026-09-15 | GPU tests | 1 min | 18% | 48 / 56 °C | 337 W | - | 1 passed |
| 2026-09-15 | GPU tests | 1 min | 7% | 48 / 53 °C | 272 W | - | 2 passed |
| 2026-09-15 | GPU tests | 1 min | 7% | 46 / 48 °C | 181 W | - | 2 passed |
| 2026-09-15 | reference/prompt-tuning/e2b-it | 7 min | 13% | 45 / 56 °C | 374 W | 0 |  |
| 2026-09-15 | reference/stage1/e2b-it/summary-bf16 | 3 min | 82% | 60 / 66 °C | 400 W | 0 |  |
| 2026-09-16 | E001-uniform-quantization/bf16-unknown-again/e2b-it/summary-bf16 | 19 min | 78% | 59 / 66 °C | 400 W | 0 |  |
| 2026-09-16 | E001-uniform-quantization/e2b-it/summary-d2 | 1 h 40 min | 82% | 61 / 67 °C | 404 W | 1 |  |
| 2026-09-16 | E001-uniform-quantization/e2b-it/summary-d4 | 54 min | 82% | 64 / 69 °C | 407 W | 0 |  |
| 2026-09-16 | E001-uniform-quantization/e2b-it/summary-d6 | 52 min | 83% | 64 / 68 °C | 408 W | 0 |  |
| 2026-09-16 | E001-uniform-quantization/e2b-it/summary-judged-bf16 | 16 min | 84% | 67 / 70 °C | 400 W | 0 |  |
| 2026-09-16 | E001-uniform-quantization/e2b-it/summary-judged-d2 | 16 min | 85% | 68 / 70 °C | 400 W | 0 |  |
| 2026-09-16 | E001-uniform-quantization/e2b-it/summary-judged-d4 | 16 min | 85% | 67 / 70 °C | 401 W | 0 |  |
| 2026-09-16 | E001-uniform-quantization/e2b-it/summary-judged-d6 | 16 min | 85% | 67 / 70 °C | 400 W | 0 |  |
| 2026-09-16 | E001-uniform-quantization/e2b-it/summary-judged-d8 | 16 min | 85% | 67 / 70 °C | 400 W | 0 |  |
| 2026-09-16 | GPU tests | 1 min | 12% | 48 / 57 °C | 325 W | - | 5 passed |
| 2026-09-16 | GPU tests | 14 min | 50% | 58 / 69 °C | 398 W | - | 22 passed, 1 failed |
| 2026-09-16 | GPU tests | 1 min | 9% | 51 / 56 °C | 278 W | - | 5 passed |
| 2026-09-16 | GPU tests | 5 min | 12% | 50 / 59 °C | 280 W | - | 4 passed |
| 2026-09-16 | GPU tests | 1 min | 21% | 49 / 58 °C | 363 W | - | 6 passed |
| 2026-09-16 | GPU tests | 1 min | 7% | 45 / 46 °C | 124 W | - | 8 passed |
| 2026-09-16 | GPU tests | 1 min | 8% | 46 / 46 °C | 123 W | - | 4 passed |
| 2026-09-16 | GPU tests | 18 min | 57% | 58 / 69 °C | 399 W | - | 22 passed, 1 failed |
| 2026-09-16 | GPU tests | 18 min | 61% | 59 / 69 °C | 399 W | - | 23 passed |
| 2026-09-16 | GPU tests | 1 min | 15% | 55 / 63 °C | 375 W | - | 6 passed |
| 2026-09-16 | GPU tests | 1 min | 21% | 47 / 52 °C | 275 W | - | 3 passed |
| 2026-09-16 | GPU tests | 1 min | 36% | 50 / 61 °C | 375 W | - | 6 passed |
| 2026-09-16 | GPU tests | 19 min | 58% | 59 / 69 °C | 400 W | - | 23 passed |
| 2026-09-17 | E002-base-precision-d2/gguf-q4_k_m/e2b-it/summary-d2 | 50 min | 83% | 63 / 68 °C | 404 W | 0 |  |
| 2026-09-17 | E002-base-precision-d2/gguf-ud-q2_k_xl/e2b-it/summary-d2 | 1 h 07 min | 78% | 61 / 67 °C | 403 W | 1 |  |
| 2026-09-17 | E002-base-precision-d2/ladder/e2b-it/summary-d2 | 12 min | 80% | 60 / 66 °C | 401 W | 0 |  |
| 2026-09-17 | E002-base-precision-d2/ladder/e2b-it/summary-d4 | 54 min | 82% | 63 / 68 °C | 404 W | 0 |  |
| 2026-09-17 | E002-base-precision-d2/ladder/e2b-it/summary-d6 | 55 min | 83% | 63 / 68 °C | 407 W | 0 |  |
| 2026-09-17 | E002-base-precision-d2/ladder/e2b-it/summary-d8 | 55 min | 83% | 63 / 68 °C | 403 W | 0 |  |
| 2026-09-17 | GPU tests: test_bake_gpu | 1 min | 29% | 49 / 56 °C | 340 W | - | 1 passed |
| 2026-09-17 | GPU tests: test_graph_decode_gpu | 20 min | 56% | 59 / 69 °C | 399 W | - | 23 passed |
| 2026-09-17 | GPU tests: test_it_gpu | 1 min | 28% | 51 / 57 °C | 279 W | - | 4 passed |
| 2026-09-17 | GPU tests: test_kquant_ladder_gpu | 15 min | 93% | 65 / 69 °C | 398 W | - | 4 passed |
| 2026-09-17 | GPU tests: test_kquant_ladder_gpu | 15 min | 94% | 65 / 70 °C | 400 W | - | 4 passed |
| 2026-09-17 | GPU tests: test_kquant_ladder_gpu | 15 min | 93% | 65 / 70 °C | 399 W | - | 4 passed |
| 2026-09-17 | GPU tests: test_padded_batch_gpu | 1 min | 35% | 52 / 61 °C | 362 W | - | 6 passed |
| 2026-09-17 | GPU tests: test_resident_gpu | 1 min | 23% | 50 / 56 °C | 280 W | - | 2 passed |
| 2026-09-17 | GPU tests: test_stand_gpu | 1 min | 29% | 51 / 57 °C | 277 W | - | 8 passed, 1 failed |
| 2026-09-17 | GPU tests: test_stand_gpu | 1 min | 23% | 56 / 61 °C | 271 W | - | 9 passed |

## Peaks

| | Value | When |
| --- | --- | --- |
| Temperature | 83 °C | 2026-09-14, calibration of the letter-choice corpora, share 0.8, before the thermal guard |
| Power | 413 W | 2026-09-15, 2026-09-15-e001-answer-d8-collapsed |
| Longest run | 1 h 40 min | 2026-09-16, E001-uniform-quantization/e2b-it/summary-d2 |
| GPU time in total | ~16 h 21 min | 83 entries |
<!-- station-log:end -->
