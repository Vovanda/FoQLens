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
| Power limit | 330 W of the default 450 W (the card allows 100-480 W) | `nvidia-smi -pl 330`, reapplied at every logon |
| Driver target temperature | 78 °C (the default is 83 °C; the card slows at 94 °C and shuts down at 97 °C) | `nvidia-smi -gtt 78`, reapplied at every logon |
| Fans | case fans and the two GPU fans on one curve of the GPU temperature: 40 °C 20%, 50 °C 40%, 65 °C 70%, 75 °C 90%, 80 °C 100%; the GPU fans do not go below 30%; the CPU fan stays with the board | Fan Control |

The ceiling of 80 °C is the owner's for this machine, below the card's own target.

## How the log is kept

Every run summary records the GPU through `GpuMonitor` - when the run started, how long it took, mean
utilization, mean and peak temperature, peak power - and the pacing through `"pacer"`, with the cooling
breaks taken. A session of the GPU tests leaves the same record in `runs/station/tests/`. A run stopped
before it wrote its summary is entered by hand in `runs/station/`, and its note says so.

`scripts/station_log.py` rebuilds the log and the peaks below from all of them, and from the summaries
of the runs deleted with the first corpus, which it reads from the git history. A time marked ~ is an
estimate: a summary written before the monitor kept a clock is counted by its samples, one a second.
The runs of E001 predate the monitor and are not in the log.

<!-- station-log:begin -->
## Log

| Date | Run | Time | Utilization, mean | Temperature, mean / peak | Power, peak | Cooling breaks | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-12 | E004-injection/e2b | ~8 min | 54% | - / - | - | - | deleted with the first corpus in 1e89a81 |
| 2026-09-12 | E005-backbone/e2b | ~14 min | 61% | - / - | - | - | deleted with the first corpus in 1e89a81 |
| 2026-09-12 | E007-dilation/e2b | ~7 min | 48% | - / - | - | - | deleted with the first corpus in 1e89a81 |
| 2026-09-12 | E008-zones-fixed-budget/e2b | ~20 min | 50% | - / - | - | - | deleted with the first corpus in 1e89a81 |
| 2026-09-12 | E009-zones-matrix/e2b | ~1 h 34 min | 40% | - / - | - | - | deleted with the first corpus in 1e89a81 |
| 2026-09-12 | E010-lens-layout-random/e2b | ~16 min | 72% | - / - | - | - | deleted with the first corpus in 1e89a81 |
| 2026-09-12 | E010-lens-layout/e2b | ~25 min | 69% | - / - | - | - | deleted with the first corpus in 1e89a81 |
| 2026-09-13 | E013-regulator-map/e2b | ~44 min | 71% | - / - | - | - | deleted with the first corpus in 1e89a81 |
| 2026-09-13 | E014-moved-zones/e2b | ~4 min | 66% | - / - | - | - | deleted with the first corpus in 1e89a81 |
| 2026-09-13 | reference/address-stability/e2b | ~6 min | 58% | - / - | - | - |  |
| 2026-09-13 | reference/shuffle-answers-canonical/e2b | ~6 min | 56% | - / - | - | - |  |
| 2026-09-13 | reference/shuffle-answers-fixed-address/e2b | ~4 min | 68% | - / - | - | - |  |
| 2026-09-13 | reference/shuffle-answers-question-only-matched/e2b | ~5 min | 60% | - / - | - | - |  |
| 2026-09-13 | reference/shuffle-answers-question-only/e2b | ~5 min | 56% | - / - | - | - |  |
| 2026-09-13 | reference/shuffle-answers/e2b | ~6 min | 60% | - / - | - | - |  |
| 2026-09-14 | GPU tests | 2 min | 14% | 47 / 54 °C | 265 W | - | 2 passed; the injection and filter smokes, entered by hand from their report |
| 2026-09-14 | GPU tests | 8 min | 16% | 47 / 57 °C | 277 W | - | 61 passed; the first session under the thermal guard, entered by hand from its report |
| 2026-09-14 | GPU tests | 1 min | 5% | 45 / 47 °C | 173 W | - | 1 passed |
| 2026-09-14 | calibration of the letter-choice corpora, share 0.8, before the thermal guard | ~25 min | 100% | - / 83 °C | 403 W | - | entered by hand from nvidia-smi, the run was stopped before its summary: a smell of burnt dust; the dust was blown out, the limits were set |

## Peaks

| | Value | When |
| --- | --- | --- |
| Temperature | 83 °C | 2026-09-14, calibration of the letter-choice corpora, share 0.8, before the thermal guard |
| Power | 403 W | 2026-09-14, calibration of the letter-choice corpora, share 0.8, before the thermal guard |
| Longest run | ~1 h 34 min | 2026-09-12, E009-zones-matrix/e2b |
| GPU time in total | ~4 h 59 min | 19 entries |
<!-- station-log:end -->
