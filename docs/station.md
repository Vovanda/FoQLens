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

## Log

Every run summary records the GPU through `GpuMonitor`: mean utilization, mean and peak temperature, mean
and peak power. The log is filled from those summaries.

| Date | Run | Time | Utilization, mean | Temperature, mean / peak | Power, peak | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-14 | calibration of the letter-choice corpora, share 0.8, before the thermal guard | ~25 min | 100% | - / 83 °C | 403 W | stopped: a smell of burnt dust; the dust was blown out, the limits above were set |
| 2026-09-14 | GPU tests under the thermal guard | 8 min 17 s | 16% | 47 / 57 °C | 277 W | 61 passed |

## Peaks

| | Value | When |
| --- | --- | --- |
| Temperature | 83 °C | 2026-09-14, before the thermal guard |
| Power | 403 W | 2026-09-14, before the power limit |
