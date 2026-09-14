"""Infrastructure: GPU utilization, memory, temperature and power of one phase of a run, sampled in the background.

The standard says a long run below ~80% utilization is a bug; this is how a run proves it is not.
nvidia-smi reports memory reserved by every process, torch the peak actually allocated by this
one - both are recorded. Temperature and power are recorded because a run that is fast but cooks the
card is not acceptable either (83 °C and 403 W on 2026-09-14; the ceiling is in gpu_share). When
the phase started and how long it took are recorded too: the station's log (foqlens/station.py) is
built from these summaries. The sampler and the clock are injectable, so the monitor is tested
without a GPU.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime

import numpy as np
import torch

Sample = tuple[float, float, float, float]  # (utilization %, memory used MiB, temperature °C, power W)
MIB = 2**20


def nvidia_smi(device: int = 0) -> Sample:
    out = subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    util, mem, temp, power = out.strip().split(",")
    return float(util), float(mem), float(temp), float(power)


def gpu_temperature(device: int = 0) -> float:
    """The core temperature of the card in °C - the sensor of gpu_share.ThermalGuard."""
    out = subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out.strip())


class GpuMonitor:
    def __init__(self, interval: float = 1.0, sampler: Callable[[], Sample] = nvidia_smi,
                 clock: Callable[[], float] = time.time):
        self.interval = interval
        self.sampler = sampler
        self.samples: list[Sample] = []
        self.torch_peak_mib: float | None = None
        self.started: float | None = None
        self.wall_s: float | None = None
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuMonitor:
        self.started = self._clock()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.wall_s = self._clock() - self.started
        if torch.cuda.is_available():
            self.torch_peak_mib = torch.cuda.max_memory_allocated() / MIB

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.sampler())
            self._stop.wait(self.interval)

    def summary(self) -> dict:
        out: dict = {"samples": len(self.samples)}
        if self.started is not None:
            out["started"] = datetime.fromtimestamp(self.started).isoformat(timespec="minutes")
        if self.wall_s is not None:
            out["wall_s"] = self.wall_s
        if self.samples:
            util, mem, temp, power = (np.array(column) for column in zip(*self.samples))
            out |= {
                "utilization_mean": float(util.mean()),
                "utilization_median": float(np.median(util)),
                "memory_reserved_peak_mib": float(mem.max()),
                "temperature_mean_c": float(temp.mean()),
                "temperature_peak_c": float(temp.max()),
                "power_mean_w": float(power.mean()),
                "power_peak_w": float(power.max()),
            }
        if self.torch_peak_mib is not None:
            out["memory_allocated_peak_mib"] = self.torch_peak_mib
        return out
