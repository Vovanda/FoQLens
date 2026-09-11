"""Infrastructure: GPU utilization and memory of one phase of a run, sampled in the background.

The standard says a long run below ~80% utilization is a bug; this is how a run proves it is not.
nvidia-smi reports memory reserved by every process, torch the peak actually allocated by this
one - both are recorded. The sampler is injectable, so the monitor is tested without a GPU.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable

import numpy as np
import torch

Sample = tuple[float, float]  # (utilization %, memory used MiB)
MIB = 2**20


def nvidia_smi(device: int = 0) -> Sample:
    out = subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    util, mem = out.strip().split(",")
    return float(util), float(mem)


class GpuMonitor:
    def __init__(self, interval: float = 1.0, sampler: Callable[[], Sample] = nvidia_smi):
        self.interval = interval
        self.sampler = sampler
        self.samples: list[Sample] = []
        self.torch_peak_mib: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuMonitor:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if torch.cuda.is_available():
            self.torch_peak_mib = torch.cuda.max_memory_allocated() / MIB

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.sampler())
            self._stop.wait(self.interval)

    def summary(self) -> dict:
        out: dict = {"samples": len(self.samples)}
        if self.samples:
            util = np.array([s[0] for s in self.samples])
            out |= {
                "utilization_mean": float(util.mean()),
                "utilization_median": float(np.median(util)),
                "memory_reserved_peak_mib": float(max(s[1] for s in self.samples)),
            }
        if self.torch_peak_mib is not None:
            out["memory_allocated_peak_mib"] = self.torch_peak_mib
        return out
