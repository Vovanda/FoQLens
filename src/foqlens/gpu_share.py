"""Infrastructure: the share of the GPU a run may take, so that the card stays usable for people.

A run takes at most `share` of the VRAM (the allocator cap, see model.forbid_spill) and keeps its
average utilization near `share` by resting after every batch: a batch that kept the GPU busy for
t seconds is followed by t (1 - share) / share seconds of rest. A batch ends with a result on the
host, so its wall time is the time the GPU was busy with it.

The share comes from a run's --gpu-share flag, which defaults to FOQLENS_GPU_SHARE or DEFAULT_SHARE.

Invariant: at share 1 a run never rests - the full-speed bench.
Invariant: busy / (busy + rest) = share for every batch.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

ENV = "FOQLENS_GPU_SHARE"
# Room left on the card for someone else (a game, another researcher's job).
DEFAULT_SHARE = 0.8


def default_share() -> float:
    """The share a run takes unless told otherwise: FOQLENS_GPU_SHARE, else DEFAULT_SHARE."""
    return float(os.environ.get(ENV, DEFAULT_SHARE))


class Throttle:
    """Rests after every batch so that the GPU is busy `share` of the time."""

    def __init__(self, share: float = 1.0, clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep):
        if not 0.0 < share <= 1.0:
            raise ValueError(f"GPU share {share} outside (0, 1]")
        self.share = share
        self._clock = clock
        self._sleep = sleep

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Around one batch whose result has reached the host: rest in proportion to its busy time."""
        start = self._clock()
        yield
        rest = (self._clock() - start) * (1.0 - self.share) / self.share
        if rest > 0:
            self._sleep(rest)


FULL = Throttle(1.0)
