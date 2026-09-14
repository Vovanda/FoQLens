"""Infrastructure: the share of the GPU a run may take, so that the card stays usable for people - and cool.

A run takes at most `share` of the VRAM (the allocator cap, see model.forbid_spill) and keeps its
average utilization near `share` by resting after every batch: a batch that kept the GPU busy for
t seconds is followed by t (1 - share) / share seconds of rest. A batch ends with a result on the
host, so its wall time is the time the GPU was busy with it.

The share comes from a run's --gpu-share flag, which defaults to FOQLENS_GPU_SHARE or DEFAULT_SHARE.

The share alone does not bound the heat: at 0.8 a run held the card at 83 °C and 403 W for half an
hour (2026-09-14). ThermalGuard wraps any pacer and reads the card's temperature between batches:
past SOFT_C the rest after a batch grows, and at CEILING_C the run waits until the card is back
down to RESUME_C.

Invariant: at share 1 a Throttle never rests - the full-speed bench.
Invariant: busy / (busy + rest) = share for every batch of a Throttle.
Invariant: a ThermalGuard never starts a batch while the last reading is at or above its ceiling,
and reads the sensor at most once per `poll` seconds outside a pause.
Invariant: a Cooldown pauses once per `every` seconds of running, before the batch that crosses it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

ENV = "FOQLENS_GPU_SHARE"
# Room left on the card for someone else (a game, another researcher's job).
DEFAULT_SHARE = 0.8

CEILING_C = 80.0  # Volodya, 2026-09-14: 80 °C is critical for this machine, whatever the card's own target
RESUME_C = 72.0   # hysteresis: cool by a margin, or the run would stall and restart at the edge every batch
SOFT_C = 75.0     # the extra rest starts this far below the ceiling, so the pause is the exception
POLL_S = 1.0      # a reading costs ~0.07 s of nvidia-smi; once a second is under a tenth of that time
COOLDOWN_EVERY_S = 3600.0  # Volodya, 2026-09-14: a cooling break once an hour
COOLDOWN_S = 300.0         # stopped at 83 °C the card read 68 after 3 s and 43 a few minutes later


def default_share() -> float:
    """The share a run takes unless told otherwise: FOQLENS_GPU_SHARE, else DEFAULT_SHARE."""
    return float(os.environ.get(ENV, DEFAULT_SHARE))


class Pacer(Protocol):
    """What a run wraps around each batch whose result has reached the host."""

    def batch(self) -> Iterator[None]: ...

    def stats(self) -> dict:
        """What the pacing did over the run, for its summary: this pacer's numbers and its inner one's."""
        ...


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

    def stats(self) -> dict:
        return {"share": self.share}


class ThermalGuard:
    """Another pacer, plus a temperature ceiling the run does not cross.

    Between SOFT_C and CEILING_C the batch is followed by extra rest, from none at SOFT_C to as long as
    the batch itself just below the ceiling. At CEILING_C and above the run pauses, reading the sensor
    every `poll` seconds, until it is at RESUME_C or below. `peak` is the hottest reading seen.
    """

    def __init__(self, inner: Pacer, sensor: Callable[[], float], ceiling: float = CEILING_C,
                 resume: float = RESUME_C, soft: float = SOFT_C, poll: float = POLL_S,
                 clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] | None = None):
        if not resume < ceiling or not soft < ceiling:
            raise ValueError(f"resume {resume} and soft {soft} must be below the ceiling {ceiling}")
        self.inner, self.sensor = inner, sensor
        self.ceiling, self.resume, self.soft, self.poll = ceiling, resume, soft, poll
        self._clock, self._sleep, self._log = clock, sleep, log
        self._read_at: float | None = None
        self._last = float("-inf")
        self.peak = float("-inf")
        self.paused_s = 0.0

    def _read(self) -> float:
        self._last = self.sensor()
        self._read_at = self._clock()
        self.peak = max(self.peak, self._last)
        return self._last

    def temperature(self) -> float:
        """The last reading, refreshed if it is older than `poll`."""
        if self._read_at is None or self._clock() - self._read_at >= self.poll:
            return self._read()
        return self._last

    @contextmanager
    def batch(self) -> Iterator[None]:
        self._cool_down()
        with self.inner.batch():
            start = self._clock()
            yield
            busy = self._clock() - start  # the batch alone, before the inner pacer's own rest
        heat = (self.temperature() - self.soft) / (self.ceiling - self.soft)
        if heat > 0:
            self._sleep(busy * min(heat, 1.0))

    def _cool_down(self) -> None:
        if self.temperature() < self.ceiling:
            return
        if self._log is not None:
            self._log(f"  GPU at {self._last:.0f} C: pausing until {self.resume:.0f} C")
        start = self._clock()
        while True:
            self._sleep(self.poll)
            if self._read() <= self.resume:
                break
        self.paused_s += self._clock() - start

    def stats(self) -> dict:
        peak = None if self.peak == float("-inf") else self.peak
        return self.inner.stats() | {"temperature_ceiling_c": self.ceiling, "temperature_peak_c": peak,
                                     "thermal_pause_s": self.paused_s}


class Cooldown:
    """Another pacer, plus a break of `pause` seconds after every `every` seconds of running.

    A long run does not only need to stay under a ceiling; it gets a regular rest whatever the
    temperature. The clock of a stretch starts at its first batch and restarts after each break.
    """

    def __init__(self, inner: Pacer, every: float = COOLDOWN_EVERY_S, pause: float = COOLDOWN_S,
                 clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] | None = None):
        if every <= 0 or pause < 0:
            raise ValueError(f"cooldown every {every} s for {pause} s")
        self.inner, self.every, self.pause = inner, every, pause
        self._clock, self._sleep, self._log = clock, sleep, log
        self._since: float | None = None
        self.breaks = 0
        self.paused_s = 0.0

    @contextmanager
    def batch(self) -> Iterator[None]:
        now = self._clock()
        if self._since is None:
            self._since = now
        elif now - self._since >= self.every:
            if self._log is not None:
                self._log(f"  cooling break: {self.pause / 60:.0f} min after {(now - self._since) / 60:.0f} min of running")
            self._sleep(self.pause)
            self.breaks += 1
            self.paused_s += self.pause
            self._since = self._clock()
        with self.inner.batch():
            yield

    def stats(self) -> dict:
        return self.inner.stats() | {"cooling_breaks": self.breaks, "cooling_break_s": self.paused_s}


FULL = Throttle(1.0)
