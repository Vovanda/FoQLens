"""Infrastructure: the progress line of a long run - step k of n, the percent done and the expected end.

The expected end is the mean time of a finished step times the steps left, counted from the start,
so a run can be followed from its log without counting by hand.

Invariant: after the last step the expected end is the time of that step.
Invariant: the percent is the share of steps done, rounded down - 100 only at the end.
Invariant: there is no step past the last one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime


class Progress:
    """Counts the steps of a run of known length and says how far it is and when it should end."""

    def __init__(self, total: int, what: str = "", clock: Callable[[], float] = time.time):
        if total < 0:
            raise ValueError(f"a run of {total} steps")
        self.total = total
        self.done = 0
        self._prefix = f"{what} " if what else ""
        self._clock = clock
        self._start = clock()

    def step(self, label: str) -> str:
        """One more step done: '<what> [k/n] <label> (pct%), ETA HH:MM'."""
        if self.done >= self.total:
            raise ValueError(f"step {self.done + 1} of a run of {self.total}")
        self.done += 1
        now = self._clock()
        end = now + (now - self._start) / self.done * (self.total - self.done)
        pct = 100 * self.done // self.total
        return f"{self._prefix}[{self.done}/{self.total}] {label} ({pct}%), ETA {datetime.fromtimestamp(end):%H:%M}"
