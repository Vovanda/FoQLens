"""When a decision of the regulator is taken again, and when the one in place is kept - inertia.

The condition is outside the regulator and knows nothing of what it decides: the regulator asks, at every point of a
pass, whether this module is decided again or keeps the layout it already carries. The two ends of the family are one
decision for the whole query and a decision at every point; a schedule of a finite step is everything between.

- BySchedule: a module is decided again every `layers` layers and every `visits` passes over it - the step is set from
  outside and does not depend on what the network is doing. BySchedule(1, 1) is the densest condition, a decision in
  front of every layer at every token, and it is what every sparser one is measured against.

A condition that depends on the state (a decision held until the state moves further than a tolerance) is a different
implementation of the same protocol; it does not fit inside one captured CUDA graph, because a graph replays the
branch it was captured on.

Invariant: BySchedule(1, 1) holds nothing - every module is decided at every visit.
Invariant: a module's first visit is always a decision: there is no layout in place to keep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Inertia(Protocol):
    """Whether the layout in place is kept at this point of a pass."""

    name: str

    def holds(self, layer: int, visit: int) -> bool:
        """`layer` is the layer the module belongs to, `visit` how many times it has been reached before, 0 the first."""
        ...


@dataclass(frozen=True)
class BySchedule:
    """Decide again every `layers` layers and every `visits` visits; hold the layout in place in between."""

    layers: int = 1
    visits: int = 1
    name: str = "schedule"

    def __post_init__(self) -> None:
        if self.layers < 1 or self.visits < 1:
            raise ValueError(f"a step of {self.layers} layers and {self.visits} visits: both are at least 1")

    def holds(self, layer: int, visit: int) -> bool:
        return visit > 0 and (layer % self.layers != 0 or visit % self.visits != 0)


EVERY_LAYER = BySchedule(1, 1)  # the densest condition; the default of the regulator
