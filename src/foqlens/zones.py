"""The rules that turn a zone's lift into levels (docs/quantization-filter.md, rules 2-6), on any metric.

Where zones come from and how far they reach is graph_zones (the block graph of #4, the reach of #19); this module is
what the rules do with a lift: the ceiling of a zone from the focus strength g (rule 2), the profile of stops from
the ceiling down to the base precision (rule 3), levels from one lift (levels_from_lift, rule 6) or from the lifts of
zones of their own strength added in rungs (levels_from_rungs, #19). None of it looks at a coordinate or a picture.
Memory is what a layout costs, not what it was given.

Invariants:
- Invariant: a focus area outside [0, 1] (NaN included) is refused where it enters.
- Invariant: levels_from_lift gives the floor at lift 0 and the ceiling at lift 1, never below the floor, and a block
  one stop further out is at most one level lower.
- Invariant: levels_from_rungs with one ceiling for all is levels_from_lift on the even profile, rule 5 "sum" or "max",
  and no block rises past the highest ceiling of the zones that reach it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from foqlens.quant import LADDER, Level

PEAK_QUANTILE = 0.95  # a zone's top stands above this share of the smoothed field (graph_zones.find_graph_zones)
MAX_ZONES = 16  # the zones of one question at most, strongest first
# The rungs a zone can lift a block to by default: the read depths D2 ... D8, the ladder of
# docs/quantization-filter.md (rule 2: g = 1 is the top rung, D8). The kernel reads depths and ZERO only, and a
# model cut to D8 holds nothing above it; a run that reads its source weights passes a ladder with BF16 on top,
# and a run on a copy cut shorter passes regulator.kernel_ladder.
READ_LEVELS = tuple(lv for lv in LADDER if lv.depth)
# The ring past the zone's edge behind an empty floor (docs, the profile of a zone: "D2:1.5" at a ZERO base).
HALO_STOP = 1.5


def check_focus_area(focus_area: float) -> float:
    """The focus area as given, refused outside [0, 1] (NaN included)."""
    if not 0.0 <= focus_area <= 1.0:
        raise ValueError(f"focus area {focus_area} outside [0, 1]")
    return focus_area


def levels_from_lift(lift: np.ndarray, floor: Level, ceiling: Level, stops: Sequence[tuple[Level, float]]) -> np.ndarray:
    """Level codes [n_blocks] from the lift over the floor (docs/quantization-filter.md, rules 3 and 6).

    `stops` are the profile from the ceiling down, each the outer edge of that level's ring as a
    share of the reach. A block at lift l sits at rho* = reach (1 - l) and takes the level of the
    first stop that reaches it; past the last stop it is the floor.
    """
    check_stops(stops, floor, ceiling)
    reach = stops[-1][1]
    lift = np.clip(lift, 0.0, 1.0)
    rho = reach * (1.0 - lift)
    codes = np.full(len(lift), int(floor), dtype=np.uint8)
    inside = lift > 0.0  # a block no zone reaches stays at the floor, whatever the outermost stop is
    for level, stop in reversed(stops):
        codes[inside & (rho <= stop)] = int(level)
    return codes


def zone_ceilings(strengths: np.ndarray, focus_strength: float, floor: Level, ladder: Sequence[Level] = READ_LEVELS) -> list[Level]:
    """The ceiling of every zone from its own strength s in [0, 1]: kappa_i = gamma + floor(g s_i (m - gamma)) (#19).

    With every strength 1 this is rule 2 as it is: one ceiling for all zones.
    """
    strengths = np.asarray(strengths, dtype=float)
    if ((strengths < 0) | (strengths > 1) | np.isnan(strengths)).any():
        raise ValueError("a zone strength outside [0, 1]")
    return [ceiling_of(focus_strength * float(s), floor, ladder) for s in strengths]


def levels_from_rungs(lifts: np.ndarray, ceilings: Sequence[Level], floor: Level, combine: str = "sum",
                      ladder: Sequence[Level] = READ_LEVELS) -> np.ndarray:
    """Level codes [n_blocks] from every zone's lift [n_zones, n_blocks] and its own ceiling, on the even profile (#19).

    Rungs are counted on the run's ladder from the floor up: the floor is rung 0, kappa_i the rung of zone i's
    ceiling. The even profile is linear in rungs: a zone with lift l > 0 reads a block at min(kappa, 1 +
    floor(l kappa)) - what levels_from_lift gives on even_stops without a halo. Zones with different
    ceilings add in rungs, r = sum_i l_i kappa_i, the level being min(kappa*, 1 + floor(r)) where r > 0 and
    kappa* is the highest ceiling among the zones that reach the block - a zone never lifts a block past its
    own ceiling, however strong a neighbour is. With one ceiling for all this is rule 5 "sum" exactly. "max"
    takes the highest level any zone gives alone.
    """
    lifts = np.clip(np.asarray(lifts, dtype=float), 0.0, 1.0)
    rungs = [floor, *(lv for lv in ladder if lv > floor)]
    kappa = np.array([rungs.index(c) for c in ceilings], dtype=np.int64)
    if len(kappa) != len(lifts):
        raise ValueError(f"{len(kappa)} ceilings for {len(lifts)} zones")
    if len(kappa) == 0:
        return np.full(lifts.shape[1], int(floor), dtype=np.uint8)
    reaching = (lifts > 0) & (kappa[:, None] > 0)
    if combine == "sum":
        r = (lifts * kappa[:, None]).sum(axis=0)
        cap = np.where(reaching, kappa[:, None], 0).max(axis=0)
        rung = np.where(r > 0, np.minimum(cap, 1 + np.floor(r).astype(np.int64)), 0)
    elif combine == "max":
        alone = np.where(reaching, np.minimum(kappa[:, None], 1 + np.floor(lifts * kappa[:, None]).astype(np.int64)), 0)
        rung = alone.max(axis=0)
    else:
        raise ValueError(f"unknown combine {combine!r}, expected 'sum' or 'max'")
    return np.asarray([int(lv) for lv in rungs], dtype=np.uint8)[rung]


def check_stops(stops: Sequence[tuple[Level, float]], floor: Level, ceiling: Level) -> None:
    """Refuse a profile that is not a falloff from the ceiling down to the floor: stops above 0, strictly growing
    (docs: a stop above 1 is a ring past the zone's edge, as many as there are such stops)."""
    if not stops:
        raise ValueError("a profile needs at least one stop")
    if stops[0][0] is not ceiling:
        raise ValueError(f"the profile starts at {stops[0][0].name}, not at the ceiling {ceiling.name}")
    for (level, stop), (nxt, after) in zip(stops, stops[1:]):
        if nxt >= level:
            raise ValueError(f"the profile does not fall: {level.name} then {nxt.name}")
        if after <= stop:
            raise ValueError(f"the stops do not grow: {stop} then {after}")
    for level, stop in stops:
        if level < floor:
            raise ValueError(f"the profile reads {level.name} below the floor {floor.name}")
        if not stop > 0.0:
            raise ValueError(f"stop {stop} is not above 0")


def even_stops(floor: Level, ceiling: Level, ladder: Sequence[Level] = READ_LEVELS, halo: bool = False) -> tuple[tuple[Level, float], ...]:
    """The default profile: the rungs from the ceiling down to the floor, evenly spaced over the radius.

    Behind an empty floor the lowest rung goes past the edge instead, to HALO_STOP - the ring that
    softens the step from a zone into nothing (docs/quantization-filter.md).
    """
    rungs = [lv for lv in reversed(ladder) if floor < lv <= ceiling]
    if not rungs:
        return ()
    inner = rungs[:-1] if halo else rungs
    stops = [(lv, (i + 1) / len(inner)) for i, lv in enumerate(inner)] if inner else []
    return tuple(stops + ([(rungs[-1], HALO_STOP)] if halo else []))


def ceiling_of(focus_strength: float, floor: Level, ladder: Sequence[Level] = READ_LEVELS) -> Level:
    """The level at a zone's center: focus_strength of the way from the floor to the top (rule 2)."""
    if not 0.0 <= focus_strength <= 1.0:
        raise ValueError(f"focus strength {focus_strength} outside [0, 1]")
    above = [lv for lv in ladder if lv > floor]
    if not above:
        return floor
    # rule 2: the ceiling is floor(g k) rungs above the floor, so half the way up a D4 floor is D6
    rungs = min(int(focus_strength * len(above)), len(above))
    return above[rungs - 1] if rungs else floor
