"""Expert zones: the peaks of a query's mask on the weight map, and the layout they give (docs/quantization-filter.md).

A mask is a field over the weight map. It is rasterized on a GRID x GRID grid and smoothed (sum and
count smoothed apart, so empty cells take their neighbours' value); every hill of the smoothed field
standing above PEAK_QUANTILE is one zone, with a center at its top and a base radius - the radius of
the disc with the area of the hill above half height.

The focus area f in [0, 1] sets every radius, R = r f / (1 - f): at 0 the zones shrink to their
centers (a hard edge), at 0.5 they are as found, at 1 they cover the whole map (no mask).

The layout (docs/quantization-filter.md) has no budget: precision_lift gives every
block how far it is lifted over the floor, 1 at a zone's center and 0 at its reach, and
levels_from_lift turns that into levels between a floor and a ceiling along the profile's stops.
Memory is what the layout costs, not what it was given.

Invariants:
- Invariant: a focus area outside [0, 1] (NaN included) is refused where it enters.
- Invariant: one zone per hill - a weaker top inside a stronger zone's half-height area is not a zone.
- Invariant: random zones keep the number and the radii of the zones they control; only the centers move.
- Invariant: precision_lift is 0 outside every zone's reach and 1 at a center; combining lifts never
  lowers a block below the strongest single lift.
- Invariant: levels_from_lift gives the floor at lift 0 and the ceiling at lift 1, never below the
  floor, and a block one stop further out is at most one level lower.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from foqlens.quant import LADDER, Level

GRID = 64
SMOOTH_CELLS = 1.5  # gaussian sigma of the smoothing, in grid cells
PEAK_QUANTILE = 0.95  # a zone's top stands above this share of the smoothed field
MAX_ZONES = 16
# The rungs a zone can lift a block to: the ladder above ZERO, up to bf16 (docs/quantization-filter.md,
# rule 2: g = 1 is the top rung). The codes follow the ladder, so levels compare by precision. What a
# model can actually carry is a choice of the run, not of the layout - a floor of D4 leaves D2 out of the
# rings, and a run that drops bf16 passes a ladder without it.
READ_LEVELS = LADDER[1:]
# A stop past 1 puts a ring outside the zone's radius. On a 2D map a ring out to 1.5 covers 1.25 of
# the zone's area; further than that it is a second zone, and the size belongs to the focus area.
MAX_STOP = 1.5
HALO_STOP = 1.5
PLACES_TRIED = 40  # landings a moved figure tries before taking the one that costs what the original did
EPS = 1e-12


@dataclass(frozen=True)
class Zones:
    centers: np.ndarray  # [n, dims] on the weight map
    radii: np.ndarray  # [n] base radii, weight map units


def check_focus_area(focus_area: float) -> float:
    """The focus area as given, refused outside [0, 1] (NaN included)."""
    if not 0.0 <= focus_area <= 1.0:
        raise ValueError(f"focus area {focus_area} outside [0, 1]")
    return focus_area


def _smoothed(coords: np.ndarray, field: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The field on the grid, smoothed as sum / count so that empty cells take their neighbours' value."""
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    cell = np.maximum(hi - lo, np.finfo(float).eps) / (grid - 1)
    idx = np.clip(np.round((coords - lo) / cell).astype(int), 0, grid - 1)
    total = np.zeros((grid, grid))
    count = np.zeros((grid, grid))
    np.add.at(total, (idx[:, 0], idx[:, 1]), field)
    np.add.at(count, (idx[:, 0], idx[:, 1]), 1.0)
    smooth = ndimage.gaussian_filter(total, SMOOTH_CELLS) / np.maximum(ndimage.gaussian_filter(count, SMOOTH_CELLS), EPS)
    return smooth, lo, cell


def find_zones(field: np.ndarray, coords: np.ndarray, grid: int = GRID) -> Zones:
    """The zones of a mask `field` [n_blocks] over the weight map `coords` [n_blocks, 2], strongest first."""
    smooth, lo, cell = _smoothed(coords, field, grid)
    tops = np.argwhere((smooth == ndimage.maximum_filter(smooth, size=3)) & (smooth > np.quantile(smooth, PEAK_QUANTILE)))
    tops = tops[np.argsort(-smooth[tuple(tops.T)], kind="stable")]
    base = np.median(smooth)
    claimed = np.zeros_like(smooth, dtype=bool)
    centers, radii = [], []
    for top in tops:
        if claimed[tuple(top)] or len(centers) == MAX_ZONES:
            continue
        regions, _ = ndimage.label(smooth >= base + (smooth[tuple(top)] - base) / 2)
        hill = regions == regions[tuple(top)]
        claimed |= hill
        centers.append(lo + top * cell)
        radii.append(np.sqrt(hill.sum() * float(np.prod(cell)) / np.pi))
    dims = coords.shape[1]
    return Zones(centers=np.asarray(centers, dtype=float).reshape(-1, dims), radii=np.asarray(radii, dtype=float))


def random_zones(zones: Zones, coords: np.ndarray, rng: np.random.Generator) -> Zones:
    """The same number of zones with the same radii, centered on random blocks of the weight map."""
    at = rng.choice(len(coords), size=len(zones.radii), replace=False)
    return Zones(centers=coords[at].copy(), radii=zones.radii.copy())


def moved_zones(zones: Zones, coords: np.ndarray, rng: np.random.Generator,
                weights: np.ndarray | None = None, reach: float = 1.0) -> Zones:
    """The same zones somewhere else on the map: turned around their own center and put down elsewhere.

    The honest control for a layout whose memory is a result. Random centers of the same radii
    (random_zones) spend more, because a query's zones overlap each other and scattered ones do not -
    a control that costs more and answers worse says nothing about the address. Moving the zones as
    one rigid figure keeps their count, their radii and every distance between them, and changes only
    where they sit, so what is compared is the place and nothing else.

    With `weights` the landing is chosen so that the figure covers about as much weight as it did
    where it came from - blocks are not spread evenly over the map, and a query's zones sit where they
    are sparse, so a plain landing costs a fifth more.

    Invariant: the moved figure keeps the pairwise distances of the original, up to floating point.
    """
    if len(zones.radii) == 0:
        return zones
    shifted = zones.centers - zones.centers.mean(axis=0)
    if coords.shape[1] == 2:  # a rotation exists on the plane the weight map is drawn on
        angle = rng.uniform(0, 2 * np.pi)
        turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shifted = shifted @ turn.T
    if weights is None:
        return Zones(centers=shifted + coords[rng.integers(len(coords))], radii=zones.radii.copy())
    # Blocks do not lie evenly on the map, so the same figure covers more of them in some places than
    # in others - and a query's zones sit where they are sparse. Of PLACES_TRIED landings the one whose
    # covered weight is closest to the original is taken, so the control spends what the layout spends.
    want = _covered(zones, coords, weights, reach)
    best, distance = None, np.inf
    for _ in range(PLACES_TRIED):
        moved = Zones(centers=shifted + coords[rng.integers(len(coords))], radii=zones.radii.copy())
        gap = abs(_covered(moved, coords, weights, reach) - want)
        if gap < distance:
            best, distance = moved, gap
    return best


def _covered(zones: Zones, coords: np.ndarray, weights: np.ndarray, reach: float) -> float:
    """The weight of the blocks the zones reach, as a share of all weight."""
    if len(zones.radii) == 0:
        return 0.0
    inside = (np.linalg.norm(coords[:, None, :] - zones.centers[None], axis=-1) <= zones.radii * reach).any(axis=1)
    return float(weights[inside].sum() / weights.sum())


def precision_lift(
    coords: np.ndarray, zones: Zones, focus_area: float, reach: float = 1.0, combine: str = "sum",
) -> np.ndarray:
    """How far every block is lifted over the floor, 0 ... 1 (docs/quantization-filter.md, rules 1, 4 and 5).

    One zone lifts a block by 1 - d / (R reach), clipped at 0, so the lift is 1 at its center and
    fades to 0 at `reach` radii - the outermost stop of the profile. Zones that cover the same block
    combine by `combine`: "sum" adds their lifts, capped at 1; "max"
    takes the strongest alone.
    """
    check_focus_area(focus_area)
    if len(zones.radii) == 0 or focus_area == 0.0 or reach <= 0.0:
        return np.zeros(len(coords))
    if focus_area == 1.0:
        return np.ones(len(coords))
    radii = zones.radii * (focus_area / (1.0 - focus_area)) * reach
    d = np.linalg.norm(coords[:, None, :] - zones.centers[None], axis=-1)
    lifts = np.clip(1.0 - d / radii, 0.0, None)
    if combine == "sum":
        return np.minimum(lifts.sum(axis=1), 1.0)
    if combine == "max":
        return lifts.max(axis=1)
    raise ValueError(f"unknown combine {combine!r}, expected 'sum' or 'max'")


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


def check_stops(stops: Sequence[tuple[Level, float]], floor: Level, ceiling: Level) -> None:
    """Refuse a profile that is not a falloff from the ceiling down to the floor."""
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
        if not 0.0 < stop <= MAX_STOP:
            raise ValueError(f"stop {stop} outside (0, {MAX_STOP}]")


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
