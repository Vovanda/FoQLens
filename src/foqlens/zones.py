"""Expert zones: the peaks of a query's mask on the weight map, and the layout they give (docs/zones.md).

A mask is a field over the weight map. It is rasterized on a GRID x GRID grid and smoothed (sum and
count smoothed apart, so empty cells take their neighbours' value); every hill of the smoothed field
standing above PEAK_QUANTILE is one zone, with a center at its top and a base radius - the radius of
the disc with the area of the hill above half height.

The focus area f in [0, 1] sets every radius, R = r f / (1 - f): at 0 the zones shrink to their
centers (a hard edge), at 0.5 they are as found, at 1 they cover the whole map (no mask).

Two layouts read those zones. The legacy one (E008, E009) spends a given precision share p over the
rings of the log-sharpness psi = max_i (-d_i / R_i): a mean of bits between the D4 background and
the D8 centers, 4 + 4p. The graded one (E010, docs/lens.md) has no budget: precision_lift gives every
block how far it is lifted over the floor, 1 at a zone's center and 0 at its reach, and
levels_from_lift turns that into levels between a floor and a ceiling along the profile's stops.
Memory is what the layout costs, not what it was given.

Invariants:
- Invariant: a focus area outside [0, 1] (NaN included) is refused where it enters; so is a precision share (budget.py).
- Invariant: one zone per hill - a weaker top inside a stronger zone's half-height area is not a zone.
- Invariant: focus area 0 gives a hard edge (D8 plateaus, D4 around); focus area 1, or no zone at all,
  the precision share spent evenly, without a mask.
- Invariant: every layout at precision share p holds the weighted mean bits at 4 + 4p, up to one block's step.
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

from foqlens import budget as bg
from foqlens.quant import Level

GRID = 64
SMOOTH_CELLS = 1.5  # gaussian sigma of the smoothing, in grid cells
PEAK_QUANTILE = 0.95  # a zone's top stands above this share of the smoothed field
MAX_ZONES = 16
# Background ... centers of the legacy layout. D2 is not used there: uncalibrated 2 bits break the
# model (experiments/E006-read-depths/results.md).
LEVELS = (Level.D4, Level.D6, Level.D8)
# The ladder of the graded layout: every level a block can be read at, coarse first. What a model can
# actually carry is a choice of the run, not of the layout - a floor of D4 leaves D2 out of the rings.
READ_LEVELS = (Level.D2, Level.D4, Level.D6, Level.D8)
# A stop past 1 puts a ring outside the zone's radius. On a 2D map a ring out to 1.5 covers 1.25 of
# the zone's area; further than that it is a second zone, and the size belongs to the focus area.
MAX_STOP = 1.5
HALO_STOP = 1.5
# A block climbs one level per ring of psi it is inside; the rings are RING_GAP apart in psi.
RING_GAP = np.log(2.0)
SEARCH_STEPS = 60
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


def budget_bits(precision_share: float) -> float:
    """The mean of bits a precision share spends: from all background (0) to all centers (1)."""
    low, high = LEVELS[0].bits, LEVELS[-1].bits
    return low + bg.check_precision_share(precision_share) * (high - low)


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


def log_sharpness(coords: np.ndarray, zones: Zones, focus_area: float) -> np.ndarray:
    """psi [n_blocks] = max_i (-d_i / R_i), R_i = r_i f / (1 - f).

    At focus area 1 the radius is infinite and psi = 0 everywhere; without zones as well - there is no
    mask. At focus area 0 the radius is 0 and only the order survives, psi = max_i (-d_i / r_i), which the
    layout reads as a hard edge.
    """
    check_focus_area(focus_area)
    if focus_area == 1.0 or len(zones.radii) == 0:
        return np.zeros(len(coords))
    d = np.linalg.norm(coords[:, None, :] - zones.centers[None], axis=-1)
    scale = 1.0 if focus_area == 0.0 else focus_area / (1.0 - focus_area)
    return (-d / (zones.radii * scale)).max(axis=1)


def precision_lift(
    coords: np.ndarray, zones: Zones, focus_area: float, reach: float = 1.0, combine: str = "sum",
) -> np.ndarray:
    """How far every block is lifted over the floor, 0 ... 1 (docs/lens.md, rules 1, 4 and 5).

    One zone lifts a block by 1 - d / (R reach), clipped at 0, so the lift is 1 at its center and
    fades to 0 at `reach` radii - the outermost stop of the profile. Zones that cover the same block
    combine by `combine`: "sum" adds their lifts, capped at 1, as thin lenses in contact do; "max"
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
    """Level codes [n_blocks] from the lift over the floor (docs/lens.md, rules 3 and 6).

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
    softens the step from a lens into nothing (docs/lens.md).
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


def _levels_of(psi: np.ndarray, edge: float, rings: int) -> np.ndarray:
    """Level index per block: 0 (background) ... rings, one per ring of psi above `edge`."""
    return np.clip(np.floor((psi - edge) / RING_GAP) + 1, 0, rings).astype(int)


def layout_at_budget(
    psi: np.ndarray, weights: np.ndarray, bits: float, tiebreak: np.ndarray, hard_edge: bool = False,
) -> np.ndarray:
    """Level codes [n_blocks] from log-sharpness at a weighted mean of `bits`.

    The outer edge of the rings is searched so that the mean stays at or below the budget; the rest
    of the budget goes one step at a time to the next blocks by psi, ties broken by `tiebreak`
    (a permutation rank). A hard edge has one ring: D8 inside, D4 outside.
    """
    level_bits = np.array([lv.bits for lv in LEVELS], dtype=float)
    top = len(LEVELS) - 1
    rings = 1 if hard_edge else top
    step = (level_bits[-1] - level_bits[0]) / rings

    def mean_bits(level_idx: np.ndarray) -> float:
        return float((level_bits[0] + step * level_idx) @ weights / weights.sum())

    # Invariant of the search: at `lo` the mean is above the budget (every block on top), at `hi` not (all background).
    lo, hi = psi.min() - (rings + 1) * RING_GAP - 1.0, psi.max() + 1.0
    for _ in range(SEARCH_STEPS):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_bits(_levels_of(psi, mid, rings)) > bits else (lo, mid)
    idx = _levels_of(psi, hi, rings)
    remaining = (bits - mean_bits(idx)) * weights.sum()
    if remaining > 0:
        order = np.lexsort((tiebreak, -psi))
        candidates = order[idx[order] < rings]
        cum = np.cumsum(step * weights[candidates])
        take = min(int(np.searchsorted(cum, remaining)) + 1, len(candidates))  # at most one block past the budget
        idx[candidates[:take]] += 1
    codes = np.array([int(lv) for lv in LEVELS], dtype=np.uint8)
    return codes[idx * (top if hard_edge else 1)]
