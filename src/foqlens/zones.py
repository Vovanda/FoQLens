"""Expert zones: the peaks of a query's mask on the weight map, and the layout they give (docs/zones.md).

A mask is a field over the weight map. It is rasterized on a GRID x GRID grid and smoothed (sum and
count smoothed apart, so empty cells take their neighbours' value); every hill of the smoothed field
standing above PEAK_QUANTILE is one zone, with a center at its top and a base radius - the radius of
the disc with the area of the hill above half height.

The focus area f in [0, 1] sets every radius, R = r f / (1 - f): at 0 the zones shrink to their
centers (a hard edge), at 0.5 they are as found, at 1 they cover the whole map (no mask). A block's
log-sharpness is the strongest zone at its place, psi = max_i (-d_i / R_i). The layout spends a
given precision share p - a mean of bits between the D4 background and the D8 centers, 4 + 4p -
over the rings of psi.

Invariants:
- Invariant: a focus area outside [0, 1] (NaN included) is refused where it enters; so is a precision share (budget.py).
- Invariant: one zone per hill - a weaker top inside a stronger zone's half-height area is not a zone.
- Invariant: focus area 0 gives a hard edge (D8 plateaus, D4 around); focus area 1, or no zone at all,
  the precision share spent evenly, without a mask.
- Invariant: every layout at precision share p holds the weighted mean bits at 4 + 4p, up to one block's step.
- Invariant: random zones keep the number and the radii of the zones they control; only the centers move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from foqlens import budget as bg
from foqlens.quant import Level

GRID = 64
SMOOTH_CELLS = 1.5  # gaussian sigma of the smoothing, in grid cells
PEAK_QUANTILE = 0.95  # a zone's top stands above this share of the smoothed field
MAX_ZONES = 16
# Background ... centers. D2 is not used: uncalibrated 2 bits break the model (docs/results-residual.md).
LEVELS = (Level.D4, Level.D6, Level.D8)
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
