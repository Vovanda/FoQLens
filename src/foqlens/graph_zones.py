"""Expert zones on the block graph, and how far they reach (issues #4, #19).

The surface the zones live on is a graph (owner, 2026-09-16): every block linked to its nearest blocks
in a metric (metric.neighbour_table). zones.find_zones finds zones on a raster of the 2D weight map; here
the same procedure runs on the graph, so the space is an input and no picture is drawn:

- a peak is a block whose smoothed mask is at least that of every neighbour and above PEAK_QUANTILE;
- its hill is the connected region of the graph standing above half its height, and a weaker top inside
  a stronger hill is not a zone;
- the base radius is a share of weight mass, not the area of a blob: the smallest radius at which the
  blocks around the peak hold as much weight as the hill (#4).

A zone is a center and a radius, and its figure is every block within the radius by the distances of
the layout's metric - the geodesic along the graph (metric.geodesic), or its resistive form - so the
figure is arbitrary and follows the graph, not a ball cut across the space. A zone lifts the blocks it
reaches by rule 4 of docs/quantization-filter.md, from 1 at its center to 0 at its reach times the last
stop. How far it reaches is replaceable (Reach): FoundReach stretches the found radius by f / (1 - f)
(rule 1 as it is), FrontReach sends every zone a share f of the width of the network (owner, 2026-09-16).

Invariants:
- Invariant: one zone per hill - a weaker top inside a stronger zone's hill is not a zone.
- Invariant: the blocks within a zone's radius hold at least its hill's weight, and no smaller radius does.
- Invariant: a lift is 1 at a zone's center and 0 at and beyond its reach times the last stop; a reach of 0
  lifts nothing.
- Invariant: either reach at f = 1 lifts every block - the whole network, as precision_lift.
- Invariant: lifts do not decrease as f grows, for either reach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from foqlens.metric import PAD, BlockMetric
from foqlens.zones import MAX_ZONES, PEAK_QUANTILE, check_focus_area


@dataclass(frozen=True)
class GraphZones:
    centers: np.ndarray  # [n] block indices
    radii: np.ndarray  # [n] base radii, in the metric's units


def _neighbour_mean(field: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Every block's value averaged with its neighbours': one step of smoothing on the graph."""
    real = table != PAD
    summed = field + np.where(real, field[np.where(real, table, 0)], 0.0).sum(axis=1)
    return summed / (1 + real.sum(axis=1))


def block_graph(table: np.ndarray) -> csr_matrix:
    heads = np.repeat(np.arange(len(table)), table.shape[1])
    tails = table.ravel()
    keep = tails != PAD
    return csr_matrix((np.ones(keep.sum()), (heads[keep], tails[keep])), shape=(len(table), len(table)))


def _hill(graph: csr_matrix, above: np.ndarray, top: int) -> np.ndarray:
    """The blocks above the threshold connected to `top` through blocks above it."""
    inside = np.flatnonzero(above)
    _, labels = connected_components(graph[inside][:, inside], directed=False)
    hill = np.zeros(len(above), dtype=bool)
    hill[inside[labels == labels[np.searchsorted(inside, top)]]] = True
    return hill


def mass_radius(distances: np.ndarray, weights: np.ndarray, mass: float) -> float:
    """The smallest radius at which the blocks within it of one center (`distances`) hold `mass` of weight."""
    order = np.argsort(distances, kind="stable")
    held = np.cumsum(weights[order])
    return float(distances[order[min(np.searchsorted(held, mass), len(order) - 1)]])


def find_graph_zones(field: np.ndarray, metric: BlockMetric, table: np.ndarray, weights: np.ndarray,
                    max_zones: int = MAX_ZONES, peak_quantile: float = PEAK_QUANTILE) -> GraphZones:
    """The zones of a mask `field` [n_blocks] in `metric`, found on its neighbour `table`, strongest first."""
    table = np.asarray(table)
    smooth = _neighbour_mean(np.asarray(field, dtype=np.float64), table)
    around = np.where(table != PAD, smooth[np.where(table != PAD, table, 0)], -np.inf).max(axis=1)
    tops = np.flatnonzero((smooth >= around) & (smooth > np.quantile(smooth, peak_quantile)))
    tops = tops[np.argsort(-smooth[tops], kind="stable")]
    base = np.median(smooth)
    graph = block_graph(table)
    claimed = np.zeros(len(smooth), dtype=bool)
    centers, radii = [], []
    for top in tops:
        if claimed[top] or len(centers) == max_zones:
            continue
        hill = _hill(graph, smooth >= base + (smooth[top] - base) / 2, int(top))
        claimed |= hill
        d = metric.distances(torch.tensor([int(top)]))[0].cpu().numpy()
        centers.append(int(top))
        radii.append(mass_radius(d, weights, float(weights[hill].sum())))
    return GraphZones(centers=np.asarray(centers, dtype=np.int64), radii=np.asarray(radii, dtype=float))


class Reach(Protocol):
    def radii(self, zones: GraphZones) -> np.ndarray:
        """How far every zone reaches, before the profile's last stop: [n]; inf is everywhere, 0 is nowhere."""
        ...


@dataclass(frozen=True)
class FoundReach:
    """Rule 1 as it is: the found radius stretched by f / (1 - f) - 0 at f = 0, as found at 0.5, everything at 1."""

    focus_area: float

    def __post_init__(self) -> None:
        check_focus_area(self.focus_area)

    def radii(self, zones: GraphZones) -> np.ndarray:
        if self.focus_area == 1.0:
            return np.full(len(zones.radii), np.inf)
        return zones.radii * (self.focus_area / (1.0 - self.focus_area))


@dataclass(frozen=True)
class FrontReach:
    """The owner's front: every zone reaches f times the width of the network - 0 at f = 0, the whole width at 1."""

    focus_area: float
    width: float  # the width of the network along the metric (metric.sweep_width), fixed per metric so f means one thing

    def __post_init__(self) -> None:
        check_focus_area(self.focus_area)

    def radii(self, zones: GraphZones) -> np.ndarray:
        # f = 1 is the whole network by definition: the width is measured by sweeps and may fall short of the diameter
        reach = np.inf if self.focus_area == 1.0 else self.focus_area * self.width
        return np.full(len(zones.radii), reach)


def zone_lifts(zones: GraphZones, radii: np.ndarray, metric: BlockMetric, last_stop: float = 1.0) -> np.ndarray:
    """Every zone's lift of every block by rule 4: max(0, 1 - d / (R s_last)), [n_zones, n_blocks]."""
    if len(zones.centers) == 0:
        return np.zeros((0, metric.n_blocks))
    d = metric.distances(torch.as_tensor(zones.centers)).cpu().numpy()
    reach = (np.asarray(radii, dtype=float) * last_stop)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lifts = np.clip(1.0 - d / reach, 0.0, None)
    lifts[np.broadcast_to(reach == 0, lifts.shape)] = 0.0  # a zone that reaches nowhere lifts nothing, its center too
    lifts[np.broadcast_to(np.isinf(reach), lifts.shape)] = 1.0
    return lifts
