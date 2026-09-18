"""Zone strategies by name: a run is a list of words and knobs, and every word picks one class behind its interface.

The path of #19 - a score over the blocks, the surface it lies on, the zones found on it, how far they reach, how
strong each is, the levels they give - is assembled here from names, so a run script says `graph="mutual-nicdm",
medium="jump", zones="topic", strength="equal"` and never builds the parts itself. A new strategy is a new entry in
its table, not a branch in the old ones (CLAUDE.md, SOLID). Where the score comes from - the mask source - is the
bench's (pipeline.Bench.source); this module starts from the scores.

The space is built once from calibration scores (Space.build): the co-activation metric M1, its neighbour graph and
the graph's width, which fixes what f means for every question. A question's surface is that graph at rest, or its
medium of the question's own activity (M4, M4b).

Invariant: every name a table does not hold is refused with the names it does.
Invariant: a layout built here is the same as the one built from its parts by hand - the registry only picks.
Invariant: f = 0 lifts the zone centers alone and f = 1, g = 1 lifts every block of a question with a zone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from foqlens import graph_zones as gz
from foqlens.layouts import (
    EqualStrength,
    GraphZoneLayout,
    GraphZoneSource,
    QuantileLevels,
    QueryGraphZones,
    TopicGraphZones,
    TopicMeans,
    ZoneStrength,
)
from foqlens.metric import (
    CoactivationMetric,
    HarmonicConductance,
    JumpConductance,
    MediumSurface,
    StillSurface,
    Surface,
    geodesic,
    mutual_nicdm_table,
    neighbour_table,
    sweep_width,
)
from foqlens.quant import Level
from foqlens.zones import READ_LEVELS

GRAPHS = {"union": neighbour_table, "mutual-nicdm": mutual_nicdm_table}  # how the block graph is built (#4)
MEDIA = {"harmonic": HarmonicConductance, "jump": JumpConductance}  # M4, M4b (#4); "still" is the graph at rest
STILL = "still"
ZONES = ("query", "topic")  # the question's own mask, or its topic's mean without it
COMBINE = ("sum", "max")  # rule 5
NEIGHBOURS = 16  # blocks every block is linked to: the k of #4's measurements on E001's masks


def _pick(table: dict | tuple, name: str, what: str):
    if name not in table:
        raise ValueError(f"unknown {what} {name!r}, expected one of {sorted(table)}")
    return table[name] if isinstance(table, dict) else name


@dataclass(frozen=True)
class Space:
    """The block graph of a calibration set: its neighbour table, edge lengths and width along it."""

    table: torch.Tensor  # [n_blocks, width] neighbours, PAD-padded
    lengths: torch.Tensor  # [n_blocks, width] edge lengths in M1
    width: float  # the width of the network along the graph: f = 1 reaches it

    @classmethod
    def build(cls, calibration: np.ndarray, graph: str = "mutual-nicdm", k: int = NEIGHBOURS,
              device: str | torch.device = "cpu") -> Space:
        """From raw scores [questions, n_blocks] of calibration questions, never of the questions laid out."""
        table, lengths = _pick(GRAPHS, graph, "graph")(CoactivationMetric(calibration, device), k)
        return cls(table, lengths, sweep_width(geodesic(table, lengths)))

    def surface(self, medium: str = STILL, activity: np.ndarray | None = None,
                background: np.ndarray | None = None) -> Surface:
        """The graph at rest, or through the medium of every question's `activity` against its `background`."""
        if medium == STILL:
            return StillSurface(self.table.cpu().numpy(), geodesic(self.table, self.lengths))
        conductance = _pick(MEDIA, medium, "medium")
        if activity is None or background is None:
            raise ValueError(f"the medium {medium!r} needs every question's activity and its background")
        return MediumSurface(self.table, self.lengths, activity, background, conductance)


@dataclass(frozen=True)
class Knobs:
    """The controls of docs/quantization-filter.md: base precision, focus_area f, focus_strength g, combining."""

    floor: Level
    focus_area: float
    focus_strength: float
    combine: str = "sum"

    def __post_init__(self) -> None:
        _pick(COMBINE, self.combine, "combine")


def zone_source(zones: str, scores: np.ndarray, surface: Surface, weights: np.ndarray,
                domains: tuple[str, ...] | None = None) -> GraphZoneSource:
    """Where a question's zones come from: its own scores, or its topic's mean scores without it."""
    if _pick(ZONES, zones, "zones") == "query":
        return QueryGraphZones(scores, surface, weights)
    if domains is None:
        raise ValueError("topic zones need every question's topic")
    return TopicGraphZones(TopicMeans(scores, domains), surface, weights)


def zone_layout(name: str, zones: str, scores: np.ndarray, space: Space, surface: Surface, weights: np.ndarray,
                knobs: Knobs, ladder: tuple[Level, ...] = READ_LEVELS, strength: ZoneStrength = EqualStrength(),
                domains: tuple[str, ...] | None = None) -> GraphZoneLayout:
    """Zones on the block graph with every part picked by name; the reach is R = f D along the space's graph."""
    return GraphZoneLayout(name, zone_source(zones, scores, surface, weights, domains),
                           gz.FrontReach(knobs.focus_area, space.width), surface, knobs.floor, knobs.focus_strength,
                           strength, knobs.combine, ladder)


def per_block_layout(name: str, scores: np.ndarray, knobs: Knobs,
                     ladder: tuple[Level, ...] = READ_LEVELS) -> QuantileLevels:
    """The control the zones must beat (#19): the top share f of blocks by score, no zones, the same knobs."""
    return QuantileLevels(name, scores, knobs.focus_area, knobs.focus_strength, knobs.floor, ladder)
