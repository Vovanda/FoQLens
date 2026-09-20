"""Zone strategies by name: a run is a list of words and knobs, and every word picks one class behind its interface.

The path of #19 - a score over the blocks, the surface it lies on, the zones found on it, how far they reach, how
strong each is, the levels they give - is assembled here from names, so a run script says `graph="mutual-nicdm",
medium="jump", zones="topic", strength="equal"` and never builds the parts itself. A new strategy is a new entry in
its table, not a branch in the old ones (CLAUDE.md, SOLID). Where the score comes from - the mask source - is the
bench's (pipeline.Bench.source); this module starts from the scores.

The space is built once: from calibration scores (Space.build) - the co-activation metric M1, its neighbour graph and
the graph's width, which fixes what f means for every question - or from the weights alone (Space.signal_path, the
signal's path, M3). A question's surface is that graph's fixed distances, or its medium of the question's own activity (M4, M4b).

Invariant: every name a table does not hold is refused with the names it does.
Invariant: a layout built here is the same as the one built from its parts by hand - the registry only picks.
Invariant: f = 0 lifts the zone centers alone and f = 1, g = 1 lifts every block of a question with a zone.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from foqlens import graph_zones as gz
from foqlens.coupling import STRONGEST, signal_path_table
from foqlens.layouts import (
    EqualStrength,
    GraphZoneLayout,
    GraphZoneSource,
    KnapsackLevels,
    QuantileLevels,
    QueryGraphZones,
    TopicGraphZones,
    TopicMeans,
    ZoneStrength,
    knapsack_price,
)
from foqlens.metric import (
    CoactivationMetric,
    FixedSurface,
    HarmonicConductance,
    JumpConductance,
    MediumSurface,
    Surface,
    geodesic,
    mutual_nicdm_table,
    neighbour_table,
    sweep_width,
)
from foqlens.quant import Level
from foqlens.runlog import stage
from foqlens.zones import READ_LEVELS

GRAPHS = {"union": neighbour_table, "mutual-nicdm": mutual_nicdm_table}  # how the block graph is built (#4)
MEDIA = {"harmonic": HarmonicConductance, "jump": JumpConductance}  # M4, M4b (#4); "fixed" is the graph's geodesic for every question
FIXED = "fixed"
ZONES = ("query", "topic")  # the question's own mask, or its topic's mean without it
COMBINE = ("sum", "max")  # rule 5
# How far a zone reaches, from (f, the width of the network, the blocks of the graph): f D for every zone (rule 1), f D
# split by the zones' own widths, or the radius at which a ball growing exponentially covers the share f (LogReach)
REACHES = {"equal": lambda f, width, blocks: gz.EqualReach(f, width),
           "proportional": lambda f, width, blocks: gz.ProportionalReach(f, width),
           "log": gz.LogReach}
NEIGHBOURS = 16  # blocks every block is linked to: the k of #4's measurements on E001's masks
LOG = logging.getLogger(__name__)


def _pick(table: dict | tuple, name: str, what: str):
    if name not in table:
        raise ValueError(f"unknown {what} {name!r}, expected one of {sorted(table)}")
    return table[name] if isinstance(table, dict) else name


@dataclass(frozen=True)
class Space:
    """A block graph - of calibration scores or of the weights: its neighbour table, edge lengths, width along it."""

    table: torch.Tensor  # [n_blocks, width] neighbours, PAD-padded
    lengths: torch.Tensor  # [n_blocks, width] edge lengths along the graph
    width: float  # the width of the network along the graph: f = 1 reaches it

    @classmethod
    def build(cls, calibration: np.ndarray, graph: str = "mutual-nicdm", k: int = NEIGHBOURS,
              device: str | torch.device = "cpu") -> Space:
        """From raw scores [questions, n_blocks] of calibration questions, never of the questions laid out."""
        table, lengths = _pick(GRAPHS, graph, "graph")(CoactivationMetric(calibration, device), k)
        return cls(table, lengths, sweep_width(geodesic(table, lengths)))

    @classmethod
    def signal_path(cls, weights: Mapping[str, torch.Tensor], n_heads: int, strongest: int = STRONGEST) -> Space:
        """Mechanism 5: the graph of the signal's path through the weights (coupling, M3 of #4) - no calibration."""
        table, lengths = signal_path_table(weights, n_heads, strongest)
        return cls(table, lengths, sweep_width(geodesic(table, lengths)))

    def surface(self, medium: str = FIXED, activity: np.ndarray | None = None,
                background: np.ndarray | None = None) -> Surface:
        """The graph's geodesic for every question, or through the medium of every question's `activity` against its
        `background`."""
        if medium == FIXED:
            return FixedSurface(self.table.cpu().numpy(), geodesic(self.table, self.lengths))
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
    budget_bits: float | None = None  # the knapsack's mean bits per weight; the zones follow f and g instead

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
                reach: str = "equal",
                domains: tuple[str, ...] | None = None) -> GraphZoneLayout:
    """Zones on the block graph with every part picked by name; the reach along the space's graph by `reach`."""
    return reach_layout(name, zone_source(zones, scores, surface, weights, domains), space, surface, knobs, ladder,
                        strength, reach)


def reach_layout(name: str, source: GraphZoneSource, space: Space, surface: Surface, knobs: Knobs,
                 ladder: tuple[Level, ...] = READ_LEVELS, strength: ZoneStrength = EqualStrength(),
                 reach: str = "equal") -> GraphZoneLayout:
    """Zones already found by `source`, laid out with the knobs and the reach picked by name: the part of a layout that
    changes along a sweep over f and the reach, while the source and its zones stay."""
    reaching = _pick(REACHES, reach, "reach")(knobs.focus_area, space.width, space.table.shape[0])
    return GraphZoneLayout(name, source, reaching, surface, knobs.floor, knobs.focus_strength, strength, knobs.combine,
                           ladder)


def per_block_layout(name: str, scores: np.ndarray, knobs: Knobs,
                     ladder: tuple[Level, ...] = READ_LEVELS) -> QuantileLevels:
    """The control the zones must beat (#19): the top share f of blocks by score, no zones, the same knobs."""
    return QuantileLevels(name, scores, knobs.focus_area, knobs.focus_strength, knobs.floor, ladder)


# ==== The mechanisms of the filter, by name (Volodya 19.09: checked from the simplest to verify up) ====


@dataclass(frozen=True)
class Inputs:
    """What a mechanism may read: the questions' scores and the calibration's, and the model's weights for M3."""

    scores: np.ndarray  # [questions, n_blocks] the laid-out questions' raw scores
    calibration: np.ndarray  # [calibration questions, n_blocks] raw scores of other questions
    block_weights: np.ndarray  # [n_blocks] weights a block holds
    domains: tuple[str, ...] | None = None  # every laid-out question's topic, for topic zones
    model_weights: Mapping[str, torch.Tensor] | None = None  # [out, in] of every module, for the signal's path
    n_heads: int | None = None
    prior: np.ndarray | None = None  # [n_blocks] a static sensitivity of every block (the oracle's mean on calibration)

    @property
    def background(self) -> np.ndarray:
        return self.calibration.mean(axis=0)

    @property
    def excess(self) -> np.ndarray:
        """The scores in excess of the calibration's background: a zone grows where a question stands out."""
        return self.scores - self.background


class Spaces:
    """The block graphs of one set of inputs, each built once on first use: they depend on the calibration or the
    weights, never on f, g or the reach, so a sweep over the knobs builds them once (the signal's path is minutes of
    CPU, the co-activation graph seconds)."""

    def __init__(self, inputs: Inputs, graph: str = "mutual-nicdm", k: int = NEIGHBOURS):
        self.inputs, self.graph, self.k = inputs, graph, k
        self._built: dict[str, Space] = {}
        self._zones: dict[str, tuple[GraphZoneSource, Surface]] = {}

    def coactivation(self) -> Space:
        if "coactivation" not in self._built:
            with stage(LOG, f"graph {self.graph} k {self.k} of {len(self.inputs.calibration)} calibration questions"):
                self._built["coactivation"] = Space.build(self.inputs.calibration, self.graph, self.k)
        return self._built["coactivation"]

    def signal_path(self) -> Space:
        if "signal_path" not in self._built:
            if self.inputs.model_weights is None or self.inputs.n_heads is None:
                raise ValueError("the signal's path needs the model's weights and its number of heads")
            with stage(LOG, f"graph of the signal's path k {self.k}"):
                self._built["signal_path"] = Space.signal_path(self.inputs.model_weights, self.inputs.n_heads, self.k)
        return self._built["signal_path"]

    def zones(self, mechanism: str, make) -> tuple[GraphZoneSource, Surface]:
        """A mechanism's zone source and surface, made once: its zones are found once per question and shared by every
        f and reach of the sweep."""
        if mechanism not in self._zones:
            self._zones[mechanism] = make()
        return self._zones[mechanism]


def _knapsack_of(name: str, sensitivity, spaces: Spaces, knobs: Knobs, ladder: tuple[Level, ...]) -> KnapsackLevels:
    """The knapsack over `sensitivity` (scores -> [questions, n_blocks]), its price fitted on the calibration's."""
    if knobs.budget_bits is None:
        raise ValueError("the knapsack needs a budget of mean bits per weight")
    inputs = spaces.inputs
    per_weight = lambda scores: np.clip(sensitivity(scores), 0.0, None) / inputs.block_weights  # noqa: E731
    price = knapsack_price(per_weight(inputs.calibration), inputs.block_weights, knobs.floor, ladder, knobs.budget_bits)
    return KnapsackLevels(name, per_weight(inputs.scores), price, knobs.floor, ladder)


def _knapsack(spaces: Spaces, knobs: Knobs, ladder: tuple[Level, ...], reach: str):
    # the raw score as the sensitivity - absolute, not the excess that tells questions apart
    return _knapsack_of("knapsack", lambda scores: scores, spaces, knobs, ladder)


def _knapsack_modulated(spaces: Spaces, knobs: Knobs, ladder: tuple[Level, ...], reach: str):
    # a static sensitivity of every block, modulated by how much louder the question's address is there than its
    # background: the address alone is mostly the background, and its thresholds would fall by module class
    inputs = spaces.inputs
    if inputs.prior is None:
        raise ValueError("the modulated knapsack needs a prior sensitivity of every block")
    background = np.maximum(inputs.background, np.finfo(float).tiny)
    return _knapsack_of("knapsack-modulated", lambda scores: inputs.prior * scores / background, spaces, knobs, ladder)


def _per_block(spaces: Spaces, knobs: Knobs, ladder: tuple[Level, ...], reach: str):
    return per_block_layout("per-block", spaces.inputs.excess, knobs, ladder)


def _static(spaces: Spaces, knobs: Knobs, ladder: tuple[Level, ...], reach: str):
    inputs, space = spaces.inputs, spaces.coactivation()

    def make():
        surface = space.surface()
        return QueryGraphZones(inputs.excess, surface, inputs.block_weights), surface

    source, surface = spaces.zones("static", make)
    return reach_layout("static", source, space, surface, knobs, ladder, reach=reach)


def _topic(spaces: Spaces, knobs: Knobs, ladder: tuple[Level, ...], reach: str):
    inputs, space = spaces.inputs, spaces.coactivation()

    def make():
        surface = space.surface()
        return zone_source("topic", inputs.excess, surface, inputs.block_weights, inputs.domains), surface

    source, surface = spaces.zones("topic", make)
    return reach_layout("topic", source, space, surface, knobs, ladder, reach=reach)


def _medium(spaces: Spaces, knobs: Knobs, ladder: tuple[Level, ...], reach: str):
    inputs, space = spaces.inputs, spaces.coactivation()

    def make():
        surface = space.surface("harmonic", activity=inputs.scores, background=inputs.background)
        return QueryGraphZones(inputs.excess, surface, inputs.block_weights), surface

    source, surface = spaces.zones("medium", make)
    return reach_layout("medium", source, space, surface, knobs, ladder, reach=reach)


def _signal_path(spaces: Spaces, knobs: Knobs, ladder: tuple[Level, ...], reach: str):
    inputs, space = spaces.inputs, spaces.signal_path()

    def make():
        surface = space.surface()
        return QueryGraphZones(inputs.excess, surface, inputs.block_weights), surface

    source, surface = spaces.zones("signal-path", make)
    return reach_layout("signal-path", source, space, surface, knobs, ladder, reach=reach)


# In the order they are checked: the simplest to build and to verify first (per block, no zones - the control).
MECHANISMS = {"per-block": _per_block, "knapsack": _knapsack, "knapsack-modulated": _knapsack_modulated,
              "static": _static, "topic": _topic, "medium": _medium,
              "signal-path": _signal_path}


def mechanism_layout(mechanism: str, inputs: Inputs | Spaces, knobs: Knobs, ladder: tuple[Level, ...] = READ_LEVELS,
                     graph: str = "mutual-nicdm", k: int = NEIGHBOURS, reach: str = "equal"):
    """The layout policy of a mechanism of the filter, by its name, over the questions of `inputs`; given Spaces, their
    graphs are reused (graph and k are theirs)."""
    spaces = inputs if isinstance(inputs, Spaces) else Spaces(inputs, graph, k)
    return _pick(MECHANISMS, mechanism, "mechanism")(spaces, knobs, ladder, reach)
