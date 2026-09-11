"""Step 3 layout policies: which blocks each question reads sharp.

Every policy answers one question - the level of every block for a given set of questions - so
the evaluation loop does not know how a layout is made. A new way to allocate precision is a new
class with the same interface. Policies take a precision share (see budget.py): the share of the
precision range spent, 0 = everything coarse, 1 = everything sharp.

Invariants:
- Invariant: every policy at the same precision share spends the same share of weights.
- Invariant: Random is reproducible per question from its seed and differs between questions.
- Invariant: a question never sees its own mask through its topic's mean (leave-one-out).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from foqlens import zones
from foqlens import budget as bg
from foqlens.quant import Level


class LayoutPolicy(Protocol):
    name: str

    def levels(self, indices: np.ndarray) -> np.ndarray:
        """Level codes [len(indices), n_blocks] for the questions with these indices."""
        ...


def _rng(seed: int, i: int, precision_share: float, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, i, round(precision_share * 1000), salt])


class TopicMeans:
    """Mean mask of every topic, summed once; a question's own topic mean leaves the question out."""

    def __init__(self, scores: np.ndarray, domains: tuple[str, ...]):
        self.scores = scores
        self.domains = tuple(domains)
        labels = np.array(self.domains)
        self._sum = {d: scores[labels == d].sum(axis=0) for d in dict.fromkeys(self.domains)}
        self._count = {d: int((labels == d).sum()) for d in self._sum}

    def mean(self, index: int, topic: str) -> np.ndarray:
        if self.domains[index] == topic:
            return (self._sum[topic] - self.scores[index]) / (self._count[topic] - 1)
        return self._sum[topic] / self._count[topic]


def topic_masks(scores: np.ndarray, domains: tuple[str, ...], index: int, topic: str) -> np.ndarray:
    """Mean mask of a topic's questions; the question itself is left out when it belongs to that topic."""
    return TopicMeans(scores, domains).mean(index, topic)


@dataclass(frozen=True)
class Uniform:
    level: Level
    n_blocks: int

    @property
    def name(self) -> str:
        return f"uniform_{self.level.name.lower()}"

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.full((len(indices), self.n_blocks), int(self.level), dtype=np.uint8)


@dataclass(frozen=True)
class Directed:
    """Each question's own top blocks at bf16 within the precision share, the rest at the coarse level.

    scores [questions, n_blocks], background subtracted.
    """

    source: str
    precision_share: float
    scores: np.ndarray
    weights: np.ndarray
    coarse: Level = Level.NF4

    @property
    def name(self) -> str:
        return f"directed_{self.source}_{self.precision_share:.3f}"

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [bg.to_levels(bg.directed(self.scores[i], self.weights, self.precision_share), lo=self.coarse) for i in indices]
        )


@dataclass(frozen=True)
class TopicMask:
    """A topic's mask - of the question's own topic ("own") or of its paired topic ("other") - at the precision share."""

    target: str  # "own" | "other"
    source: str
    precision_share: float
    means: TopicMeans
    partner: dict
    weights: np.ndarray
    coarse: Level = Level.NF4

    @property
    def name(self) -> str:
        return f"{self.target}_topic_{self.source}_{self.precision_share:.3f}"

    def _mask(self, i: int) -> np.ndarray:
        own = self.means.domains[i]
        return self.means.mean(i, own if self.target == "own" else self.partner[own])

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [bg.to_levels(bg.directed(self._mask(int(i)), self.weights, self.precision_share), lo=self.coarse) for i in indices]
        )


@dataclass(frozen=True)
class Random:
    """Random blocks at bf16 within the same precision share, the rest at the coarse level; reproducible per question."""

    precision_share: float
    weights: np.ndarray
    seed: int = 0
    coarse: Level = Level.NF4

    @property
    def name(self) -> str:
        return f"random_{self.precision_share:.3f}"

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                bg.to_levels(bg.random_layout(self.weights, self.precision_share, _rng(self.seed, int(i), self.precision_share)), lo=self.coarse)
                for i in indices
            ]
        )


@dataclass(frozen=True)
class Backbone:
    """Generic block importance, the same for every question, for the whole precision share."""

    precision_share: float
    backbone: np.ndarray
    weights: np.ndarray
    coarse: Level = Level.NF4

    @property
    def name(self) -> str:
        return f"backbone_{self.precision_share:.3f}"

    def levels(self, indices: np.ndarray) -> np.ndarray:
        row = bg.to_levels(bg.directed(self.backbone, self.weights, self.precision_share), lo=self.coarse)
        return np.repeat(row[None], len(indices), axis=0)


@dataclass(frozen=True)
class BackboneFill:
    """The backbone for `share` of the precision share; the rest filled by the own topic, the other topic or random blocks."""

    fill: str  # "own" | "other" | "random"
    source: str  # topic mask source; unused for the random fill
    precision_share: float
    share: float
    backbone: np.ndarray
    weights: np.ndarray
    means: TopicMeans | None = None
    partner: dict | None = None
    seed: int = 0
    coarse: Level = Level.NF4
    dilation: str = "none"  # name of the neighbour table the fill is widened by (see neighbours.py)
    neighbours: np.ndarray | None = None

    def __post_init__(self) -> None:
        if (self.dilation == "none") != (self.neighbours is None):
            raise ValueError(f"dilation {self.dilation!r} needs a neighbour table exactly when it is not 'none'")

    @property
    def name(self) -> str:
        what = "random" if self.fill == "random" else f"{self.fill}_{self.source}"
        wide = "" if self.dilation == "none" else f"_{self.dilation}"
        return f"bb{self.share:.2f}_{what}{wide}_{self.precision_share:.3f}"

    def _fill(self, i: int) -> np.ndarray:
        if self.fill == "random":
            return _rng(self.seed, i, self.precision_share, salt=1).random(len(self.weights))
        own = self.means.domains[i]
        return self.means.mean(i, own if self.fill == "own" else self.partner[own])

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                bg.to_levels(
                    bg.layered(self.backbone, self._fill(int(i)), self.weights, self.precision_share, self.share, self.neighbours),
                    lo=self.coarse,
                )
                for i in indices
            ]
        )


class TopicZones:
    """The expert zones of every question's own and paired topic mask on the weight map, found once each."""

    def __init__(self, means: TopicMeans, partner: dict, coords: np.ndarray):
        self.means = means
        self.partner = partner
        self.coords = coords
        self._cache: dict[tuple[int, str], zones.Zones] = {}

    def get(self, index: int, target: str) -> zones.Zones:
        """target: "own" (leave-one-out) or "other" (the paired topic)."""
        key = (index, target)
        if key not in self._cache:
            own = self.means.domains[index]
            topic = own if target == "own" else self.partner[own]
            self._cache[key] = zones.find_zones(self.means.mean(index, topic), self.coords)
        return self._cache[key]


@dataclass(frozen=True)
class ZoneLayout:
    """Levels from expert zones at a focus area and a precision share (docs/zones.md).

    kind: "own" / "other" - the question's topic zones or its paired topic's; "random" - as many
    zones with the same radii as the own ones, around random blocks, the same centers at every focus area;
    "fixed" - one set of zones for every question (the backbone's); "uniform" - no zones, the
    precision share spent evenly, without a mask.
    """

    kind: str
    source: str
    focus_area: float
    precision_share: float
    coords: np.ndarray
    weights: np.ndarray
    topics: TopicZones | None = None
    fixed: zones.Zones | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        zones.check_focus_area(self.focus_area)
        bg.check_precision_share(self.precision_share)

    @property
    def name(self) -> str:
        if self.kind == "uniform":
            return f"zone_uniform_ps{self.precision_share:.3f}"
        return f"zone_{self.kind}_{self.source}_fa{self.focus_area:.2f}_ps{self.precision_share:.3f}"

    def _zones(self, i: int) -> zones.Zones:
        if self.kind == "uniform":
            return zones.Zones(centers=np.zeros((0, self.coords.shape[1])), radii=np.zeros(0))
        if self.kind == "fixed":
            return self.fixed
        if self.kind == "random":
            return zones.random_zones(self.topics.get(i, "own"), self.coords, _rng(self.seed, i, 0.0, salt=3))
        return self.topics.get(i, self.kind)

    def levels(self, indices: np.ndarray) -> np.ndarray:
        focus_area = 1.0 if self.kind == "uniform" else self.focus_area
        bits = zones.budget_bits(self.precision_share)
        rows = []
        for i in indices:
            psi = zones.log_sharpness(self.coords, self._zones(int(i)), focus_area)
            tie = _rng(self.seed, int(i), 0.0, salt=2).permutation(len(self.weights))
            rows.append(zones.layout_at_budget(psi, self.weights, bits, tie, hard_edge=focus_area == 0.0))
        return np.stack(rows)
