"""Step 3 layout policies: which blocks each question reads sharp.

Every policy answers one question - the level of every block for a given set of questions - so
the evaluation loop does not know how a layout is made. A new way to allocate precision is a new
class with the same interface. Policies take a precision share (see budget.py): the share of the
precision range spent, 0 = everything coarse, 1 = everything sharp.

A zone layout is made of three parts behind their own interfaces - the zones of a question
(ZoneSource), the field they make over the weight map (Field), and how the field becomes levels
(LevelRule) - so a new kind of zones, profile or rule is a new class, not a branch.

Invariants:
- Invariant: every policy at the same precision share spends the same share of weights.
- Invariant: Random is reproducible per question from its seed and differs between questions.
- Invariant: a question never sees its own mask through its topic's mean (leave-one-out).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

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


class FillSource(Protocol):
    """What a question's blocks are ranked by: a topic's mask or random numbers - a new one is a new class."""

    kind: ClassVar[str]  # the word in the policy name

    def label(self, source: str) -> str:
        """The policy-name part of this fill for a mask source."""
        ...

    def vector(self, index: int, n_blocks: int) -> np.ndarray:
        """A value per block [n_blocks] for question `index`; higher is taken first."""
        ...


@dataclass(frozen=True)
class OwnTopic:
    """The mean mask of the question's own topic, the question itself left out."""

    means: TopicMeans
    kind: ClassVar[str] = "own"

    def label(self, source: str) -> str:
        return f"own_{source}"

    def vector(self, index: int, n_blocks: int) -> np.ndarray:
        return self.means.mean(index, self.means.domains[index])


@dataclass(frozen=True)
class OtherTopic:
    """The mean mask of the question's paired topic."""

    means: TopicMeans
    partner: dict
    kind: ClassVar[str] = "other"

    def label(self, source: str) -> str:
        return f"other_{source}"

    def vector(self, index: int, n_blocks: int) -> np.ndarray:
        return self.means.mean(index, self.partner[self.means.domains[index]])


@dataclass(frozen=True)
class TopicMask:
    """A topic's mask - the question's own topic or its paired topic - at the precision share."""

    topic: FillSource
    source: str
    precision_share: float
    weights: np.ndarray
    coarse: Level = Level.NF4

    @property
    def name(self) -> str:
        return f"{self.topic.kind}_topic_{self.source}_{self.precision_share:.3f}"

    def levels(self, indices: np.ndarray) -> np.ndarray:
        mask = lambda i: self.topic.vector(int(i), len(self.weights))  # noqa: E731
        return np.stack([bg.to_levels(bg.directed(mask(i), self.weights, self.precision_share), lo=self.coarse) for i in indices])


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


class TopicZones:
    """The expert zones of every question's own and paired topic mask on the weight map, found once each."""

    def __init__(self, means: TopicMeans, partner: dict, coords: np.ndarray):
        self.means = means
        self.partner = partner
        self.coords = coords
        self._cache: dict[tuple[int, str], zones.Zones] = {}

    def _find(self, index: int, topic: str) -> zones.Zones:
        key = (index, topic)
        if key not in self._cache:
            self._cache[key] = zones.find_zones(self.means.mean(index, topic), self.coords)
        return self._cache[key]

    def own(self, index: int) -> zones.Zones:
        """The zones of the question's own topic, the question itself left out."""
        return self._find(index, self.means.domains[index])

    def other(self, index: int) -> zones.Zones:
        """The zones of the question's paired topic."""
        return self._find(index, self.partner[self.means.domains[index]])


# --- Zone layouts, made of three replaceable parts: which zones, what field they make, how it becomes levels.


class ZoneSource(Protocol):
    def zones(self, index: int) -> zones.Zones:
        """The expert zones of question `index`."""
        ...


class Field(Protocol):
    def field(self, coords: np.ndarray, zones_: zones.Zones) -> np.ndarray:
        """A value per block [n_blocks] from the zones on the weight map; higher is sharper."""
        ...


class LevelRule(Protocol):
    def levels(self, field: np.ndarray, index: int) -> np.ndarray:
        """Level codes [n_blocks] of question `index` from its field."""
        ...


@dataclass(frozen=True)
class OwnZones:
    """The zones of the question's own topic (leave-one-out)."""

    topics: TopicZones

    def zones(self, index: int) -> zones.Zones:
        return self.topics.own(index)


@dataclass(frozen=True)
class OtherZones:
    """The zones of the question's paired topic."""

    topics: TopicZones

    def zones(self, index: int) -> zones.Zones:
        return self.topics.other(index)


@dataclass(frozen=True)
class RandomZones:
    """As many zones with the same radii as the own ones, around random blocks - the same centers for every field."""

    topics: TopicZones
    seed: int = 0

    def zones(self, index: int) -> zones.Zones:
        return zones.random_zones(self.topics.own(index), self.topics.coords, _rng(self.seed, index, 0.0, salt=3))


@dataclass(frozen=True)
class MovedZones:
    """The query's own zones, carried elsewhere on the map as one rigid figure (zones.moved_zones).

    The control for a layout whose memory is a result: same count, same radii, same distances between
    the zones, another place. What it tests is the address alone.
    """

    topics: TopicZones
    weights: np.ndarray | None = None   # block sizes: with them the landing is matched by cost
    reach: float = 1.0
    seed: int = 0

    def zones(self, index: int) -> zones.Zones:
        return zones.moved_zones(self.topics.own(index), self.topics.coords, _rng(self.seed, index, 0.0, salt=7),
                                 self.weights, self.reach)


@dataclass(frozen=True)
class FixedZones:
    """One set of zones for every question (the backbone's)."""

    fixed: zones.Zones

    def zones(self, index: int) -> zones.Zones:
        return self.fixed


@dataclass(frozen=True)
class ZoneLayout:
    """Levels from expert zones (docs/quantization-filter.md): zones of a question -> a field -> levels.

    Each part is replaced by a new class with the same interface; the layout does not know which.
    """

    name: str
    source: ZoneSource
    field: Field
    rule: LevelRule
    coords: np.ndarray

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.stack([self.rule.levels(self.field.field(self.coords, self.source.zones(int(i))), int(i)) for i in indices])


@dataclass(frozen=True)
class LiftField:
    """The field of the graded layout: how far every block is lifted over the floor (zones.precision_lift)."""

    focus_area: float
    reach: float = 1.0
    combine: str = "sum"

    def __post_init__(self) -> None:
        zones.check_focus_area(self.focus_area)

    def field(self, coords: np.ndarray, zones_: zones.Zones) -> np.ndarray:
        return zones.precision_lift(coords, zones_, self.focus_area, self.reach, self.combine)


@dataclass(frozen=True)
class GradedLevels:
    """The rule of the graded layout: the lift becomes levels along a profile (zones.levels_from_lift)."""

    floor: Level
    stops: tuple[tuple[Level, float], ...]

    def levels(self, field: np.ndarray, index: int) -> np.ndarray:
        return zones.levels_from_lift(field, self.floor, self.stops[0][0], self.stops) if self.stops             else np.full(len(field), int(self.floor), dtype=np.uint8)


def graded_zone_layout(
    name: str, source: ZoneSource, focus_area: float, focus_strength: float, coords: np.ndarray,
    floor: Level = Level.D4, combine: str = "sum", halo: bool = False,
    stops: tuple[tuple[Level, float], ...] | None = None,
) -> ZoneLayout:
    """The layout of E010 (ADDENDUM-11, docs/quantization-filter.md): a floor everywhere, zones graded up to a ceiling.

    The ceiling is focus_strength of the way from the floor to the top of the ladder; the profile is
    even by default, with the lowest rung pushed past the edge when `halo` is on. There is no budget:
    what the layout costs is what its zones ask for.
    """
    ceiling = zones.ceiling_of(focus_strength, floor)
    profile = stops if stops is not None else zones.even_stops(floor, ceiling, halo=halo)
    reach = profile[-1][1] if profile else 1.0
    return ZoneLayout(name, source, LiftField(focus_area, reach, combine), GradedLevels(floor, profile), coords)


@dataclass(frozen=True)
class ShuffledLevels:
    """The levels of another policy, shuffled over the blocks: the same memory with no mask at all.

    The control of ADDENDUM-11 and of the preregistration's second baseline: it holds the layout's
    mean bits and its mix of levels exactly, and only forgets where they belong. Blocks differ in how
    many weights they hold, so the shuffle stays inside groups of equal weight - otherwise the control
    would quietly spend a little more or less memory than the layout it controls.
    """

    name: str
    of: LayoutPolicy
    weights: np.ndarray
    seed: int = 0

    def levels(self, indices: np.ndarray) -> np.ndarray:
        levels = self.of.levels(indices)
        out = levels.copy()
        groups = [np.flatnonzero(self.weights == w) for w in np.unique(self.weights)]
        for row, i in enumerate(indices):
            rng = _rng(self.seed, int(i), 0.0, salt=5)
            for group in groups:
                out[row, group] = levels[row, rng.permutation(group)]
        return out
