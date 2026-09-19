"""Layout policies: the level of every block for each question.

Every policy answers one question - the level of every block for a given set of questions - so
the evaluation loop does not know how a layout is made. A new way to allocate precision is a new
class with the same interface.

A zone layout on the block graph (GraphZoneLayout) is made of parts behind their own interfaces - the zones of a
question (GraphZoneSource), the surface they reach along (metric.Surface), how far they reach (graph_zones.Reach),
how strong each is (ZoneStrength) - so a new kind of zones, reach or strength is a new class, not a branch. The
per-block control (QuantileLevels) takes the same knobs with no zones; AttentionLevel holds rule 6 over any policy.
The policies of the first bench (Uniform, Directed, TopicMask, Random, ShuffledLevels) take a precision share (see
budget.py): the share of the precision range spent, 0 = everything coarse, 1 = everything sharp.

Invariants:
- Invariant: every precision-share policy at the same share spends the same share of weights.
- Invariant: Random is reproducible per question from its seed and differs between questions.
- Invariant: a question never sees its own mask through its topic's mean (leave-one-out).
- Invariant: the per-block control lifts the top share f of the blocks (ceil(fN), ties aside), at f = 0 the top one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np

from foqlens import graph_zones, zones
from foqlens import budget as bg
from foqlens.metric import Surface
from foqlens.projection import Projection
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


@dataclass(frozen=True)
class ShuffledLevels:
    """The levels of another policy, shuffled over the blocks: the same memory with no mask at all.

    A control of the preregistration: it holds the layout's
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


# --- Zones on the block graph (#4, #19): the zones of a question, how far they reach, how strong each is.


class GraphZoneSource(Protocol):
    def zones(self, index: int) -> graph_zones.GraphZones:
        """The expert zones of question `index` on the block graph."""
        ...


class ZoneStrength(Protocol):
    def strengths(self, index: int, zones_: graph_zones.GraphZones) -> np.ndarray:
        """Every zone's own strength in [0, 1]: how far its ceiling rises of the way g allows."""
        ...


@dataclass(frozen=True)
class QueryGraphZones:
    """The zones of the question's own mask - the address read from the query itself."""

    scores: np.ndarray  # [questions, n_blocks], background subtracted
    surface: Surface
    weights: np.ndarray  # block sizes in weights

    def zones(self, index: int) -> graph_zones.GraphZones:
        return graph_zones.find_graph_zones(self.scores[index], self.surface.metric(index), self.surface.table, self.weights)


class TopicGraphZones:
    """The zones of the question's own topic mean (leave-one-out), found once per question."""

    def __init__(self, means: TopicMeans, surface: Surface, weights: np.ndarray):
        self.means, self.surface, self.weights = means, surface, weights
        self._cache: dict[int, graph_zones.GraphZones] = {}

    def zones(self, index: int) -> graph_zones.GraphZones:
        if index not in self._cache:
            mean = self.means.mean(index, self.means.domains[index])
            self._cache[index] = graph_zones.find_graph_zones(mean, self.surface.metric(index), self.surface.table, self.weights)
        return self._cache[index]


@dataclass(frozen=True)
class EqualStrength:
    """Every zone at full strength: rule 2 as it is, one ceiling for all zones."""

    def strengths(self, index: int, zones_: graph_zones.GraphZones) -> np.ndarray:
        return np.ones(len(zones_.centers))


# The share of calibration zone centers whose projected score stays below the strength scale Z: the top 1% saturate
# at full strength instead of one outlier pressing every other zone down, as a normalisation by the maximum would (#19).
STRENGTH_QUANTILE = 0.99
# The lift of a block at the per-block threshold: above 0, so it reads the first rung over the floor - any positive
# lift below 1 / (rungs above the floor) does, and the share lifted is then the top f of the blocks, ties aside.
THRESHOLD_LIFT = 1e-6


@dataclass(frozen=True)
class ProjectedStrength:
    """A zone's strength from the heads (#19): s_i = clip(sigma(A)[c_i] / Z, 0, 1).

    signals [questions, units] are the heads' energies (source 3 of #18), clipped at 0; sigma is the fitted
    projection of #18 onto every block (projection.Projection.apply, centred as it was fitted), read at the
    zone's center; `scale` Z is fixed on calibration (strength_scale) so strengths compare across questions.
    """

    signals: np.ndarray
    projection: Projection
    scale: float

    def strengths(self, index: int, zones_: graph_zones.GraphZones) -> np.ndarray:
        a = np.clip(np.asarray(self.signals[index : index + 1], dtype=float), 0.0, None)
        sigma = self.projection.apply(a)[0].cpu().numpy()
        return np.clip(sigma[zones_.centers] / self.scale, 0.0, 1.0)


def strength_scale(projection: Projection, signals: np.ndarray, centers: list[np.ndarray],
                   quantile: float = STRENGTH_QUANTILE) -> float:
    """Z: the `quantile` of the projected scores at the zone centers of calibration questions (signals [q, units],
    centers[q] their zones' centers)."""
    sigma = projection.apply(np.clip(np.asarray(signals, dtype=float), 0.0, None)).cpu().numpy()
    at_centers = np.concatenate([sigma[q, c] for q, c in enumerate(centers) if len(c)])
    return float(np.quantile(at_centers, quantile))


@dataclass(frozen=True)
class GraphZoneLayout:
    """Levels from zones on the block graph: zones -> how far each reaches -> lifts -> levels in rungs (#19).

    Every part is a class behind its own interface - the zones (GraphZoneSource), the surface they
    reach along (metric.Surface: the graph's fixed distances, or its medium per question), the reach
    (graph_zones.Reach), the strength of a zone (ZoneStrength) - and the knobs mean the same whichever
    source the zones come from: f is the reach, g the ceiling. The level map is the even profile without a
    halo (zones.levels_from_rungs), on `ladder` - the rungs the run's model reads (regulator.kernel_ladder).
    """

    name: str
    source: GraphZoneSource
    reach: graph_zones.Reach
    surface: Surface
    floor: Level
    focus_strength: float
    strength: ZoneStrength = EqualStrength()
    combine: str = "sum"
    ladder: tuple[Level, ...] = zones.READ_LEVELS

    def levels(self, indices: np.ndarray) -> np.ndarray:
        rows = []
        for i in indices:
            found = self.source.zones(int(i))
            lifts = graph_zones.zone_lifts(found, self.reach.radii(found), self.surface.metric(int(i)))
            ceilings = zones.zone_ceilings(self.strength.strengths(int(i), found), self.focus_strength, self.floor,
                                           self.ladder)
            rows.append(zones.levels_from_rungs(lifts, ceilings, self.floor, self.combine, self.ladder))
        return np.stack(rows)


@dataclass(frozen=True)
class AttentionLevel:
    """Rule 6 of docs/quantization-filter.md over any policy: a q_proj or k_proj block never reads ZERO.

    A zero row of q_proj or k_proj flattens attention - a distortion, not emptiness - so where a policy leaves
    such a block at ZERO (a ZERO base outside the zones) it reads the attention level A instead. Every other
    block, and every attention block the policy lifted, keeps the policy's level.
    """

    policy: LayoutPolicy
    blocks: np.ndarray  # bool [n_blocks]: the blocks of q_proj and k_proj
    level: Level  # A: the lowest non-zero rung by default (docs)

    @property
    def name(self) -> str:
        return self.policy.name

    def levels(self, indices: np.ndarray) -> np.ndarray:
        codes = np.asarray(self.policy.levels(indices), dtype=np.uint8)
        return np.where(self.blocks[None] & (codes == int(Level.ZERO)), np.uint8(int(self.level)), codes)


@dataclass(frozen=True)
class QuantileLevels:
    """The per-block regulator (#19) - the control the zones must beat, not a mechanism.

    The top share f of blocks by the question's own score rise over the floor, graded by score up to the
    ceiling of g; the level map is the zones' own even profile, so what differs is only that there are no
    zones. It spends the same share of blocks whatever the question - a preset budget - and knows no
    connectedness.
    """

    name: str
    scores: np.ndarray  # [questions, n_blocks]
    focus_area: float
    focus_strength: float
    floor: Level
    ladder: tuple[Level, ...] = zones.READ_LEVELS

    def __post_init__(self) -> None:
        zones.check_focus_area(self.focus_area)

    def levels(self, indices: np.ndarray) -> np.ndarray:
        ceiling = zones.ceiling_of(self.focus_strength, self.floor, self.ladder)
        rows = []
        for i in indices:
            s = np.asarray(self.scores[int(i)], dtype=float)
            threshold = np.quantile(s, 1.0 - self.focus_area)
            span = s.max() - threshold
            graded = (s - threshold) / span if span > 0 else np.ones_like(s)
            # a block at the threshold reads the first rung: the top share f is lifted, at f = 0 the top block alone
            lift = np.where(s >= threshold, THRESHOLD_LIFT + (1.0 - THRESHOLD_LIFT) * graded, 0.0)
            rows.append(zones.levels_from_rungs(lift[None], [ceiling], self.floor, ladder=self.ladder))
        return np.stack(rows)
