"""Step 3 layout policies: which blocks each question reads sharp.

Every policy answers one question - the level of every block for a given set of questions - so
the evaluation loop does not know how a layout is made. A new way to allocate precision is a new
class with the same interface. Policies take an aperture (see budget.py): the share of weights
read sharp, 0 = closed, 1 = fully open.

Invariants:
- Invariant: every policy at the same aperture spends the same share of weights.
- Invariant: Random is reproducible per question from its seed and differs between questions.
- Invariant: a question never sees its own mask through its topic's mean (leave-one-out).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from foqlens import budget as bg
from foqlens.quant import Level


class LayoutPolicy(Protocol):
    name: str

    def levels(self, indices: np.ndarray) -> np.ndarray:
        """Level codes [len(indices), n_blocks] for the questions with these indices."""
        ...


def _rng(seed: int, i: int, aperture: float, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, i, round(aperture * 1000), salt])


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
    """Each question's own top blocks at bf16 within the aperture, the rest at the coarse level.

    scores [questions, n_blocks], background subtracted.
    """

    source: str
    aperture: float
    scores: np.ndarray
    weights: np.ndarray
    coarse: Level = Level.NF4

    @property
    def name(self) -> str:
        return f"directed_{self.source}_{self.aperture:.3f}"

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [bg.to_levels(bg.directed(self.scores[i], self.weights, self.aperture), lo=self.coarse) for i in indices]
        )


@dataclass(frozen=True)
class TopicMask:
    """A topic's mask - of the question's own topic ("own") or of its paired topic ("other") - at the aperture."""

    target: str  # "own" | "other"
    source: str
    aperture: float
    means: TopicMeans
    partner: dict
    weights: np.ndarray
    coarse: Level = Level.NF4

    @property
    def name(self) -> str:
        return f"{self.target}_topic_{self.source}_{self.aperture:.3f}"

    def _mask(self, i: int) -> np.ndarray:
        own = self.means.domains[i]
        return self.means.mean(i, own if self.target == "own" else self.partner[own])

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [bg.to_levels(bg.directed(self._mask(int(i)), self.weights, self.aperture), lo=self.coarse) for i in indices]
        )


@dataclass(frozen=True)
class Random:
    """Random blocks at bf16 within the same aperture, the rest at the coarse level; reproducible per question."""

    aperture: float
    weights: np.ndarray
    seed: int = 0
    coarse: Level = Level.NF4

    @property
    def name(self) -> str:
        return f"random_{self.aperture:.3f}"

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                bg.to_levels(bg.random_layout(self.weights, self.aperture, _rng(self.seed, int(i), self.aperture)), lo=self.coarse)
                for i in indices
            ]
        )


@dataclass(frozen=True)
class Backbone:
    """Generic block importance, the same for every question, for the whole aperture."""

    aperture: float
    backbone: np.ndarray
    weights: np.ndarray
    coarse: Level = Level.NF4

    @property
    def name(self) -> str:
        return f"backbone_{self.aperture:.3f}"

    def levels(self, indices: np.ndarray) -> np.ndarray:
        row = bg.to_levels(bg.directed(self.backbone, self.weights, self.aperture), lo=self.coarse)
        return np.repeat(row[None], len(indices), axis=0)


@dataclass(frozen=True)
class BackboneFill:
    """The backbone for `share` of the aperture; the rest filled by the own topic, the other topic or random blocks."""

    fill: str  # "own" | "other" | "random"
    source: str  # topic mask source; unused for the random fill
    aperture: float
    share: float
    backbone: np.ndarray
    weights: np.ndarray
    means: TopicMeans | None = None
    partner: dict | None = None
    seed: int = 0
    coarse: Level = Level.NF4

    @property
    def name(self) -> str:
        what = "random" if self.fill == "random" else f"{self.fill}_{self.source}"
        return f"bb{self.share:.2f}_{what}_{self.aperture:.3f}"

    def _fill(self, i: int) -> np.ndarray:
        if self.fill == "random":
            return _rng(self.seed, i, self.aperture, salt=1).random(len(self.weights))
        own = self.means.domains[i]
        return self.means.mean(i, own if self.fill == "own" else self.partner[own])

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                bg.to_levels(bg.layered(self.backbone, self._fill(int(i)), self.weights, self.aperture, self.share), lo=self.coarse)
                for i in indices
            ]
        )
