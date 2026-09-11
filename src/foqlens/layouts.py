"""Step 3 layout policies: which blocks each question reads sharp.

Every policy answers one question - the level of every block for a given set of questions - so
the evaluation loop does not know how a layout is made. A new way to allocate precision is a new
class with the same interface. Directed and Random take an aperture (see budget.py): the share of
weights read sharp, 0 = closed, 1 = fully open.

Invariants:
- Invariant: Directed and Random at the same aperture spend the same share of weights.
- Invariant: Random is reproducible per question from its seed and differs between questions.
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
class Random:
    """Random blocks at bf16 within the same aperture, the rest at the coarse level; reproducible per question."""

    aperture: float
    weights: np.ndarray
    seed: int = 0
    coarse: Level = Level.NF4

    @property
    def name(self) -> str:
        return f"random_{self.aperture:.3f}"

    def _rng(self, i: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, i, round(self.aperture * 1000)])

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [bg.to_levels(bg.random_layout(self.weights, self.aperture, self._rng(int(i))), lo=self.coarse) for i in indices]
        )
