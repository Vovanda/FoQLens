"""The maps of a run: one file holding every oracle's map of every question at every tolerance.

A question no longer has one map. Every oracle gives it a field of demand, and every tolerance the answer is allowed
reads that field into its own map, so the maps of a run are a block with four axes - oracle, tolerance, question,
group. Keeping them as one block is what stops them being mixed up: the questions and the groups are listed once and
every map stands at its own place, named by the oracle and the tolerance that made it.

- demand: [oracles, questions, groups], the field itself, which does not depend on the tolerance.
- levels: [oracles, tolerances, questions, groups], the map read from it.
- bounds: the field scale - the three bounds a field is read into rungs at, one fewer than there are rungs. A
  tolerance `k` lets the answer lose k times as much, so a rung leaving a share r of the base's error is taken
  while demand stays below 1 / (1 + r / k). At k = 1 these are the bench's own bounds, one point of the space of
  three bounds and not the optimum. A field's own scale is found by the answers; the uniform field scale stands
  between the optima of the good fields and is what overlays are compared on (docs/glossary.md).
- labels: `<oracle>@k<tolerance>`, the name a map is answered and compared under.

Invariant: levels[o, j] is what reading demand[o] at the bounds of ks[j] gives, group for group.
Invariant: a label names one map of the block and no other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def bounds(ratios: np.ndarray, k: float) -> np.ndarray:
    """The field scale at tolerance `k`: the bounds a demand field is read into rungs at, [rungs, groups].

    A rung leaving a share r of the base's error is taken while the demand stays below 1 / (1 + r / k), so the
    scale is the ladder's own measurement and `k` - how much more the answer may lose - is the only knob.
    """
    return 1.0 / (1.0 + np.asarray(ratios, dtype=float) / float(k))


def read_map(demand: np.ndarray, ratios: np.ndarray, k: float, codes: Sequence[int], top: int) -> np.ndarray:
    """One oracle's map at one tolerance: every group takes the coarsest rung whose bound its demand stays below,
    and the top rung where none does. [questions, groups] demand -> [questions, groups] codes."""
    fits = demand[:, None, :] < bounds(ratios, k)[None, :, :]
    order = np.array(codes, dtype=np.uint8)
    return np.where(fits.any(axis=1), order[fits.argmax(axis=1)], top).astype(np.uint8)


def rung_agreement(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """How much of two maps is the same rung, group for group: [questions] in 0 ... 1.

    The strictest of the three likenesses - it charges a map for every group it reads one rung apart, however small
    that step is in memory.
    """
    return (np.asarray(left) == np.asarray(right)).mean(axis=-1)


def raised_jaccard(left: np.ndarray, right: np.ndarray, base: int) -> np.ndarray:
    """The shape of two maps against each other: the groups both lift over the intersection of the groups either
    lifts - [questions] in 0 ... 1, and 1 where neither lifts anything.

    Shape without height: a group read at D4 and the same group at D8 count as one and the same lift. What this
    misses is how far each was lifted, and that is what `bits_apart` holds.
    """
    a, b = np.asarray(left) != base, np.asarray(right) != base
    union = (a | b).sum(axis=-1)
    return np.where(union > 0, (a & b).sum(axis=-1) / np.maximum(union, 1), 1.0)


def bits_apart(left: np.ndarray, right: np.ndarray, bits: dict[int, float], weights: np.ndarray) -> np.ndarray:
    """What two maps disagree on, in memory: the weighted mean of |bits| a weight apart - [questions].

    The two likenesses above count groups; this one counts what the disagreement costs, so a pair differing on a
    heavy MLP is charged more than a pair differing on a light attention group. Zero means the same memory read the
    same way everywhere.
    """
    table = np.zeros(max(bits) + 1)
    for code, value in bits.items():
        table[code] = value
    gap = np.abs(table[np.asarray(left)] - table[np.asarray(right)])
    return (gap * weights).sum(axis=-1) / weights.sum()


def label(oracle: str, k: float) -> str:
    """The name a map is answered under: the oracle and the tolerance that made it."""
    return f"{oracle}@k{k:g}"


@dataclass(frozen=True)
class ScalePoint:
    """One point of a grid of field scales, as its answers came out: whose field, which bounds, what it cost and
    what it held - of the questions and of the hard ones among them."""

    oracle: str
    bounds: tuple[float, float, float]
    cost: float
    answered: int
    hard: int


def own_scale(points: Sequence[ScalePoint], answered: int, hard: int) -> ScalePoint | None:
    """An oracle's own field scale: of its points that hold the answer, the cheapest - None where none holds it.

    Holding the answer is `answered` questions and `hard` of the hard ones; the thresholds are the search's, not
    this function's. Ties in cost go to the lower bounds, so the choice does not depend on the order of the grid.
    """
    holds = [p for p in points if p.answered >= answered and p.hard >= hard]
    return min(holds, key=lambda p: (p.cost, p.bounds)) if holds else None


def scale_loss(point: ScalePoint, own: ScalePoint) -> tuple[int, int, float]:
    """What a point costs an oracle against its own scale: questions lost, hard questions lost, memory overpaid.

    None of the three goes below zero: a point that holds more than the oracle's own scale, or costs less, is not
    thereby allowed to pay for what it loses elsewhere - the three are read in order, and a loss stays a loss.
    """
    return (max(0, own.answered - point.answered), max(0, own.hard - point.hard),
            round(max(0.0, point.cost - own.cost), 4))


def least_worst_scale(points: Sequence[ScalePoint], answered: int, hard: int
                      ) -> tuple[tuple[float, float, float] | None, dict[str, tuple[int, int, float]]]:
    """Of the bounds every field of a grid holds a point at, the ones whose worst loss is the smallest.

    Every field keeps its own scale (`own_scale`), and a candidate is charged, field by field, what it loses
    against that scale (`scale_loss`); the worst loss is taken component by component and the smallest worst wins.

    This is a compromise over the fields given, and a compromise is only as good as they are: a weak field drags
    the answer into the loose region, where over half the network is lifted. It is not by itself the uniform field
    scale - that one stands between the optima of the fields worth having (docs/glossary.md) - so which fields go
    in is the caller's judgement, not this function's.
    """
    by_oracle: dict[str, list[ScalePoint]] = {}
    for point in points:
        by_oracle.setdefault(point.oracle, []).append(point)
    own = {name: own_scale(group, answered, hard) for name, group in by_oracle.items()}
    charged = {name: group for name, group in by_oracle.items() if own[name] is not None}
    if not charged:
        return None, {}
    candidates = set.intersection(*({p.bounds for p in group} for group in charged.values()))
    if not candidates:
        return None, {}
    at = {name: {p.bounds: p for p in group} for name, group in charged.items()}
    losses = {bounds: {name: scale_loss(at[name][bounds], own[name]) for name in charged} for bounds in candidates}
    worst = {bounds: tuple(max(loss[i] for loss in by_name.values()) for i in range(3))
             for bounds, by_name in losses.items()}
    best = min(candidates, key=lambda b: (worst[b], b))
    return best, losses[best]


@dataclass(frozen=True)
class Maps:
    """Every map of a run, the questions and the groups listed once."""

    corpus: np.ndarray  # [questions]
    ids: np.ndarray  # [questions]
    groups: np.ndarray  # [groups] names
    oracles: np.ndarray  # [oracles] names
    ks: np.ndarray  # [tolerances]
    demand: np.ndarray  # [oracles, questions, groups]
    levels: np.ndarray  # [oracles, tolerances, questions, groups]
    ratios: np.ndarray  # [rungs, groups]
    eps: np.ndarray  # [oracles, questions] the threshold each oracle found for each question

    def flat(self) -> dict[str, np.ndarray]:
        """The block as the layouts a run answers: label -> [questions, groups]."""
        return {label(str(o), float(k)): self.levels[i, j]
                for i, o in enumerate(self.oracles) for j, k in enumerate(self.ks)}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, corpus=self.corpus, ids=self.ids, groups=self.groups, oracles=self.oracles,
                            ks=self.ks, demand=self.demand, levels=self.levels, ratios=self.ratios, eps=self.eps)
        return path

    @classmethod
    def read(cls, path: Path) -> "Maps":
        kept = np.load(path, allow_pickle=True)
        return cls(**{f: kept[f] for f in ("corpus", "ids", "groups", "oracles", "ks", "demand", "levels",
                                           "ratios", "eps")})
