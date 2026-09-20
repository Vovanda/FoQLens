"""The oracles' block scores at the granularity of the groups the oracles by trying read, brought to one shape and laid
over one another - read from their kept files, never from the model.

- block_group_ids: every block's group (a layer's attention or the rest of the layer) from its layer and module kind.
- to_groups: blocks' scores summed over a group, so a block oracle meets the oracles by trying at their granularity.
- field: the one method that makes an oracle a field, the same for every one of them - its values divided by a scale
  of its own (unit_scale) and cut into 0 ... 1 (unit_field). What is negative reads 0, a group that gains from a
  coarse reading being no group to raise, and what stands above the scale reads 1. The contract of the matrix -
  [questions, groups], one order of questions and of groups for all the oracles - is what lets the rungs lay out
  over the fields and one analysis serve all of them.
- overlay: several unit fields as one. `product` is agreement: a group stays high only where every oracle holds it
  high, and one oracle near zero puts it out; the n-th root brings the product back onto the scale of a single field,
  so overlays of two oracles and of four are read by the same rule. `mean` is the soft one, where a single high
  oracle carries a group against the others. `least` is the severest agreement - the lowest oracle alone.
- bootstrap: an interval of a statistic of the questions by resampling them.

Invariants:
- Invariant: to_groups keeps the sum of the scores; a NaN score adds nothing.
- Invariant: unit_field lands in 0 ... 1 and keeps the order of the values it is given.
- Invariant: an overlay of one field is that field; an overlay of fields in 0 ... 1 stays in 0 ... 1.
- Invariant: least <= product <= mean everywhere - the usual order of those means.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def block_group_ids(block_layer: np.ndarray, block_kind: np.ndarray, names: list[str]) -> np.ndarray:
    """Every block's group in the oracle by trying's order (foqlens.group_oracle names), from its layer and module
    kind: [n_blocks]."""
    label = [f"{layer}.{'attention' if kind.startswith('self_attn.') else 'mlp'}"
             for layer, kind in zip(block_layer.tolist(), block_kind.tolist())]
    return np.array([names.index(g) for g in label])


def to_groups(scores: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """[questions, n_blocks] -> [questions, n_groups]: every group's blocks summed."""
    out = np.zeros((scores.shape[0], n_groups))
    np.add.at(out.T, groups, np.nan_to_num(scores, nan=0.0).T)
    return out


def field(values: np.ndarray, share: float = 0.99) -> np.ndarray:
    """An oracle as a field: one method, the same for every one of them.

    The contract is the field, and it is one contract: a matrix [questions, groups] in 0 ... 1, the questions and the
    groups in an order given from outside and the same for all of them, a larger number meaning a group this question
    needs more. Whatever the oracle measured and in whatever units - an answer swept group by group, an energy of the
    error, a gradient - ends inside it. Only under one contract do the rungs lay out over the fields at all, and only
    then is one analysis written once and run over all of them.
    """
    return unit_field(values, unit_scale(values, share))


def unit_scale(values: np.ndarray, share: float = 0.99) -> float:
    """The scale of an oracle's own units: the `share` quantile of what it gives over the questions it was read on.

    A quantile and not the maximum: one question with an outlying group would otherwise set the scale for all of them
    and press every other field towards zero.
    """
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    return float(np.quantile(positive, share)) if positive.size else 1.0


def unit_field(values: np.ndarray, scale: float) -> np.ndarray:
    """An oracle's importance on one scale, 0 ... 1: its own units divided by its own `scale`, cut at both ends."""
    if scale <= 0:
        raise ValueError(f"a scale of {scale}: an oracle's scale is positive")
    return np.clip(np.nan_to_num(np.asarray(values, dtype=float), nan=0.0) / scale, 0.0, 1.0)


def overlay(fields: list[np.ndarray], how: str = "product") -> np.ndarray:
    """Several unit fields as one, on the same scale as a single field.

    `product` - the n-th root of the product, agreement: one oracle near zero puts a group out. `mean` - the soft
    one, a high oracle carries a group against the others. `least` - the lowest oracle alone.
    """
    stack = np.stack([np.asarray(f, dtype=float) for f in fields])
    if how == "product":
        return np.exp(np.log(np.clip(stack, 1e-12, None)).mean(axis=0))
    if how == "mean":
        return stack.mean(axis=0)
    if how == "least":
        return stack.min(axis=0)
    raise ValueError(f"no overlay {how!r}: product, mean or least")


def bootstrap(values: np.ndarray, statistic: Callable[[np.ndarray], float], draws: int, seed: int,
              level: float) -> tuple[float, float]:
    """A `level` interval of `statistic` over `values`' first axis, by `draws` resamples."""
    rng = np.random.default_rng(seed)
    stats = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)]
    tail = (1 - level) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))
