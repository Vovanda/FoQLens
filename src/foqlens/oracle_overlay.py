"""The oracles' block scores at the granularity of the groups the oracles by trying read, and a bootstrap interval - read
from their kept files, never from the model.

- block_group_ids: every block's group (a layer's attention or the rest of the layer) from its layer and module kind.
- to_groups: blocks' scores summed over a group, so a block oracle meets the oracles by trying at their granularity.
- bootstrap: an interval of a statistic of the questions by resampling them.

Invariants:
- Invariant: to_groups keeps the sum of the scores; a NaN score adds nothing.
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


def bootstrap(values: np.ndarray, statistic: Callable[[np.ndarray], float], draws: int, seed: int,
              level: float) -> tuple[float, float]:
    """A `level` interval of `statistic` over `values`' first axis, by `draws` resamples."""
    rng = np.random.default_rng(seed)
    stats = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)]
    tail = (1 - level) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))
