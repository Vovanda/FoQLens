"""Step 3: turning masks into a layout of levels at a given precision share.

The precision share p is the share of the precision range spent: with two levels, the share of
weights read sharp, every other block read coarse. Share 0 - everything coarse; share 1 - everything
sharp. The mask says where to sharpen, the precision share how much. At share p with nf4 as the
coarse level the nominal mean is 4 + 12 p bits. Directed and random layouts use the same rule on the
same block weights, so at one precision share they spend the same budget.

Invariant: every layout at precision share p holds at least a share p of all weights sharp, and at
most one block more than needed.
"""

from __future__ import annotations

import numpy as np

from foqlens.precision import Controller
from foqlens.quant import Level


def block_weights(ctl: Controller) -> np.ndarray:
    """Number of weights in every block, in the controller's block order."""
    return np.concatenate([m.block_sizes() * m.in_features for m in ctl.modules.values()]).astype(np.int64)


def take_until(order: np.ndarray, weights: np.ndarray, amount: float, sharp: np.ndarray | None = None) -> np.ndarray:
    """Add blocks in the given order (skipping those already sharp) until their weights reach amount."""
    sharp = np.zeros(len(weights), dtype=bool) if sharp is None else sharp.copy()
    if amount <= 0:
        return sharp
    order = order[~sharp[order]]
    cum = np.cumsum(weights[order])
    sharp[order[: min(int(np.searchsorted(cum, amount)) + 1, len(order))]] = True
    return sharp


def check_precision_share(precision_share: float) -> float:
    """The precision share as given, refused outside [0, 1] (NaN included)."""
    if not 0.0 <= precision_share <= 1.0:
        raise ValueError(f"precision share {precision_share} outside [0, 1]")
    return precision_share


def select_by_budget(order: np.ndarray, weights: np.ndarray, precision_share: float) -> np.ndarray:
    """Blocks taken in the given order until their weights reach the precision share of all weights."""
    return take_until(order, weights, check_precision_share(precision_share) * weights.sum())


def directed(scores: np.ndarray, weights: np.ndarray, precision_share: float) -> np.ndarray:
    """The highest-scoring blocks within the precision share."""
    return select_by_budget(np.argsort(-scores, kind="stable"), weights, precision_share)


def random_layout(weights: np.ndarray, precision_share: float, rng: np.random.Generator) -> np.ndarray:
    """Random blocks within the same precision share."""
    return select_by_budget(rng.permutation(len(weights)), weights, precision_share)


def to_levels(sharp: np.ndarray, hi: Level = Level.BF16, lo: Level = Level.NF4) -> np.ndarray:
    """Boolean layout (any shape) -> level codes: sharp blocks at hi, the rest at lo."""
    return np.where(sharp, int(hi), int(lo)).astype(np.uint8)


def apply(ctl: Controller, sharp: np.ndarray, hi: Level = Level.BF16, lo: Level = Level.NF4) -> None:
    """Write a boolean layout, [n_blocks] or [batch, n_blocks], into the controller."""
    ctl.set_layout(to_levels(sharp, hi, lo))
