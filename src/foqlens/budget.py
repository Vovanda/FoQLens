"""Step 3: turning masks into a precision layout at a given aperture.

The aperture is the share of weights read sharp (bf16); every other block is read coarse.
Aperture 0 is a closed lens - everything coarse; aperture 1 is fully open - everything sharp.
The mask says where to sharpen, the aperture how much. At aperture a with nf4 as the coarse level
the nominal mean is 4 + 12 a bits. Directed and random layouts use the same rule on the same
block weights, so at one aperture they spend the same budget.

Invariant: every layout at aperture a holds at least a share a of all weights sharp, and at most
one block more than needed.
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


def select_by_budget(order: np.ndarray, weights: np.ndarray, aperture: float) -> np.ndarray:
    """Blocks taken in the given order until their weights reach the aperture's share of all weights."""
    return take_until(order, weights, aperture * weights.sum())


def directed(scores: np.ndarray, weights: np.ndarray, aperture: float) -> np.ndarray:
    """The highest-scoring blocks within the aperture."""
    return select_by_budget(np.argsort(-scores, kind="stable"), weights, aperture)


def random_layout(weights: np.ndarray, aperture: float, rng: np.random.Generator) -> np.ndarray:
    """Random blocks within the same aperture."""
    return select_by_budget(rng.permutation(len(weights)), weights, aperture)


def layered(backbone: np.ndarray, fill: np.ndarray, weights: np.ndarray, aperture: float, share: float) -> np.ndarray:
    """The backbone's top blocks for `share` of the aperture, the rest of the aperture in the fill's order."""
    sharp = directed(backbone, weights, aperture * share)
    remaining = aperture * weights.sum() - weights[sharp].sum()
    return take_until(np.argsort(-fill, kind="stable"), weights, remaining, sharp)


def to_levels(sharp: np.ndarray, hi: Level = Level.BF16, lo: Level = Level.NF4) -> np.ndarray:
    """Boolean layout (any shape) -> level codes: sharp blocks at hi, the rest at lo."""
    return np.where(sharp, int(hi), int(lo)).astype(np.uint8)


def apply(ctl: Controller, sharp: np.ndarray, hi: Level = Level.BF16, lo: Level = Level.NF4) -> None:
    """Write a boolean layout, [n_blocks] or [batch, n_blocks], into the controller."""
    ctl.set_layout(to_levels(sharp, hi, lo))
