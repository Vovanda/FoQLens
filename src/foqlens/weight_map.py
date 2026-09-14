"""The weight map: blocks that light up together over many queries lie close (docs/quantization-filter.md).

Every block is described by its raw mask values over a set of questions, standardized per block;
the map is the first `dims` principal axes of those descriptions, so the distance on the map
approximates the co-activation distance sqrt(2 (1 - correlation)). Expert zones are drawn on it.

Invariant: the map is deterministic - the same masks give the same coordinates (axis signs fixed).
Invariant: scaling or shifting one block's mask values does not move it (standardized per block).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def coactivation_map(masks: np.ndarray, dims: int = 2) -> np.ndarray:
    """Raw masks [questions, n_blocks] -> coordinates of the blocks on the weight map [n_blocks, dims]."""
    x = masks.T.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / np.maximum(x.std(axis=1, keepdims=True), EPS)
    x = x - x.mean(axis=0, keepdims=True)  # principal axes of the blocks as points
    u, sv, _ = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :dims] * sv[:dims] / np.sqrt(x.shape[1])
    signs = np.sign(coords[np.abs(coords).argmax(axis=0), np.arange(dims)])
    return coords * np.where(signs == 0, 1.0, signs)
