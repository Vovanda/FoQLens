"""Paired comparisons over questions.

Invariant: paired_bootstrap is reproducible from its seed and its interval always contains the
observed mean difference.
"""

from __future__ import annotations

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> dict:
    """Mean of a - b over paired items and its 95% percentile interval from resampling the items."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.ndim != 1 or diff.size == 0:
        raise ValueError("paired_bootstrap needs two equal non-empty 1-d arrays")
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(resamples, diff.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    mean = float(diff.mean())
    return {"mean": mean, "lo": float(min(lo, mean)), "hi": float(max(hi, mean)), "above_zero": bool(lo > 0), "n": int(diff.size)}
