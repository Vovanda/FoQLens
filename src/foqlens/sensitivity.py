"""Does the address stand for the sensitivity: an estimate of how much a question needs its blocks (the working address)
against the oracle (the gradient's Taylor score), block by block for every question.

- top_overlap: of a question's top share k of blocks by the estimate, the share that is also in the oracle's top k -
  where the estimate puts the ceiling, is the oracle's ceiling there too. Chance is k.
- rank_correlation: Spearman's rho of the two over a question's blocks - all of them, or only those outside the
  estimate's top k (the tail that is expected to fall to the base).

Invariant: top_overlap of a score against itself is 1, and rank_correlation of a score against itself is 1.
Invariant: both are per question: [questions].
"""

from __future__ import annotations

import numpy as np


def _top(scores: np.ndarray, share: float) -> np.ndarray:
    """bool [questions, n_blocks]: every question's top share of blocks by score, at least one."""
    k = max(1, int(round(share * scores.shape[1])))
    top = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(top, np.argpartition(-scores, k - 1, axis=1)[:, :k], True, axis=1)
    return top


def top_overlap(estimate: np.ndarray, oracle: np.ndarray, share: float) -> np.ndarray:
    """The share of every question's top `share` blocks by `estimate` that are in its top by `oracle`: [questions]."""
    mine, theirs = _top(estimate, share), _top(oracle, share)
    return (mine & theirs).sum(axis=1) / mine.sum(axis=1)


def _ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def rank_correlation(estimate: np.ndarray, oracle: np.ndarray, outside_share: float = 0.0) -> np.ndarray:
    """Spearman's rho of every question's blocks, leaving out its top `outside_share` by the estimate: [questions]."""
    keep = ~_top(estimate, outside_share) if outside_share > 0 else np.ones(estimate.shape, dtype=bool)
    rho = []
    for e, o, k in zip(estimate, oracle, keep):
        a, b = _ranks(e[k]), _ranks(o[k])
        a, b = a - a.mean(), b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        rho.append(float((a * b).sum() / denom) if denom > 0 else 0.0)
    return np.array(rho)
