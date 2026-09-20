"""The checks a new oracle passes before its fields are trusted (plan ideal-models): each one read against what is
already known of the same questions - the oracle by trying's NLL at the same level, the lower rung's energy.

Questions are matched by (corpus, id): ids repeat across corpora, and the files hold different subsets and orders.

Invariant: a check reads only questions both sides hold, and says how many that is.
"""

from __future__ import annotations

import numpy as np

from foqlens.sensitivity import rank_correlation


def matched(keys_a: list[tuple[str, str]], keys_b: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Positions in a and in b of the questions both hold, in a's order."""
    where = {k: i for i, k in enumerate(keys_b)}
    pairs = [(i, where[k]) for i, k in enumerate(keys_a) if k in where]
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    a, b = zip(*pairs)
    return np.array(a), np.array(b)


def nll_agreement(nll: np.ndarray, reference: np.ndarray) -> dict:
    """The same answers' NLL at the same level, measured two ways: how far apart, over the questions both hold."""
    both = ~np.isnan(nll) & ~np.isnan(reference)
    gap = np.abs(nll[both] - reference[both])
    return {"questions": int(both.sum()), "max_abs": float(gap.max()) if both.any() else None,
            "median_abs": float(np.median(gap)) if both.any() else None,
            "mean_nll": float(nll[both].mean()) if both.any() else None,
            "mean_reference": float(reference[both].mean()) if both.any() else None}


def rank_agreement(scores: np.ndarray, reference: np.ndarray) -> dict:
    """Spearman's rho of every question's groups between an oracle and a reference ([questions, groups] both)."""
    rho = rank_correlation(scores, reference)
    return {"questions": int(len(rho)), "rho_median": float(np.median(rho)) if len(rho) else None,
            "rho_positive_share": float((rho > 0).mean()) if len(rho) else None}


def rung_ratio(upper: np.ndarray, lower: np.ndarray) -> dict:
    """The upper rung's error energy against the lower's on the same questions and blocks ([questions, n_blocks]):
    finite, and how much smaller - the median ratio over blocks where the lower rung moves anything."""
    moving = lower > 0
    ratio = upper[moving] / lower[moving]
    return {"finite": bool(np.isfinite(upper).all()), "never_negative": bool((upper >= 0).all()),
            "ratio_median": float(np.median(ratio)), "upper_not_above_share": float((ratio <= 1).mean())}


def end_gaps(ends: np.ndarray, topics: np.ndarray, tolerance: float) -> dict:
    """Where the base alone is enough, per topic: the answer's NLL with every block low against every block high
    ([questions, 2]) - the share where low is within the tolerance of high, the share where low is even better, and
    the median gap. There every group's precision field is the base."""
    found = {}
    for topic in np.unique(topics):
        low, high = ends[topics == topic, 0], ends[topics == topic, 1]
        both = ~np.isnan(low) & ~np.isnan(high)
        gap = low[both] - high[both]
        found[str(topic)] = {"questions": int(both.sum()), "within_tolerance": float((gap <= tolerance).mean()),
                             "low_better": float((gap < 0).mean()), "gap_median": float(np.median(gap)),
                             "high_median": float(np.median(high[both])), "low_median": float(np.median(low[both]))}
    return found
