"""Step 2+: mask geometry, computed on the step 1 mask vectors.

Written before the step 1 summary is opened. Every test is fixed here as a number, so that
reading the preregistered direction off the result needs no further choice. Masks are compared
as domain means (the mean mask over a domain's queries). "Top" sets are taken at several
fractions and every count is compared with what independent random sets of that size would give,
since the preregistration fixes no threshold for this exploratory pass.

Tests and the preregistered direction each one reads:

- additivity - least-squares fit of the biophysics mask on the biology and physics masks;
  "not additive" reads as a low R^2 and a junction zone (next test);
- junction zone - share of biophysics top blocks in neither the biology nor the physics top set,
  against the chance share (1 - f)^2; "a third group exists" reads as observed > chance;
- isthmus - among blocks outside both component top sets, share raised (above background) on
  biophysics, biology and physics at once, against the product of the three raised shares;
  "exists" reads as observed > chance;
- support shape - Gini, normalized entropy and participation ratio of a non-negative mask;
  "two bundles, not a blob" reads as biophysics concentration close to biology's with a wider
  mass (higher participation ratio);
- hierarchy - how many of the natural-science domains each block is a top block in, against the
  binomial expectation for independent domains; "hierarchical" reads as excess at both ends
  (blocks top in all domains and blocks top in exactly one);
- linearity - the same two-component fit on mean-pooled representations and on masks;
  "non-linear" reads as the mask R^2 clearly below the representation R^2.
"""

from __future__ import annotations

from math import comb

import numpy as np

from foqlens.scoring import gini, normalized_entropy

FRACTIONS = (0.01, 0.05, 0.10)


def domain_means(vectors: np.ndarray, labels: list[str], domains: list[str]) -> dict[str, np.ndarray]:
    labels_arr = np.asarray(labels)
    return {d: vectors[labels_arr == d].mean(axis=0) for d in domains}


def fit_two(target: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Least squares target ~ ca * a + cb * b (no intercept): coefficients and R^2 around the target mean."""
    x = np.stack([a, b], axis=1).astype(np.float64)
    t = target.astype(np.float64)
    coef, *_ = np.linalg.lstsq(x, t, rcond=None)
    residual = t - x @ coef
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - float((residual**2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return {"coef_a": float(coef[0]), "coef_b": float(coef[1]), "r2": r2}


def top_set(x: np.ndarray, frac: float) -> np.ndarray:
    """Boolean mask of the round(frac * n) highest entries (at least one)."""
    k = max(1, round(frac * x.size))
    out = np.zeros(x.size, dtype=bool)
    out[np.argsort(x)[-k:]] = True
    return out


def junction(mixed: np.ndarray, a: np.ndarray, b: np.ndarray, frac: float) -> dict:
    """Share of the mixed mask's top blocks that are top blocks of neither component."""
    top_m, top_a, top_b = top_set(mixed, frac), top_set(a, frac), top_set(b, frac)
    observed = float((top_m & ~top_a & ~top_b).sum() / top_m.sum())
    return {"observed": observed, "chance": (1 - frac) ** 2}


def isthmus(mixed: np.ndarray, a: np.ndarray, b: np.ndarray, frac: float) -> dict:
    """Among blocks outside both component top sets: share raised on the mixed domain and on both components."""
    outside = ~top_set(a, frac) & ~top_set(b, frac)
    raised_m, raised_a, raised_b = mixed[outside] > 0, a[outside] > 0, b[outside] > 0
    observed = float((raised_m & raised_a & raised_b).mean())
    chance = float(raised_m.mean() * raised_a.mean() * raised_b.mean())
    return {"observed": observed, "chance": chance, "outside_blocks": int(outside.sum())}


def concentration(x: np.ndarray) -> dict:
    """Concentration of a non-negative mask; participation ratio is 1 for a flat mask and 1/n for one block."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0, None)
    participation = float(x.sum() ** 2 / (x.size * (x**2).sum())) if x.any() else 0.0
    return {"gini": gini(x), "normalized_entropy": normalized_entropy(x) if x.any() else 0.0, "participation": participation}


def hierarchy(means: dict[str, np.ndarray], frac: float) -> dict:
    """Blocks by the number of domains they are top blocks in, observed and expected for independent domains."""
    tops = np.stack([top_set(v, frac) for v in means.values()])
    counts = np.bincount(tops.sum(axis=0), minlength=len(means) + 1)
    n, d = tops.shape[1], len(means)
    expected = [n * comb(d, k) * frac**k * (1 - frac) ** (d - k) for k in range(d + 1)]
    return {"domains": list(means), "observed": counts.tolist(), "expected": expected}


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


def linearity(rep: dict[str, np.ndarray], mask: dict[str, np.ndarray], mixed: str, a: str, b: str) -> dict:
    """The two-component fit and the cosine triple, once on representations and once on masks."""

    def side(v: dict[str, np.ndarray]) -> dict:
        return {
            "fit": fit_two(v[mixed], v[a], v[b]),
            "cos": {f"{mixed}|{a}": cosine(v[mixed], v[a]), f"{mixed}|{b}": cosine(v[mixed], v[b]), f"{a}|{b}": cosine(v[a], v[b])},
        }

    return {"representation": side(rep), "mask": side(mask)}
