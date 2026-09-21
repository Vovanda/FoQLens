"""What the precision fields of the questions share and where they differ - the statistics a filter is built from, read
from the kept fields, never from the model.

A field here is a question's value a group [questions, groups]: the field's value of its level (precision_field.values,
0.2 ... 1.0) or its height above the base (value - the base's value, >= 0).

- variance_shares: the field split into what every question shares (a group's mean), what its corpus adds, a question's
  offset (all of it higher or lower - the threshold) and the rest - the question's own pattern, the part an address
  must predict; the shares of the spread each explains.
- agreement: two fields on the same questions - the Spearman rank correlation over the groups, the quadratic-weighted
  kappa of the levels, the share of groups on one level - a value a question.
- consensus, band: the median of several oracles' fields a question, and the spread between them - the band of
  uncertainty.
- patterns: the fields as a sum of few non-negative patterns (NMF on the heights): the share of the heights k patterns
  explain, for k = 1.. - how many features a question's field needs.
- centres, profile, zone_stops: the zones of a field - its centres are the groups at a local maximum above the base,
  of any height; the profile is the mean value at every distance in layers from a centre, for the same kind of group
  and the other, and the radius of a rung is the farthest distance whose mean value is at that rung or above; the
  stops write one zone in the notation of the filter - every rung's reach as a share of that zone's radius.
- permutation_p: how often a statistic of shuffled questions reaches the one observed.
- shapes: what shape a question's layout has - the whole network at the base, a map of several rungs, or the whole
  network at the top. A flat layout carries no zones at all, and the two flat shapes come about for opposite reasons:
  at the base the question holds without precision anywhere, at the top the oracle found no threshold and raised
  everything. Both answer as the whole network does whatever an oracle knows, so a number read over all the questions
  at once says less than it seems.

Invariants:
- Invariant: a layout of one rung is `base` or `top` and never `map`; every question falls into exactly one shape.
- Invariant: the shares of variance_shares sum to 1.
- Invariant: agreement of a field with itself is 1 in every measure.
- Invariant: a field of one centre and a spread that falls with the distance has its centre found and a profile that
  falls.
- Invariant: a prediction equal to the truth is at error 0 against the background, one equal to the background at
  ratio 1.
- Invariant: the Jaccard of a field's centres against itself is 1 wherever it has one, and Holm keeps the order of a
  family of p-values, multiplying the smallest by the size of the family.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

NMF_ITERATIONS = 2000  # the fields are small ([~300, 70]): enough to converge, seconds
NMF_TOLERANCE = 1e-6


def variance_shares(x: np.ndarray, corpus: np.ndarray) -> dict[str, float]:
    """The shares of the spread of x [questions, groups] around its mean: the groups' means ("common"), the corpus's
    means beyond them ("corpus"), every question's mean beyond those ("offset"), and the rest ("pattern")."""
    total = ((x - x.mean()) ** 2).sum()
    common = x.mean(axis=0, keepdims=True)
    by_corpus = np.zeros_like(x)
    for c in np.unique(corpus):
        by_corpus[corpus == c] = x[corpus == c].mean(axis=0)
    rest = x - by_corpus
    offset = rest.mean(axis=1, keepdims=True)
    pattern = rest - offset
    parts = {"common": ((common - x.mean()) ** 2).sum() * len(x), "corpus": ((by_corpus - common) ** 2).sum(),
             "offset": (offset ** 2).sum() * x.shape[1], "pattern": (pattern ** 2).sum()}
    return {k: float(v / total) if total > 0 else 0.0 for k, v in parts.items()}


def agreement(a_values: np.ndarray, b_values: np.ndarray, a_levels: np.ndarray,
              b_levels: np.ndarray) -> dict[str, np.ndarray]:
    """Per question, how two fields agree: Spearman of the values over the groups, the quadratic-weighted kappa and
    the share of equal levels. A question where a field is constant has NaN where a rank or a kappa is undefined."""
    q = len(a_values)
    rho, kappa = np.full(q, np.nan), np.full(q, np.nan)
    for i in range(q):
        if np.ptp(a_values[i]) > 0 and np.ptp(b_values[i]) > 0:
            rho[i] = spearmanr(a_values[i], b_values[i]).statistic
        if len(np.unique(np.concatenate([a_levels[i], b_levels[i]]))) > 1:
            kappa[i] = cohen_kappa_score(a_levels[i], b_levels[i], weights="quadratic")
    return {"spearman": rho, "kappa": kappa, "equal": (a_levels == b_levels).mean(axis=1)}


def against_background(predicted: np.ndarray, true: np.ndarray, background: np.ndarray) -> dict[str, float]:
    """How far a prediction of a field lands from the truth, against the background - what answering with the mean of
    the questions already gives: the mean absolute error of each and their ratio.

    A rank cannot say this where the groups are few - over two groups a Spearman is +1 or -1 whatever the values are -
    and a layer of the bench holds two groups, attention and mlp. This reads on any number of them, one included.

    Invariant: a prediction equal to the truth has error 0, and one equal to the background has ratio 1.
    """
    error = float(np.abs(predicted - true).mean())
    plain = float(np.abs(background - true).mean())
    return {"error": error, "background_error": plain, "ratio": float(error / plain) if plain > 0 else float("nan")}


def consensus(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[oracles, questions, groups] -> the median a question and group, and the band (max - min) between the oracles."""
    return np.median(fields, axis=0), fields.max(axis=0) - fields.min(axis=0)


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per question, the share of centres two fields agree on: |a and b| / |a or b| over the groups, [questions].

    `a` and `b` are boolean [questions, groups] - a field's centres (`centres`). A question where neither field has a
    centre is NaN: there is nothing to agree or disagree about.

    Invariant: a field against itself is 1 wherever it has a centre, and 0 against a field whose centres are elsewhere.
    """
    both = np.logical_and(a, b).sum(axis=1).astype(float)
    either = np.logical_or(a, b).sum(axis=1).astype(float)
    return np.divide(both, either, out=np.full(len(both), np.nan), where=either > 0)


def clusters_against_corpora(x: np.ndarray, corpus: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Cluster the questions by their fields into as many clusters as there are corpora, and read the clustering
    against the corpora themselves: the adjusted Rand index, 0 for a split unrelated to the corpora and 1 for one that
    reproduces them.

    This says how much of a field is the kind of the task. It is the coarse half of what an address has to predict:
    the fine half is what one question asks beyond its corpus.
    """
    kinds = np.unique(corpus)
    if len(kinds) < 2 or len(x) <= len(kinds):
        return {"clusters": len(kinds), "rand": float("nan")}
    labels = KMeans(n_clusters=len(kinds), n_init=10, random_state=seed).fit_predict(x)
    return {"clusters": len(kinds), "rand": float(adjusted_rand_score(corpus, labels))}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """The Holm-Bonferroni correction over a family of tests: every p is multiplied by how many tests are still open
    when it is reached, in growing order, and kept non-decreasing. Corrected values above 1 are cut to 1.

    Invariant: the order of the p-values is kept, the smallest is multiplied by the size of the family, and a family
    of one is unchanged.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    n, out, running = len(ordered), {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (n - i)))
        out[name] = running
    return out


def patterns(heights: np.ndarray, ks: list[int], seed: int = 0) -> list[dict]:
    """For every k, the heights [questions, groups] >= 0 as W [questions, k] @ H [k, groups] by NMF: the share of the
    heights' squared norm explained, and W and H."""
    norm = (heights ** 2).sum()
    out = []
    for k in ks:
        model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=NMF_ITERATIONS, tol=NMF_TOLERANCE)
        w = model.fit_transform(heights)
        h = model.components_
        out.append({"k": k, "explained": float(1 - ((heights - w @ h) ** 2).sum() / norm) if norm > 0 else 1.0,
                    "w": w, "h": h})
    return out


def shapes(levels: np.ndarray, base: int, top: int) -> np.ndarray:
    """The shape of every question's layout [questions, groups] of level codes: [questions] of "base", "map" or "top".

    "base" is the whole network at the base rung - the question holds without precision anywhere; "top" is the whole
    network at the top - no threshold was found and everything was raised; "map" is everything else, the only shape
    that carries zones. A flat layout of any other single rung counts as "map": it still spends what that rung costs.

    Invariant: every question gets exactly one shape, and a layout of one rung is never "map" unless that rung is
    neither the base nor the top.
    """
    out = np.full(len(levels), "map", dtype="<U4")
    out[(levels == base).all(axis=1)] = "base"
    out[(levels == top).all(axis=1)] = "top"
    return out


def centres(values: np.ndarray, layers: np.ndarray, base: float) -> np.ndarray:
    """A question's zone centres: the groups above `base` whose value no group of their layer or of the layers next to
    them exceeds - [groups] bool. A peak is of any height (Volodya 20.09): a zone's ceiling is its own, D4 as well as
    D8, and everything runs down to the base."""
    out = np.zeros(len(values), dtype=bool)
    for g in np.flatnonzero(values > base):
        near = np.abs(layers - layers[g]) <= 1
        out[g] = values[g] >= values[near].max()
    return out


def zone_stops(values: np.ndarray, layers: np.ndarray, found: np.ndarray, base: float,
               rungs: list[float]) -> list[dict]:
    """Every centre of a question written in the notation of the filter (docs/precision-regulator.md, rule 3): the
    height of its peak, its radius - the first distance in layers where nothing around it is above the base any more -
    and the stop of every rung, the farthest distance the rung reaches as a share of that radius. A centre whose zone
    never comes down to the base inside the network has the radius of the network."""
    out = []
    reach = int(layers.max() - layers.min()) + 1
    for g in np.flatnonzero(found):
        near = np.abs(layers - layers[g])
        height = np.array([values[near == d].max() if (near == d).any() else base for d in range(reach + 1)])
        above = np.flatnonzero(height <= base)
        radius = int(above[0]) if len(above) else reach
        stops = {}
        for rung in rungs:
            if rung > values[g]:
                continue
            at = np.flatnonzero(height >= rung)
            stops[rung] = float((at[-1] + 1) / radius) if radius > 0 else 0.0
        out.append({"peak": float(values[g]), "radius": radius, "stops": stops})
    return out


def profile(values: np.ndarray, layers: np.ndarray, kinds: np.ndarray, found: np.ndarray,
            reach: int) -> dict[str, np.ndarray]:
    """The mean value at every distance 0..reach in layers from the centres of the questions, for groups of the centre's
    kind ("same") and of the other ("other"): values, found [questions, groups] -> {kind: [reach + 1]} with NaN where no
    group lies. A group is counted at its distance to the nearest centre of its question."""
    sums = {k: np.zeros(reach + 1) for k in ("same", "other")}
    counts = {k: np.zeros(reach + 1) for k in ("same", "other")}
    for q in range(len(values)):
        where = np.flatnonzero(found[q])
        if not len(where):
            continue
        dist = np.abs(layers[None, :] - layers[where][:, None])  # [centres, groups]
        nearest = dist.argmin(axis=0)
        d = dist[nearest, np.arange(len(layers))]
        same = kinds == kinds[where][nearest]
        for kind, mask in (("same", same), ("other", ~same)):
            keep = mask & (d <= reach)
            np.add.at(sums[kind], d[keep], values[q, keep])
            np.add.at(counts[kind], d[keep], 1)
    return {k: np.divide(sums[k], counts[k], out=np.full(reach + 1, np.nan), where=counts[k] > 0) for k in sums}


def tubes(values: np.ndarray, layers: np.ndarray, base: float) -> list[dict]:
    """The raised part of a question's field split into runs connected along the depth (Volodya 20.09: a zone is a tube
    that pierces the layers, its cross-section as wide as it needs at that depth): for every run its length in layers,
    the groups it holds, its widest and its mean cross-section, and where it starts and ends as a share of the depth.
    A group belongs to a run when its layer touches the run's."""
    above = np.flatnonzero(values > base)
    if not len(above):
        return []
    order = above[np.argsort(layers[above], kind="stable")]
    runs, current = [], [order[0]]
    for g in order[1:]:
        if layers[g] - layers[current[-1]] <= 1:
            current.append(g)
        else:
            runs.append(current)
            current = [g]
    runs.append(current)
    depth = max(layers.max() - layers.min() + 1, 1)
    out = []
    for run in runs:
        at = layers[np.array(run)]
        width = np.bincount(at - at.min())
        out.append({"layers": int(at.max() - at.min() + 1), "groups": len(run),
                    "widest": int(width.max()), "mean_width": float(width[width > 0].mean()),
                    "from": float(at.min() / depth), "to": float(at.max() / depth)})
    return out


def radius(curve: np.ndarray, at_least: float) -> int:
    """The farthest distance from 0 over which the curve stays at `at_least` or above without a break; -1 if not at 0."""
    below = np.flatnonzero(~(curve >= at_least))
    return int(below[0]) - 1 if len(below) else len(curve) - 1


def permutation_p(statistic: Callable[[np.ndarray], float], labels: np.ndarray, draws: int, seed: int) -> float:
    """The share of `draws` shuffles of the labels whose statistic reaches the observed one (one-sided, >=), with the
    observed counted among them."""
    rng = np.random.default_rng(seed)
    seen = statistic(labels)
    hits = sum(statistic(rng.permutation(labels)) >= seen for _ in range(draws))
    return float((hits + 1) / (draws + 1))
