"""The overlay of the oracles: how far they agree, how much of what a question needs is the same for every question, and
whether a topic shares its zones - read from their kept scores, never from the model.

- to_groups: blocks' scores summed over a group (a layer's attention, the rest of the layer), so a block oracle meets
  the oracles by trying at their granularity.
- contrast: a group's score over its mean over the questions - a block oracle's sum over a group is mostly the group's
  size and the background every question shares, and its contrast is what the question itself asks.
- group_ranks: every question's groups ranked by a score, 0 the least needed, as a share of the groups - comparable
  across questions whatever the scale of the score.
- static_share: of the spread of the ranks over questions and groups, the share the groups' mean ranks explain - the part
  of what questions need that is the same for every question; the rest is what a filter by the question can win.
- jaccard_within_between: the mean Jaccard of the minimal masks of two questions of one topic, and of two of different
  topics: a topic with zones of its own shows the first above the second.
- bootstrap: an interval of a statistic of the questions by resampling them.
- chain_sets: every question's chain (its groups at `high`) and its switched-off groups as sets [questions, groups].
- pair_jaccard: two oracles' sets on the same questions, a Jaccard per question - what the oracles share.
- frequency, antinodes: how often each group is in a set over the questions; the antinodes are the groups nearly every
  question has in its chain, the nodes the groups nearly every question can switch off - the rest varies by question.
- band_spread: the share of each band of layers a set takes, its mean and its spread over the questions within a topic
  and between topics - where the spread is large the band is the question's, where small it is common.

Invariants:
- Invariant: static_share is 1 when every question ranks the groups alike and near 0 when the ranks are random.
- Invariant: a Jaccard of a mask with itself is 1; empty masks are left out.
- Invariant: the contrast of every read group has mean 1 over the questions; a scale per group changes nothing.
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


def contrast(scores: np.ndarray) -> np.ndarray:
    """Every group's score over its mean over the questions: [questions, groups] - what a question asks of a group
    beyond what every question does. A group no question reads stays 0."""
    mean = scores.mean(axis=0)
    return np.divide(scores, mean, out=np.zeros_like(scores, dtype=float), where=mean > 0)


def group_ranks(scores: np.ndarray) -> np.ndarray:
    """Every question's groups ranked by score, as a share: 0 the lowest, 1 the highest: [questions, groups]."""
    ranks = np.argsort(np.argsort(scores, axis=1, kind="stable"), axis=1, kind="stable").astype(float)
    return ranks / max(scores.shape[1] - 1, 1)


def static_share(ranks: np.ndarray) -> float:
    """The share of the ranks' spread the groups' mean ranks explain: 1 - within-group variance / total variance."""
    total = ranks.var()
    return float(1.0 - (ranks - ranks.mean(axis=0)).var() / total) if total > 0 else 1.0


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else np.nan


def jaccard_within_between(masks: np.ndarray, topics: np.ndarray) -> tuple[float, float]:
    """The mean Jaccard of the masks [questions, groups] of pairs of one topic, and of pairs of different topics."""
    keep = masks.any(axis=1)
    masks, topics = masks[keep], topics[keep]
    within, between = [], []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            (within if topics[i] == topics[j] else between).append(_jaccard(masks[i], masks[j]))
    return float(np.nanmean(within)) if within else np.nan, float(np.nanmean(between)) if between else np.nan


def bootstrap(values: np.ndarray, statistic: Callable[[np.ndarray], float], draws: int, seed: int,
              level: float) -> tuple[float, float]:
    """A `level` interval of `statistic` over `values`' first axis, by `draws` resamples."""
    rng = np.random.default_rng(seed)
    stats = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)]
    tail = (1 - level) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))


def chain_sets(orders: np.ndarray, minimal: np.ndarray, zeroed: np.ndarray,
               n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Every question's chain - the first `minimal` groups of its order - and its switched-off groups - the last
    `zeroed` of its order, the least needed: two bool [questions, groups]; a question with no chain (minimal < 0) has
    empty sets."""
    chain = np.zeros((len(orders), n_groups), dtype=bool)
    off = np.zeros_like(chain)
    for q, (order, m, z) in enumerate(zip(orders, minimal, zeroed)):
        if m < 0:
            continue
        chain[q, order[:m]] = True
        if z > 0:
            off[q, order[::-1][:z]] = True
    return chain, off


def pair_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per question, the Jaccard of two oracles' sets [questions, groups]; NaN where both are empty."""
    union = (a | b).sum(axis=1)
    return np.where(union > 0, (a & b).sum(axis=1) / np.maximum(union, 1), np.nan)


def frequency(sets: np.ndarray) -> np.ndarray:
    """The share of the questions whose set holds each group: [groups]."""
    return sets.mean(axis=0) if len(sets) else np.zeros(sets.shape[1])


def antinodes(freq: np.ndarray, share: float) -> np.ndarray:
    """The groups at least `share` of the questions hold: sorted group indices."""
    return np.flatnonzero(freq >= share)


def band_spread(sets: np.ndarray, layers: np.ndarray, bands: list[tuple[int, int]], topics: np.ndarray) -> list[dict]:
    """For each band [lo, hi) of layers: the share of the band's groups a question's set takes - its mean, its
    standard deviation over the questions of one topic (averaged over the topics), and the spread of the topics' means."""
    out = []
    for lo, hi in bands:
        inside = (layers >= lo) & (layers < hi)
        share = sets[:, inside].mean(axis=1)
        means = {t: share[topics == t].mean() for t in np.unique(topics)}
        within = np.mean([share[topics == t].std() for t in means]) if means else np.nan
        out.append({"band": [int(lo), int(hi)], "mean": float(share.mean()), "within_topic_std": float(within),
                    "between_topics_std": float(np.std(list(means.values())))})
    return out
