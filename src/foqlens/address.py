"""Is a mask source an address: does it name the question, not its wrapper, and does a cheap reading of it hold.

Written before any mask of this pass is looked at; every reading of a result is fixed here.

A source's masks are compared as their excess over the background - every mask minus the mean mask of the same
questions read the same way (the same wrapper, the same precision, the same layers), so that what all questions
share, the wrapper included, drops out and only what tells a question apart is left.

- identification(a, b): the same questions read two ways, a question's excess in `a` against every excess in `b` by
  cosine. The share of questions whose nearest excess in `b` is their own, both ways averaged; chance is 1/n. It is
  the address's first invariant: the zones follow the question, so its excess must find itself across a change that
  keeps the question.
- Two readings are compared: the question in the frozen 0-shot wrapper against the same question under two examples
  (the prompt changes, the question does not), and the working address - the first layers at the base precision, the
  pass the regulator can afford before the weights are chosen - against the full pass at bf16 on the same blocks.
- layer_profile: the share of every layer in the positive excess, averaged over the questions - where the address
  lives.
- Meaning, not words: identification between a question and its paraphrase (the same meaning in other words, names
  kept), against the same identification of their bags of tokens. A source that beats its bag finds the meaning; one
  that does not finds the words the two share.
- deep_address: the zones act after the working layers, so the working reading must name the address there, not only
  on the blocks it read. A ridge projection fitted on calibration questions carries the first `depth` layers onto the
  rest; the predicted deep address is identified against the real one. Few layers read carry little; many leave few
  to the zones - the best depth is between, and the run reports both curves.
- adaptive_depth: whether a question tells at a shallow depth that it needs a deeper one. Every depth predicts one
  target - the layers from the deepest depth on - so that depths differ only in what they read; per question its hit
  and margin; whether the questions a deeper reading rescues stand out at the shallow one (AUC); the stop-or-go-deeper
  policy's identification against its mean depth.
- stop_summary: how deep a set of questions reads under the silhouette rule. Read on two sets written for it, trivial
  questions and hard ones: the hard ones read deeper where the AUC of their stops against the trivial ones' is above
  0.5; a set that mostly hits the cap has no stop to compare.
- agreement: two sources score different blocks, so their masks are not compared entry by entry; they agree where they
  place the same questions near and far alike - the correlation of their question-by-question cosines.

Fixed readings: a source is an address where identification between the wrappers is at least USABLE; its working
reading holds where identification between the working and the full reading is at least USABLE too. Below it the
excess carries more of the reading than of the question.

Invariant: identification of masks against themselves is 1 when no two excesses point the same way, and chance is 1/n.
Invariant: layer_profile sums to 1 over the layers when every question has some positive excess.
"""

from __future__ import annotations

import numpy as np

from foqlens.projection import Projection

# An address that finds its own question in fewer than half the cases carries more of the reading than of the
# question: its zones would follow the wrapper as often as the meaning.
USABLE = 0.5
TINY = 1e-12  # a mask with no excess has no direction; its cosine is 0 against everything


def excess(masks: np.ndarray) -> np.ndarray:
    """Every mask minus the mean mask of the same reading: [questions, n_blocks]."""
    masks = np.asarray(masks, dtype=np.float64)
    return masks - masks.mean(axis=0)


def _unit(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, TINY)


def identification(a: np.ndarray, b: np.ndarray) -> dict:
    """How often a question's excess in `a` is nearest to its own in `b` and back; the same questions in one order."""
    if a.shape != b.shape:
        raise ValueError(f"two readings of the same questions, not {a.shape} and {b.shape}")
    cos = _unit(excess(a)) @ _unit(excess(b)).T
    n = len(cos)
    own = np.arange(n)
    found = ((cos.argmax(axis=1) == own).mean() + (cos.argmax(axis=0) == own).mean()) / 2
    off = cos[~np.eye(n, dtype=bool)]
    return {"identified": float(found), "chance": 1 / n, "own_cos": float(np.diag(cos).mean()),
            "other_cos": float(off.mean()) if off.size else 0.0}


def assess(frozen: np.ndarray, shots: np.ndarray, layers: np.ndarray) -> dict:
    """One source on one set of questions: its masks in the frozen wrapper against the two-example one, and the layer
    profile of the frozen reading."""
    return {"questions": len(frozen), "wrappers": identification(frozen, shots), "layers": layer_profile(frozen, layers)}


def working_address(frozen: np.ndarray, working: np.ndarray, seen: np.ndarray) -> dict:
    """A working reading against the full one on the blocks it sees (`seen` [n_blocks]): does the cheap pass find the
    same address."""
    return identification(working[:, seen], frozen[:, seen])


def deep_address(train_work: np.ndarray, train_full: np.ndarray, test_work: np.ndarray, test_full: np.ndarray,
                 layers: np.ndarray, weights: np.ndarray, depth: int, ridge: float, start: int = 0) -> dict:
    """Does the working reading of layers [start, depth) name the address where the zones act - the layers after it.

    The pass still runs every layer before `depth` at the base; `start` only chooses which of them the address is read
    from, so that layers close to the tokens can be left out. A ridge projection (projection.Projection) is fitted on
    the calibration questions from their working reading on the read blocks to their full reading on the deep ones; the
    laid-out questions' deep address predicted by it is identified against their real one. `ridge` is relative:
    alpha = ridge times the mean variance of a read block, so that one value means the same at every window. Beside
    it, the share of the weights left to the zones."""
    early, deep = (layers >= start) & (layers < depth), layers >= depth
    x = train_work[:, early]
    alpha = ridge * float(np.var(x, axis=0).mean()) * len(x)
    predicted = Projection.fit(x, train_full[:, deep], alpha).apply(test_work[:, early]).cpu().numpy()
    return {**identification(predicted, test_full[:, deep]), "zoned_weight_share": float(weights[deep].sum() / weights.sum())}


def predict_deep(train_work: np.ndarray, train_full: np.ndarray, test_work: np.ndarray, layers: np.ndarray,
                 read_until: int, target_from: int, ridge: float) -> np.ndarray:
    """The laid-out questions' address in the layers from `target_from` on, predicted from their working reading of
    layers [0, read_until) by a ridge projection fitted on the calibration questions: [test questions, target blocks].
    With one target for every depth, two depths differ only in what they read."""
    read, target = layers < read_until, layers >= target_from
    x = train_work[:, read]
    alpha = ridge * float(np.var(x, axis=0).mean()) * len(x)
    return Projection.fit(x, train_full[:, target], alpha).apply(test_work[:, read]).cpu().numpy()


def per_question(predicted: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Every question's own verdict: whether its predicted address is nearest to its own real one (hit), and by how
    much its own cosine exceeds the best other's (margin; below 0 is a miss)."""
    cos = _unit(excess(predicted)) @ _unit(excess(actual)).T
    own = np.diag(cos).copy()
    others = cos.copy()
    np.fill_diagonal(others, -np.inf)
    margin = own - others.max(axis=1)
    return {"hit": margin > 0, "margin": margin}


def rank_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """How often a positive scores above a negative (ties count half): 0.5 is no signal. NaN without both kinds."""
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    above = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(above)


def adaptive_depth(predicted: dict[int, np.ndarray], actual: np.ndarray, low: int, high: int) -> dict:
    """Does a question tell at depth `low` whether it needs `high`: per depth the questions hit; how far the predicted
    address moves from one depth to the next; among the questions missed at `low`, whether those that `high` rescues
    stand out by their margin at `low` or by how much their prediction still moved into `low` (AUC, 0.5 is no signal);
    and the policy "stop at low when the margin is at least t, else read to high" - identification against mean depth."""
    depths = sorted(predicted)
    verdicts = {d: per_question(predicted[d], actual) for d in depths}
    # on the excess, as identification is: the mean address is shared and far larger, so on raw predictions it alone
    # sets the cosine and every step looks like 1e-4
    moved = {d: 1 - np.sum(_unit(excess(predicted[d])) * _unit(excess(predicted[p])), axis=1)
             for p, d in zip(depths, depths[1:])}
    hit_low, hit_high = verdicts[low]["hit"], verdicts[high]["hit"]
    missed = ~hit_low
    rescued = hit_high[missed]
    margin = verdicts[low]["margin"]
    policy = []
    for t in np.quantile(margin, np.linspace(0, 1, 11)):
        deeper = margin < t
        policy.append({"threshold": float(t), "mean_depth": float(low + (high - low) * deeper.mean()),
                       "identified": float(np.where(deeper, hit_high, hit_low).mean())})
    return {
        "hits": {d: float(v["hit"].mean()) for d, v in verdicts.items()},
        "moved": {d: float(m.mean()) for d, m in moved.items()},
        "hit_low_and_high": int((hit_low & hit_high).sum()), "only_low": int((hit_low & ~hit_high).sum()),
        "only_high": int((~hit_low & hit_high).sum()), "neither": int((~hit_low & ~hit_high).sum()),
        "rescue_auc_margin": rank_auc(margin[missed], rescued),
        "rescue_auc_moved": rank_auc(moved[low][missed], rescued) if low in moved else float("nan"),
        "policy": policy,
    }


def silhouette_overlap(predicted: dict[int, np.ndarray], k: int) -> dict[int, np.ndarray]:
    """How much every question's silhouette - the top k blocks of its predicted excess, its future zones - keeps from
    the depth before: Jaccard [questions] at every depth but the first."""
    depths = sorted(predicted)
    tops = {d: np.argpartition(-excess(predicted[d]), k - 1, axis=1)[:, :k] for d in depths}
    overlap = {}
    for p, d in zip(depths, depths[1:]):
        shared = np.array([len(np.intersect1d(a, b, assume_unique=True)) for a, b in zip(tops[p], tops[d])])
        overlap[d] = shared / (2 * k - shared)
    return overlap


def depth_cap(n_layers: int, share: float) -> int:
    """The deepest a working reading may go, as a share of the network (Volodya 19.09): the budget holds whatever the
    question, and the same share carries to a model of another length."""
    if not 0 < share < 1:
        raise ValueError(f"the cap is a share of the network in (0, 1), not {share}")
    return int(share * n_layers)


def silhouette_stop(predicted: dict[int, np.ndarray], k: int, tolerance: float, patience: int, cap: int) -> np.ndarray:
    """Every question's stop (Volodya 19.09): read layer by layer; once the silhouette holds `patience` depths running
    (Jaccard at least 1 - tolerance), roll back to where that run began - the zones act from there. Reading never goes
    past `cap` layers, patience included: a question that has not settled by then stops as deep as the cap allows."""
    return silhouette_run(predicted, k, tolerance, patience, cap)[0]


def silhouette_run(predicted: dict[int, np.ndarray], k: int, tolerance: float, patience: int,
                   cap: int) -> tuple[np.ndarray, np.ndarray]:
    """silhouette_stop and whether each question settled: one that settles at the deepest depth the cap allows and one
    that never settles stop at the same depth, and only the second one hit the cap."""
    depths = [d for d in sorted(predicted) if d <= cap]
    overlap = silhouette_overlap({d: predicted[d] for d in depths}, k)
    held = np.stack([overlap[d] >= 1 - tolerance for d in depths[1:]])  # [depths - 1, questions]: depth i+1 holds i
    last = max(d for d in depths if d + patience <= cap)
    stops = np.full(held.shape[1], last)
    settled = np.zeros(held.shape[1], dtype=bool)
    for q in range(held.shape[1]):
        for i in range(len(depths) - patience):
            if depths[i] + patience > cap:
                break
            if held[i : i + patience, q].all():
                stops[q], settled[q] = depths[i], True
                break
    return stops, settled


def stop_summary(stops: np.ndarray, settled: np.ndarray) -> dict:
    """How deep a set of questions reads under the silhouette rule: the mean stop, how many stop at every depth, and
    the share that never settled and hit the cap."""
    return {"questions": len(stops), "mean_stop": float(stops.mean()),
            "stops": {int(d): int((stops == d).sum()) for d in np.unique(stops)},
            "capped_share": float((~settled).mean())}


def stop_policy(predicted: dict[int, np.ndarray], actual: np.ndarray, stops: np.ndarray, patience: int) -> dict:
    """The silhouette rule against a fixed depth: identification at every question's own stop, the mean stop and the
    mean layers read (stop + patience), beside identification at the fixed depth nearest the mean stop."""
    verdicts = {d: per_question(predicted[d], actual)["hit"] for d in predicted}
    at_stop = np.array([verdicts[int(s)][q] for q, s in enumerate(stops)])
    nearest = min(verdicts, key=lambda d: abs(d - stops.mean()))
    return {"identified": float(at_stop.mean()), "mean_stop": float(stops.mean()),
            "mean_read": float((stops + patience).mean()),
            "stops": {int(d): int((stops == d).sum()) for d in sorted(predicted)},
            "fixed_depth": int(nearest), "fixed_identified": float(verdicts[nearest].mean())}


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Do two sources place the same questions alike, whatever blocks each scores: the correlation of their
    question-by-question cosine matrices off the diagonal (representational similarity). 1 is the same geometry."""
    ca, cb = (_unit(excess(m)) @ _unit(excess(m)).T for m in (a, b))
    off = ~np.eye(len(ca), dtype=bool)
    return float(np.corrcoef(ca[off], cb[off])[0, 1])


def bag_of_tokens(token_ids: list[list[int]]) -> np.ndarray:
    """The lexical control of a paraphrase: every text as counts of its tokens, [texts, distinct tokens]. A source that
    finds a paraphrase no better than this finds its words, not its meaning."""
    vocab = {t: i for i, t in enumerate(sorted({t for ids in token_ids for t in ids}))}
    counts = np.zeros((len(token_ids), len(vocab)))
    for row, ids in enumerate(token_ids):
        np.add.at(counts[row], [vocab[t] for t in ids], 1.0)
    return counts


def layer_profile(masks: np.ndarray, layers: np.ndarray) -> dict[int, float]:
    """Every layer's share of the positive excess, averaged over the questions; `layers` [n_blocks] names each block's."""
    positive = np.clip(excess(masks), 0, None)
    totals = positive.sum(axis=1, keepdims=True)
    shares = positive / np.maximum(totals, TINY)
    return {int(layer): float(shares[:, layers == layer].sum(axis=1).mean()) for layer in np.unique(layers)}
