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


def assess(frozen: np.ndarray, shots: np.ndarray, working: np.ndarray, seen: np.ndarray, layers: np.ndarray) -> dict:
    """One source on one set of questions: its masks in the frozen wrapper, in the two-example one, and its working
    reading, which sees only the blocks `seen` [n_blocks]; the layer profile of the frozen reading."""
    return {"questions": len(frozen), "wrappers": identification(frozen, shots),
            "working": identification(working[:, seen], frozen[:, seen]), "layers": layer_profile(frozen, layers)}


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
