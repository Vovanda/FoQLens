"""A sample of questions drawn from the candidates a slice of the runs gives, in shares by corpus.

The bench compares two sides - the questions only the top rung answers and the ones the lower rungs answer too -
and a difference between them is only about the rungs while their composition is the same. So one side's shares by
corpus are given to the other, and neither is a picture of its corpus mix alone.

- shares: what part of the candidates every corpus holds.
- pick: `n` questions in those shares, the remainder going to the corpora the shares cut hardest (largest
  remainders), and the choice inside a corpus made by a generator of the given seed.

Invariant: in_other_words keeps the id and the place of every question it rewords, and drops the ones it has no
wording for.
Invariant: pick returns exactly `n` questions while the candidates hold that many, without repeats.
Invariant: pick is determined by its seed - the same candidates and seed give the same questions in the same order.
Invariant: a corpus gets no more questions than it has candidates; what it cannot take goes to the others.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

import numpy as np

Question = tuple[str, str]  # corpus, id
Row = TypeVar("Row")  # corpora.Row, which the domain of a sample needs no import of


def in_other_words(laid: list[tuple[str, Row]], wordings: dict[Question, str],
                   keep_rest: bool = False) -> list[tuple[str, Row]]:
    """The questions of `laid` a wording is given for, each carrying that wording instead of its own.

    A paraphrase is the same question asked with other words, so it keeps the question's id and its place in the
    corpus, and only the text the model reads changes. What was measured on the question is then compared with what
    was measured on the paraphrase by that id.

    `keep_rest` holds the questions no wording was written for, in their own words and in their own place: a run that
    numbers its questions by where they lie in the draw needs the list to stay as long as it was.
    """
    out = []
    for corpus, row in laid:
        wording = wordings.get((corpus, row.id))
        if wording is not None:
            out.append((corpus, replace(row, question=wording)))
        elif keep_rest:
            out.append((corpus, row))
    return out


def shares(candidates: list[Question]) -> dict[str, float]:
    """What part of `candidates` every corpus holds."""
    counts: dict[str, int] = {}
    for corpus, _ in candidates:
        counts[corpus] = counts.get(corpus, 0) + 1
    total = len(candidates)
    return {corpus: count / total for corpus, count in counts.items()} if total else {}


def counts_for(wanted: dict[str, float], n: int, available: dict[str, int]) -> dict[str, int]:
    """How many questions every corpus gives for a sample of `n` in the shares `wanted`, capped by `available`.

    The whole parts come first; the questions they leave over go to the corpora whose share was cut hardest, and a
    corpus that runs out of candidates passes what it cannot take on to the others.
    """
    exact = {corpus: share * n for corpus, share in wanted.items()}
    taken = {corpus: min(int(value), available.get(corpus, 0)) for corpus, value in exact.items()}
    while sum(taken.values()) < n:
        room = [c for c in exact if taken[c] < available.get(c, 0)]
        if not room:
            break
        corpus = max(room, key=lambda c: (exact[c] - taken[c], -taken[c], c))
        taken[corpus] += 1
    return {corpus: count for corpus, count in taken.items() if count}


def pick(candidates: list[Question], n: int, seed: int, wanted: dict[str, float] | None = None) -> list[Question]:
    """`n` of `candidates` in the shares `wanted`, or in the candidates' own shares when none are given."""
    by_corpus: dict[str, list[Question]] = {}
    for question in sorted(candidates):
        by_corpus.setdefault(question[0], []).append(question)
    counts = counts_for(wanted or shares(candidates), n, {c: len(q) for c, q in by_corpus.items()})
    rng = np.random.default_rng(seed)
    drawn: list[Question] = []
    for corpus in sorted(counts):
        questions = by_corpus[corpus]
        chosen = rng.choice(len(questions), size=counts[corpus], replace=False)
        drawn += [questions[i] for i in sorted(chosen.tolist())]
    return drawn
