"""The order a run takes through several corpora: in rounds, every corpus a share at a time, resumable.

A run of thousands of questions can stop at any point - the card is shared, the machine runs out of
room, a thermal pause - and what it has written by then must be of use: a run stopped at 20% gives 20%
of every corpus, not all of the first one and none of the rest (Volodya 14.09). So the run goes in
rounds, and round k takes the k-th slice of every corpus.

Within a corpus the order is a seeded permutation, not the file's: the first rows of a dataset are
often of one kind (TriviaQA's open with questions from one quiz site), so a slice in file order would
not be a sample of the corpus. The permutation depends on the corpus name as well as the seed, so two
corpora of the same size are not cut at the same positions.

A restart reads what is already written and skips it; nothing else is kept between runs.

Invariant: every question of every corpus is in exactly one round.
Invariant: after r of R rounds every corpus is done to within one question of r/R of its size.
Invariant: the same corpora and seed give the same rounds.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

# Rounds of a run: each takes 5% of every corpus, so a stop leaves the collections balanced to within 5%.
# Not 1%: a round asks a corpus in few batches, and a batch runs as long as its longest reply - at 1%
# ARC-Challenge came to 9 questions a round, one batch of up to 512 steps for 9 answers (2026-09-14).
ROUNDS = 20


def corpus_order(corpus: str, ids: list[str], seed: int) -> list[str]:
    """The corpus's questions in a seeded order of its own."""
    rng = np.random.default_rng([seed, zlib.crc32(corpus.encode())])
    return [ids[i] for i in rng.permutation(len(ids)).tolist()]


@dataclass(frozen=True)
class Schedule:
    """Which questions of which corpus each round asks."""

    orders: dict[str, list[str]]  # corpus -> its questions in schedule order
    rounds: int = ROUNDS

    @classmethod
    def build(cls, corpora: dict[str, list[str]], seed: int, rounds: int = ROUNDS) -> Schedule:
        return cls({name: corpus_order(name, ids, seed) for name, ids in corpora.items()}, rounds)

    def round(self, k: int) -> dict[str, list[str]]:
        """Round k: the k-th of `rounds` equal slices of every corpus (sizes differ by at most one)."""
        return {name: order[len(order) * k // self.rounds: len(order) * (k + 1) // self.rounds]
                for name, order in self.orders.items()}

    def pending(self, written: dict[str, set[str]]) -> Iterator[tuple[int, dict[str, list[str]]]]:
        """The rounds still to ask, each without what is already written; empty rounds are skipped."""
        for k in range(self.rounds):
            todo = {name: [i for i in ids if i not in written.get(name, set())] for name, ids in self.round(k).items()}
            todo = {name: ids for name, ids in todo.items() if ids}
            if todo:
                yield k, todo
