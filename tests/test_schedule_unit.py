"""A run through several corpora goes in rounds, a share of each at a time, and resumes where it stopped."""

from collections import Counter

import pytest

from foqlens.schedule import Schedule, corpus_order

CORPORA = {"triviaqa": [f"t{i}" for i in range(997)], "nq_open": [f"n{i}" for i in range(361)],
           "arc": [f"a{i}" for i in range(96)]}


def asked(schedule: Schedule, written: dict[str, set[str]], stop_after: int | None = None) -> dict[str, set[str]]:
    """Run the pending rounds, writing every question, and stop after `stop_after` questions if given."""
    count = 0
    for _, todo in schedule.pending(written):
        for name, ids in todo.items():
            for i in ids:
                if stop_after is not None and count == stop_after:
                    return written
                assert i not in written.setdefault(name, set()), f"{i} asked twice"
                written[name].add(i)
                count += 1
    return written


def test_every_question_is_in_exactly_one_round():
    s = Schedule.build(CORPORA, seed=0)
    seen = Counter((name, i) for k in range(s.rounds) for name, ids in s.round(k).items() for i in ids)
    assert set(seen.values()) == {1}
    assert {name: {i for n, i in seen if n == name} for name in CORPORA} == {n: set(ids) for n, ids in CORPORA.items()}


@pytest.mark.parametrize("r", [1, 4, 11, 19])
def test_after_r_rounds_every_corpus_is_done_to_its_share(r):
    """Volodya 14.09: a run stopped at 20% gives 20% of every corpus."""
    s = Schedule.build(CORPORA, seed=0)
    for name, ids in CORPORA.items():
        done = sum(len(s.round(k)[name]) for k in range(r))
        assert abs(done - len(ids) * r / s.rounds) <= 1


@pytest.mark.parametrize("stop_after", [0, 1, 250, 777, 1453])
def test_a_stopped_run_restarts_without_repeating_or_missing_a_question(stop_after):
    s = Schedule.build(CORPORA, seed=0)
    written = asked(s, {}, stop_after)
    written = asked(s, written)  # the restart
    assert {n: written[n] for n in CORPORA} == {n: set(ids) for n, ids in CORPORA.items()}


def test_a_restart_skips_the_rounds_already_done():
    s = Schedule.build(CORPORA, seed=0)
    first = dict(s.round(0))
    written = {name: set(ids) for name, ids in first.items()}
    k, todo = next(iter(s.pending(written)))
    assert k == 1 and todo == s.round(1)


def test_the_order_is_seeded_and_the_corpus_own():
    ids = [str(i) for i in range(500)]
    assert corpus_order("a", ids, 0) == corpus_order("a", ids, 0)
    assert corpus_order("a", ids, 0) != corpus_order("a", ids, 1)
    assert corpus_order("a", ids, 0) != corpus_order("b", ids, 0)   # two corpora of one size are cut apart
    assert corpus_order("a", ids, 0) != ids                           # not the file's order
    assert sorted(corpus_order("a", ids, 0), key=int) == ids
