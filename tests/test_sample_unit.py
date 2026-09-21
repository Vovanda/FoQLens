"""The sample of questions: shares by corpus, the remainder, the seed."""

from foqlens import sample


def candidates(counts: dict[str, int]) -> list[tuple[str, str]]:
    return [(corpus, f"{corpus}-{i}") for corpus, n in counts.items() for i in range(n)]


def by_corpus(questions: list[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for corpus, _ in questions:
        counts[corpus] = counts.get(corpus, 0) + 1
    return counts


def test_shares_are_the_parts_of_the_candidates():
    assert sample.shares(candidates({"a": 3, "b": 1})) == {"a": 0.75, "b": 0.25}


def test_shares_of_nothing_are_nothing():
    assert sample.shares([]) == {}


def test_pick_gives_exactly_what_was_asked():
    drawn = sample.pick(candidates({"a": 40, "b": 30, "c": 30}), 50, seed=1)
    assert len(drawn) == 50
    assert len(set(drawn)) == 50


def test_pick_holds_the_shares_of_the_candidates():
    drawn = sample.pick(candidates({"a": 40, "b": 30, "c": 30}), 50, seed=1)
    assert by_corpus(drawn) == {"a": 20, "b": 15, "c": 15}


def test_pick_takes_the_shares_it_is_given_over_its_own():
    drawn = sample.pick(candidates({"a": 100, "b": 100}), 10, seed=1, wanted={"a": 0.8, "b": 0.2})
    assert by_corpus(drawn) == {"a": 8, "b": 2}


def test_the_remainder_goes_where_the_share_was_cut_hardest():
    # 10 of shares 0.34 / 0.33 / 0.33 are 3.4 / 3.3 / 3.3: three each, and the last one to the largest remainder.
    drawn = sample.pick(candidates({"a": 100, "b": 100, "c": 100}), 10, seed=2,
                        wanted={"a": 0.34, "b": 0.33, "c": 0.33})
    assert by_corpus(drawn) == {"a": 4, "b": 3, "c": 3}


def test_a_corpus_gives_no_more_than_it_has():
    drawn = sample.pick(candidates({"a": 2, "b": 100}), 10, seed=3, wanted={"a": 0.5, "b": 0.5})
    assert by_corpus(drawn) == {"a": 2, "b": 8}


def test_the_seed_decides_the_draw():
    counts = {"a": 40, "b": 30, "c": 30}
    assert sample.pick(candidates(counts), 20, seed=7) == sample.pick(candidates(counts), 20, seed=7)
    assert sample.pick(candidates(counts), 20, seed=7) != sample.pick(candidates(counts), 20, seed=8)


def test_the_order_of_the_candidates_does_not_move_the_draw():
    counts = {"a": 40, "b": 30, "c": 30}
    straight = sample.pick(candidates(counts), 20, seed=5)
    assert sample.pick(list(reversed(candidates(counts))), 20, seed=5) == straight


def test_asking_for_more_than_there_is_gives_all_of_it():
    drawn = sample.pick(candidates({"a": 3, "b": 2}), 50, seed=1)
    assert len(drawn) == 5


def test_a_reworded_question_keeps_its_id_and_its_place():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Row:
        id: str
        question: str

    laid = [("c", Row("1", "Who wrote it?")), ("c", Row("2", "Where is it?")), ("d", Row("1", "When?"))]
    got = sample.in_other_words(laid, {("c", "2"): "In what place is it?", ("d", "1"): "At what time?"})
    assert [(corpus, row.id, row.question) for corpus, row in got] == [
        ("c", "2", "In what place is it?"), ("d", "1", "At what time?")]  # question 1 of c has no wording: left out


def test_keeping_the_rest_leaves_the_list_as_long_as_it_was():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Row:
        id: str
        question: str

    laid = [("c", Row("1", "Who wrote it?")), ("c", Row("2", "Where is it?"))]
    got = sample.in_other_words(laid, {("c", "2"): "In what place is it?"}, keep_rest=True)
    assert [(row.id, row.question) for _, row in got] == [("1", "Who wrote it?"), ("2", "In what place is it?")]
