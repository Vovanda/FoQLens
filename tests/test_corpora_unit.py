"""The free-answer corpora: every question carries a number, the same on every reading of the pinned revision."""

import pytest

from foqlens import corpora
from foqlens.corpora import Row, arc_closed_rows, hotpot_rows, nq_rows, squad_rows, trivia_answers, trivia_rows


def trivia(qid: str, question: str, value: str = "v", aliases: tuple = ()) -> dict:
    return {"question_id": qid, "question": question, "answer": {"value": value, "aliases": list(aliases)}}


def arc(qid: str, question: str, key: str = "B") -> dict:
    return {"id": qid, "question": question, "answerKey": key,
            "choices": {"label": ["A", "B", "C"], "text": ["oxygen", "carbon dioxide", "helium"]}}


def test_trivia_answers_keep_every_alias_once_with_the_value_first():
    record = trivia("tc_1", "q", "Sunset Boulevard", ("Sunset Blvd", "Sunset Boulevard", "West Sunset Boulevard"))
    assert trivia_answers(record) == ("Sunset Boulevard", "Sunset Blvd", "West Sunset Boulevard")


def test_a_trivia_question_listed_twice_is_kept_once_at_its_first_row():
    rows = trivia_rows([trivia("tc_2", "first"), trivia("tc_33", "other"), trivia("tc_2", "second reading")])
    assert [(r.id, r.question) for r in rows] == [("tc_2", "first"), ("tc_33", "other")]


def test_nq_open_numbers_its_questions_by_row_and_takes_the_answers_as_listed():
    rows = nq_rows([{"question": "when?", "answer": ["14 December 1972 UTC", "December 1972"]},
                    {"question": "who?", "answer": ["x"]}])
    assert rows == [Row("0", "when?", ("14 December 1972 UTC", "December 1972")), Row("1", "who?", ("x",))]


def test_squad_keeps_its_id_its_passage_and_no_answers_for_the_unanswerable():
    rows = squad_rows([{"id": "56dd", "question": "q", "context": "c", "answers": {"text": [], "answer_start": []}}])
    assert rows == [Row("56dd", "q", (), "c")]


def test_hotpot_joins_its_paragraphs_under_their_titles():
    rows = hotpot_rows([{"id": "5a8b", "question": "q", "answer": "a",
                         "context": {"title": ["A", "B"], "sentences": [["one. ", "two."], ["three."]]}}])
    assert rows == [Row("5a8b", "q", ("a",), "A: one. two.\n\nB: three.")]


def test_arc_asked_without_its_options_is_scored_against_the_right_options_text():
    assert arc_closed_rows([arc("Mercury_1", "What gas do plants take in?")]) == [
        Row("Mercury_1", "What gas do plants take in?", ("carbon dioxide",))]


def test_an_arc_question_that_points_at_its_options_is_left_out():
    """Without the options it has no answer, whatever the model knows."""
    pointing = ["Which of the following best describes the objects?", "Which of these steps should come first?",
                "The following are true EXCEPT", "Pick one of the items listed below."]
    assert arc_closed_rows([arc(f"id{i}", q) for i, q in enumerate(pointing)]) == []


def test_an_arc_question_whose_key_is_not_among_its_labels_is_left_out():
    assert arc_closed_rows([arc("x", "q", key="E")]) == []


def test_every_corpus_is_read_at_a_pinned_revision():
    """A dataset updated upstream must not change the corpus under a run."""
    for corpus in corpora.CORPORA.values():
        assert len(corpus.source.revision) == 40 and all(c in "0123456789abcdef" for c in corpus.source.revision)


@pytest.mark.data
@pytest.mark.parametrize("name, size", [("triviaqa", 9960), ("nq_open", 3610), ("squad_v2", 11873),
                                        ("hotpotqa", 7405), ("arc_challenge_closed", None),
                                        ("arc_easy_closed", None)])
def test_a_corpus_numbers_its_questions_uniquely_and_the_same_on_every_reading(name, size):
    rows, _ = corpora.read(name)
    ids = [r.id for r in rows]
    assert len(set(ids)) == len(ids)
    assert size is None or len(ids) == size
    assert [r.id for r in corpora.read(name)[0]] == ids
