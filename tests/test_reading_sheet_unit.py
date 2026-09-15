"""A sheet of answers for Claude to read and the marks written back on it."""

from dataclasses import replace

import pytest

from foqlens.corpora import Row
from foqlens.reading_sheet import CUT, DEFAULTS, Sheet, parse_marks, render
from foqlens.selection import Answer, Reading, Turn

ANSWER = Answer(corpus="triviaqa", id="tc_1", revision="0f7faf33", model="google/gemma-4-E2B-it@3e22461f",
                level="bf16", prompt="short-0", reply="Abiogenesis", answer="Abiogenesis", reasoning=None,
                exact_match=0.0, f1=0.0, judge_with_reference=0.987654, judge_without_reference=0.4,
                tokens=3, stopped=True)
ROW = Row("tc_1", "The Miller-Urey experiment aimed to investigate what?", ("Chemical origins of life",))


def test_unmarked_answers_take_the_default_and_marks_cover_numbers_and_ranges():
    readings = parse_marks("o 2 5: in other words\nn 7-8", 8, Reading.WRONG)
    assert [r for r, _ in readings] == [Reading.WRONG, Reading.OTHER_WORDS, Reading.WRONG, Reading.WRONG,
                                        Reading.OTHER_WORDS, Reading.WRONG, Reading.NO_ANSWER, Reading.NO_ANSWER]
    assert readings[1][1] == "in other words" and readings[0][1] == ""


def test_an_empty_marking_leaves_the_whole_sheet_at_its_default():
    assert parse_marks("", 3, Reading.RIGHT) == [(Reading.RIGHT, "")] * 3


def test_where_the_judges_disagree_every_answer_is_marked():
    assert DEFAULTS[Turn.DISAGREE] is None
    with pytest.raises(ValueError, match=r"\[3\]"):
        parse_marks("r 1\nw 2", 3, None)
    assert [r for r, _ in parse_marks("r 1 3\nw 2", 3, None)] == [Reading.RIGHT, Reading.WRONG, Reading.RIGHT]


@pytest.mark.parametrize("marks, error", [("w 1\no 1", "twice"), ("w 4", "not on a sheet"), ("w 0", "not on a sheet"),
                                          ("x 1", "unknown reading"), ("w 2-4", "not on a sheet")])
def test_a_mark_that_cannot_be_meant_is_an_error(marks, error):
    with pytest.raises(ValueError, match=error):
        parse_marks(marks, 3, Reading.WRONG)


def test_the_sheet_shows_the_answer_and_its_references_but_never_the_judges_scores():
    text = render([ANSWER], {"tc_1": ROW})
    assert text.startswith("1. Q: The Miller-Urey") and "REF: Chemical origins of life" in text
    assert "A: Abiogenesis" in text and "0.98" not in text and "0.4" not in text


def test_a_long_text_is_cut_with_a_mark_and_the_passage_shown_only_when_asked():
    rows = {"tc_1": replace(ROW, question="why " * 200, context="The passage.\nIt goes on.")}
    text = render([replace(ANSWER, answer=None)], rows)
    assert CUT in text and "P:" not in text and text.endswith("A: ")
    assert "   P: The passage. It goes on." in render([ANSWER], rows, passage=True)


def test_a_question_without_an_answer_says_so_on_the_sheet():
    assert "REF: (the passage has no answer)" in render([ANSWER], {"tc_1": replace(ROW, answers=())})


def test_a_sheet_survives_its_round_trip():
    sheet = Sheet("nq_open", "bf16", Turn.DOUBTFUL_NO, ("863", "3589"))
    assert Sheet.from_json(sheet.to_json()) == sheet
