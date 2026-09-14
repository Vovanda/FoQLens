"""Answering in the model's own words, scored as SQuAD scores it - with a passage and without one."""

import importlib.util
from pathlib import Path

from foqlens.extractive import NO_ANSWER, exact_match, first_line, qa_prompt, token_f1
from foqlens.generation import question_prompt

ROOT = Path(__file__).resolve().parents[1]


def calibration():
    spec = importlib.util.spec_from_file_location("corpus_calibration", ROOT / "scripts" / "corpus_calibration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_closed_book_prompt_carries_no_passage():
    """Without a passage the answer can only come from the weights, so nothing may hint at one."""
    prompt = qa_prompt(None, "Who was the man behind The Chipmunks?", ((None, "Capital of France?", "Paris"),))
    assert "Context" not in prompt
    assert prompt.endswith("Question: Who was the man behind The Chipmunks?\nAnswer:")
    assert "Question: Capital of France?\nAnswer: Paris" in prompt


def test_a_bare_closed_book_question_is_the_free_answer_prompt():
    """One question format for the model's own answer, whichever script asks it."""
    assert qa_prompt(None, "q") == question_prompt("q")


def test_a_passage_prompt_shows_the_examples_first_and_ends_where_the_answer_goes():
    prompt = qa_prompt("ctx", "q", (("c1", "q1", "a1"), ("c2", "q2", "a2")))
    blocks = prompt.split("\n\n")
    assert blocks[0] == "Context: c1\nQuestion: q1\nAnswer: a1"
    assert blocks[-1] == "Context: ctx\nQuestion: q\nAnswer:"
    assert len(blocks) == 3


def test_exact_match_ignores_case_articles_and_punctuation():
    assert exact_match("The Sunset Boulevard.", ["Sunset Blvd", "Sunset Boulevard"]) == 1.0
    assert exact_match("Sunset Strip", ["Sunset Boulevard"]) == 0.0


def test_token_f1_gives_credit_for_a_partial_answer():
    assert token_f1("David Seville", ["David Seville"]) == 1.0
    assert 0.0 < token_f1("Seville", ["David Seville"]) < 1.0
    assert token_f1("Paris", ["David Seville"]) == 0.0


def test_an_unanswerable_question_is_right_only_when_the_model_says_so():
    assert exact_match(NO_ANSWER, []) == 1.0
    assert exact_match("Paris", []) == 0.0


def test_the_answer_ends_at_the_first_line():
    assert first_line(" David Seville\n\nQuestion: next") == "David Seville"


def test_trivia_answers_keep_every_alias_once_with_the_value_first():
    record = {"answer": {"value": "Sunset Boulevard",
                         "aliases": ["Sunset Blvd", "Sunset Boulevard", "West Sunset Boulevard"]}}
    assert calibration().closed_book_answers("triviaqa", record) == [
        "Sunset Boulevard", "Sunset Blvd", "West Sunset Boulevard"]


def test_a_question_asked_without_its_options_is_scored_against_the_right_options_text():
    rows = [{"question": "What gas do plants take in?", "choices": ["oxygen", "carbon dioxide", "helium", "neon"],
             "answer": 1}]
    assert calibration().closed_from_choices(rows) == [
        {"context": None, "question": "What gas do plants take in?", "answers": ["carbon dioxide"]}]


def test_a_question_that_points_at_its_options_is_left_out():
    """Without the options it has no answer, whatever the model knows."""
    pointing = ["Which of the following best describes the objects?", "Which of these steps should come first?",
                "The following are true EXCEPT", "Pick one of the items listed below."]
    rows = [{"question": q, "choices": ["a", "b", "c", "d"], "answer": 0} for q in pointing]
    assert calibration().closed_from_choices(rows) == []


def test_nq_open_answers_are_taken_as_listed():
    record = {"answer": ["14 December 1972 UTC", "December 1972"]}
    assert calibration().closed_book_answers("nq_open", record) == ["14 December 1972 UTC", "December 1972"]
