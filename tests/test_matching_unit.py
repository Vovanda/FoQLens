"""The two-step metric: the model answers, then the options are matched to what it wrote."""

import numpy as np
import pytest

from foqlens import matching
from foqlens.evaluate import LETTERS
from foqlens.matching import first_answer, match_prompt

OPTIONS = ["Allopatric", "Sympatric", "Isolated", "Polyploidy"]


def test_the_answer_ends_where_the_model_starts_inventing_questions():
    """Base checkpoints have no chat template: after answering they keep writing a question of their own."""
    written = "The random assortment of chromosomes\n\nQuestion: Which of the following?\nAnswer: A population"
    assert first_answer(written) == "The random assortment of chromosomes"
    assert first_answer("  Sympatric speciation  ") == "Sympatric speciation"
    assert first_answer("") == ""


def test_the_matcher_is_never_told_which_option_is_right():
    prompt = match_prompt("What kind of speciation?", "It happened in one place", OPTIONS)
    assert "Sympatric" in prompt and "It happened in one place" in prompt
    for marker in ("right", "correct", "answer is"):
        assert marker not in prompt.lower()
    assert prompt.rstrip().endswith("Option:")   # the letter comes next, as in the one-step metric


def test_every_option_is_listed_once_under_its_own_letter():
    lines = match_prompt("q", "a", OPTIONS).splitlines()
    listed = [line for line in lines if line[:1] in LETTERS and line[1:3] == ". "]
    assert [line[0] for line in listed] == list(LETTERS)
    assert [line[3:] for line in listed] == OPTIONS


def test_a_mismatched_number_of_options_is_refused():
    with pytest.raises(ValueError):
        match_prompt("q", "a", OPTIONS[:3])


def test_the_matcher_reports_the_letter_it_picked_for_every_answer(monkeypatch):
    """It runs at bf16 whatever the layout under test was, so its own lean is a constant."""
    scores = np.log(np.array([[0.1, 0.7, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]]))
    monkeypatch.setattr(matching, "letter_logprobs_batch", lambda model, tok, prompts, ids: scores[: len(prompts)])

    class Controller:
        def __init__(self):
            self.levels = None

        def set_all(self, level):
            self.levels = level

    ctl = Controller()
    picked = matching.LetterMatcher((1, 2, 3, 4), None, None, ctl).match(
        ["in one place", "in two places"], ["q1", "q2"], [OPTIONS, OPTIONS])
    assert picked.tolist() == [1, 0]
    assert ctl.levels is not None and ctl.levels.name == "BF16"
