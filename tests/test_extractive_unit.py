"""Answering in the model's own words, scored as SQuAD scores it - with a passage and without one."""

from foqlens.extractive import NO_ANSWER, exact_match, first_line, qa_prompt, token_f1
from foqlens.generation import question_prompt


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
