"""The corpus selection's records and rule: an answer per line, three judges, a frozen file of numbers."""

from dataclasses import replace

import pytest

from foqlens.io import answers_path, append_answers, read_answers, written_ids
from foqlens.selection import Answer, FrozenCorpus, Reason, Verdict, split_reasoning, strip_markup, verdict

ANSWER = Answer(corpus="triviaqa", id="tc_1", revision="0f7faf33", model="google/gemma-4-E2B-it@3e22461f",
                level="bf16", prompt="short-0shot", reply="Paris", answer="Paris", reasoning=None,
                exact_match=1.0, f1=1.0, judge_with_reference=0.99, judge_without_reference=0.97,
                tokens=2, stopped=True)


@pytest.mark.parametrize("em, judge, expected", [
    (1.0, 0.99, (Verdict.KNOWN, Reason.JUDGES_AGREE)),
    (0.0, 0.01, (Verdict.UNKNOWN, Reason.JUDGES_AGREE)),
    (0.0, 0.99, (Verdict.OPEN, Reason.JUDGES_DISAGREE)),  # right in other words, or the judge is wrong
    (1.0, 0.01, (Verdict.OPEN, Reason.JUDGES_DISAGREE)),
])
def test_the_automatic_judges_decide_only_where_they_agree(em, judge, expected):
    assert verdict(replace(ANSWER, exact_match=em, judge_with_reference=judge)) == expected


@pytest.mark.parametrize("claude", [Verdict.KNOWN, Verdict.UNKNOWN])
def test_claude_decides_wherever_it_reads(claude):
    for em, judge in [(1.0, 0.99), (0.0, 0.01), (0.0, 0.99)]:
        assert verdict(replace(ANSWER, exact_match=em, judge_with_reference=judge), claude) == (claude, Reason.CLAUDE)


def test_a_refusal_is_not_known_whatever_the_judges_say():
    """NQ-open 14.09: the judge said Yes to "I do not have specific information..." against "Justin Timberlake"."""
    refused = replace(ANSWER, reply="I do not have specific information about the lineup.", exact_match=0.0,
                      judge_with_reference=0.98)
    assert verdict(refused) == (Verdict.UNKNOWN, Reason.REFUSED)
    assert verdict(refused, Verdict.KNOWN) == (Verdict.KNOWN, Reason.CLAUDE)  # Claude still decides


def test_the_judge_without_the_reference_does_not_decide():
    """It shows how much the reference carries; the verdict is the one shown the reference."""
    assert verdict(replace(ANSWER, judge_without_reference=0.0)) == verdict(ANSWER)


def test_a_reasoned_reply_gives_its_answer_after_the_last_cue():
    reply = "The ball falls faster because gravity...\nSo the heavier one.\nAnswer: they land together"
    assert split_reasoning(reply) == ("The ball falls faster because gravity...\nSo the heavier one.", "they land together")
    assert split_reasoning("Answer: Paris") == (None, "Paris")


def test_a_reply_without_its_answer_line_gives_no_answer():
    """Out of tokens mid-solution: what it wrote is reasoning, never an answer to score."""
    assert split_reasoning("**1. Understand the goal:**\nThe question asks") == \
        ("**1. Understand the goal:**\nThe question asks", None)


@pytest.mark.parametrize("dressed, bare", [("<h2>Chaplin</h2>", "Chaplin"), ("**Paris**", "Paris"),
                                           ("`east`", "east"), ("<b>Ty</b> <i>Cobb</i>", "Ty Cobb"), ("Paris", "Paris")])
def test_the_markup_an_it_model_puts_on_an_answer_comes_off(dressed, bare):
    assert strip_markup(dressed) == bare


@pytest.mark.parametrize("last", ["Answer: east", "**Answer:** east", "**Answer: east**", "answer : east", "## Answer: `east`"])
def test_the_answer_line_is_found_through_the_markup_an_it_model_puts_on_it(last):
    assert split_reasoning(f"The current pushes north.\nSo the wind pushes east.\n{last}") == \
        ("The current pushes north.\nSo the wind pushes east.", "east")


def test_an_answer_line_survives_its_round_trip(tmp_path):
    path = answers_path(tmp_path, "bf16", "triviaqa")
    append_answers(path, [ANSWER, replace(ANSWER, id="tc_2", reasoning="because")])
    assert read_answers(path) == [ANSWER, replace(ANSWER, id="tc_2", reasoning="because")]
    assert path == tmp_path / "bf16" / "triviaqa.jsonl"


def test_a_restart_appends_and_knows_what_is_written(tmp_path):
    path = answers_path(tmp_path, "D4", "nq_open")
    assert written_ids(path) == set()
    append_answers(path, [replace(ANSWER, id="1")])
    append_answers(path, [replace(ANSWER, id="2"), replace(ANSWER, id="3")])
    assert written_ids(path) == {"1", "2", "3"}
    assert [a.id for a in read_answers(path)] == ["1", "2", "3"]


def test_the_frozen_corpus_survives_its_round_trip():
    frozen = FrozenCorpus("triviaqa", "0f7faf33", "google/gemma-4-E2B-it@3e22461f", "short-0shot",
                          kept=("1", "2"), excluded={"3": "unknown:judges_agree"}, tuning=("4",), unknown_share=("3",))
    assert FrozenCorpus.from_json(frozen.to_json()) == frozen
