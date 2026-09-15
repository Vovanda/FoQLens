"""The corpus selection's records and rule: an answer per line, three judges, a frozen file of numbers."""

from dataclasses import replace

import pytest

from foqlens.extractive import NO_ANSWER
from foqlens.io import answers_path, append_answers, append_verdicts, read_answers, read_verdicts, written_ids
from foqlens.selection import (Answer, ClaudeVerdict, FrozenCorpus, Reading, Reason, Turn, Verdict, freeze,
                               split_reasoning, strip_markup, turn, two_way_choice, verdict)

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


def test_a_reply_without_an_answer_and_an_answer_the_passage_lacks_are_unknown_unread():
    """ARC: a solution cut at the token limit; SQuAD v2: a span given where the passage has no answer."""
    assert verdict(replace(ANSWER, answer=None, judge_with_reference=0.99)) == (Verdict.UNKNOWN, Reason.NO_ANSWER)
    spanned = replace(ANSWER, answer="Secondary education", exact_match=0.0, judge_with_reference=0.99)
    assert verdict(spanned, answerable=False) == (Verdict.UNKNOWN, Reason.UNANSWERABLE)
    abstained = replace(ANSWER, answer="unanswerable", exact_match=1.0)
    assert verdict(abstained, answerable=False) == (Verdict.KNOWN, Reason.JUDGES_AGREE)
    assert turn(spanned, answerable=False) is None and turn(abstained, answerable=False) == Turn.AGREED_YES


@pytest.mark.parametrize("em, f1, judge, expected", [
    (0.0, 0.0, 0.99, Turn.DISAGREE),
    (1.0, 1.0, 0.01, Turn.DISAGREE),
    (0.0, 0.8, 0.01, Turn.DOUBTFUL_NO),   # "Gary Johnson" against "Gary Earl Johnson"
    (0.0, 0.0, 0.2, Turn.DOUBTFUL_NO),    # the judge is not sure of its No
    (0.0, 0.0, 0.01, Turn.SURE_NO),
    (1.0, 1.0, 0.99, Turn.AGREED_YES),
])
def test_an_answer_waits_in_the_turn_its_judges_leave_it_in(em, f1, judge, expected):
    assert turn(replace(ANSWER, exact_match=em, f1=f1, judge_with_reference=judge)) == expected


def test_every_answer_either_waits_in_one_turn_or_is_decided_unread():
    for em, judge, reply, answer, answerable in [(1.0, 0.99, "Paris", "Paris", True), (0.0, 0.2, "x", "x", True),
                                                 (0.0, 0.99, "I don't know", "I don't know", True),
                                                 (0.0, 0.5, "", None, True), (0.0, 0.99, "x", "x", False)]:
        a = replace(ANSWER, exact_match=em, judge_with_reference=judge, reply=reply, answer=answer)
        waits = turn(a, answerable) is not None
        decided = verdict(a, answerable=answerable)[1] in (Reason.NO_ANSWER, Reason.UNANSWERABLE, Reason.REFUSED)
        assert waits != decided


@pytest.mark.parametrize("reading, known", [(Reading.RIGHT, Verdict.KNOWN), (Reading.OTHER_WORDS, Verdict.KNOWN),
                                            (Reading.WRONG, Verdict.UNKNOWN), (Reading.NO_ANSWER, Verdict.UNKNOWN)])
def test_claude_knows_two_ways_of_knowing_and_two_of_not(reading, known):
    assert ClaudeVerdict("triviaqa", "tc_1", "bf16", reading, "2026-09-15").verdict == known


def test_claude_verdicts_survive_their_round_trip_and_a_second_reading_wins(tmp_path):
    path = answers_path(tmp_path, "bf16", "triviaqa")
    first = ClaudeVerdict("triviaqa", "tc_1", "bf16", Reading.WRONG, "2026-09-15", "too general")
    append_verdicts(path, [first, replace(first, id="tc_2", reading=Reading.RIGHT, note="")])
    append_verdicts(path, [replace(first, reading=Reading.OTHER_WORDS)])
    read = read_verdicts(path)
    assert read["tc_1"] == replace(first, reading=Reading.OTHER_WORDS)
    assert read["tc_2"].reading is Reading.RIGHT
    assert ClaudeVerdict.from_json(first.to_json()) == first


@pytest.mark.parametrize("question, references, two_way", [
    ("Who is older, Annie Morton or Terry Richardson?", ("Terry Richardson",), True),
    ("Which writer was from England, Henry Roth or Robert Erskine Childers?", ("Robert Erskine Childers DSC",), True),
    ("Can BSkyB veto the presence of channels on their EPG?", ("no", "no"), True),
    ("is greenland part of europe or north america", ("North America",), True),
    ("What river flows through the Grand Canyon?", ("Colorado",), False),
    ("Macbeth belonged to which royal house or dynasty?", ("House of Dunkeld",), False),  # "or" names no options
    ("What mythical god has a hammer called Mjolnir, or Miolnir?", ("Thor",), False),
])
def test_a_guess_between_two_named_options_or_yes_no_is_marked(question, references, two_way):
    assert two_way_choice(question, references) is two_way


def test_the_frozen_corpus_survives_its_round_trip():
    frozen = FrozenCorpus("triviaqa", "0f7faf33", "google/gemma-4-E2B-it@3e22461f", "short-0shot",
                          kept=("1", "2"), excluded={"3": "unknown:judges_agree"}, tuning=("4",), unknown_share=("3",),
                          two_way=("2",))
    assert FrozenCorpus.from_json(frozen.to_json()) == frozen


def answered(i, em, judge=None, answer="x"):
    """An answer to question `i` the two automatic judges agree on unless `judge` says otherwise."""
    return replace(ANSWER, id=i, answer=answer, exact_match=em, judge_with_reference=em if judge is None else judge)


QUESTIONS = {str(i): (f"q{i}", ("a",)) for i in range(40)}


def test_a_frozen_corpus_keeps_what_the_rule_calls_known_in_the_corpus_order():
    answers = [answered(str(i), 1.0 if i % 2 else 0.0) for i in reversed(range(40))]
    frozen = freeze(answers, {"2": Verdict.KNOWN, "3": Verdict.UNKNOWN}, QUESTIONS, (), seed=0)
    kept = [str(i) for i in range(40) if i % 2 and i != 3] + ["2"]
    assert set(frozen.kept) == set(kept) and list(frozen.kept) == sorted(frozen.kept, key=int)
    assert frozen.excluded["3"] == "unknown:claude" and frozen.excluded["4"] == "unknown:judges_agree"
    assert set(frozen.kept) | set(frozen.excluded) == set(QUESTIONS)


def test_a_question_without_an_answer_is_kept_where_the_model_says_so():
    questions = {**QUESTIONS, "40": ("q40", ())}
    abstains = replace(ANSWER, id="40", answer=NO_ANSWER, exact_match=1.0, judge_with_reference=0.99)
    frozen = freeze([answered(i, 1.0) for i in QUESTIONS] + [abstains], {}, questions, (), seed=0)
    assert "40" in frozen.kept


def test_an_open_answer_stops_the_freeze():
    with pytest.raises(ValueError):
        freeze([answered("1", 0.0, judge=0.99)], {}, QUESTIONS, (), seed=0)


def test_the_unknown_share_is_a_tenth_of_the_stage2_set_drawn_from_the_excluded_by_seed():
    answers = [answered(str(i), 1.0 if i < 27 else 0.0) for i in range(40)]
    one, again = (freeze(answers, {}, QUESTIONS, ("99",), seed=7) for _ in range(2))
    assert one.unknown_share == again.unknown_share and len(one.unknown_share) == 3  # 27 known = 90%
    assert set(one.unknown_share) <= set(one.excluded) and one.tuning == ("99",)
    assert one.unknown_share != freeze(answers, {}, QUESTIONS, (), seed=8).unknown_share


def test_a_kept_guess_between_two_is_marked():
    questions = {**QUESTIONS, "40": ("Is Paris in France?", ("yes",))}
    frozen = freeze([answered(i, 1.0) for i in questions], {}, questions, (), seed=0)
    assert frozen.two_way == ("40",)


def test_answers_from_two_prompts_are_not_one_corpus():
    with pytest.raises(ValueError):
        freeze([answered("1", 1.0), replace(answered("2", 1.0), prompt="other")], {}, QUESTIONS, (), seed=0)
