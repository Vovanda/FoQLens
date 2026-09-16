"""The setups a corpus is asked in while its prompt is chosen, and what is read off their answers."""

from dataclasses import replace

import pytest

from foqlens.corpora import CORPORA, Row
from foqlens.generation import END_OF_TURN, FIRST_LINE
from foqlens.prompt_variants import (
    SETUPS, SHORT, WORKED, Variant, agreement, examples_for, is_refusal, setup_summary,
)
from foqlens.prompting import ASSISTANT, USER
from foqlens.selection import Answer, split_reasoning, tuning_sample

TRAIN = [Row(f"t{i}", f"train question {i}?", (f"answer {i}",)) for i in range(20)]
ANSWER = Answer(corpus="triviaqa", id="1", revision="rev", model="m", level="bf16", prompt="short-0", reply="Paris",
                answer="Paris", reasoning=None, exact_match=1.0, f1=1.0, tokens=2, stopped=True,
                judge_kind="Correct", judge_accepted=True, judge_reply="**Kind:** Correct\n**Accepted:** Yes")


def test_every_corpus_has_setups_and_every_setup_a_unique_name():
    assert set(SETUPS) == set(CORPORA)
    for setups in SETUPS.values():
        assert len({s.name for s in setups}) == len(setups)


@pytest.mark.parametrize("corpus", ["arc_challenge_closed", "arc_easy_closed", "hotpotqa"])
def test_the_corpora_that_reason_write_their_answer_out(corpus):
    """Volodya 14.09: ARC writes its solution, HotpotQA justifies its answer from the two passages."""
    for setup in SETUPS[corpus]:
        assert setup.written and setup.stop is END_OF_TURN and "Answer:" in setup.instruction


def test_a_short_setup_takes_the_first_line_and_a_written_one_the_last_answer_line():
    assert Variant("s", SHORT).stop is FIRST_LINE
    assert Variant("s", SHORT).extract("Paris\nIt is the capital.") == (None, "Paris")
    assert Variant("s", SHORT).extract("<h2>Chaplin</h2>") == (None, "Chaplin")
    assert Variant("w", "solve", written=True).extract("It falls.\nAnswer: down") == ("It falls.", "down")
    assert Variant("w", "solve", written=True).extract("**1. Goal:** find the") == ("**1. Goal:** find the", "")


def test_the_instruction_stands_before_every_question_the_examples_included():
    setup = Variant("s", SHORT, shots=1)
    turns = setup.messages(None, "Who wrote Hamlet?", ((None, "Capital of France?", "Paris"),))
    assert [t["role"] for t in turns] == [USER, ASSISTANT, USER]
    assert turns[0]["content"] == f"{SHORT}\n\nQuestion: Capital of France?"
    assert turns[2]["content"] == f"{SHORT}\n\nQuestion: Who wrote Hamlet?"
    assert Variant("bare", None).messages(None, "q") == [{"role": USER, "content": "Question: q"}]


def test_examples_come_from_the_train_split_and_the_same_seed_gives_the_same_ones():
    a = examples_for("triviaqa", Variant("s", SHORT, shots=2, shot_set=0), TRAIN, seed=0)
    assert a == examples_for("triviaqa", Variant("s", SHORT, shots=2, shot_set=0), TRAIN, seed=0)
    assert a != examples_for("triviaqa", Variant("s", SHORT, shots=2, shot_set=1), TRAIN, seed=0)
    assert all(q.startswith("train question") for _, q, _ in a)
    assert examples_for("triviaqa", Variant("s", SHORT), TRAIN, seed=0) == ()


def test_arc_examples_are_the_worked_solutions_each_ending_in_its_answer():
    sets = [examples_for("arc_challenge_closed", s, [], seed=0) for s in SETUPS["arc_challenge_closed"] if s.shots]
    assert len(sets) == 2 and sets[0] != sets[1]
    for _, question, reply in WORKED:
        reasoning, answer = split_reasoning(reply)
        assert reasoning and answer and "?" in question


def test_an_unanswerable_example_teaches_the_word_for_it():
    train = [Row("u", "q?", (), "passage")]
    (example,) = examples_for("squad_v2", Variant("p", SHORT, shots=1), train, seed=0)
    assert example == ("passage", "q?", "unanswerable")


def test_the_tuning_share_is_a_seeded_draw_with_a_floor():
    ids = [str(i) for i in range(1000)]
    assert len(tuning_sample(ids, 0)) == 50                        # 1% of 1000 is 10, the floor is 50
    assert len(tuning_sample([str(i) for i in range(10000)], 0)) == 100
    assert tuning_sample(ids, 0) == tuning_sample(ids, 0) != tuning_sample(ids, 1)
    assert tuning_sample(ids, 0) == sorted(tuning_sample(ids, 0), key=int)  # the corpus's own order
    assert len(tuning_sample(ids[:30], 0)) == 30


@pytest.mark.parametrize("text, refused", [
    ("I do not have the specific information to tell you", True),
    ("I'm not sure.", True),
    ("David Seville", False),
    ("The capital of France is **Paris**.", False),
])
def test_a_refusal_is_told_from_an_answer(text, refused):
    assert is_refusal(text) == refused


def test_a_setup_summary_counts_what_the_judges_said():
    rows = [ANSWER, replace(ANSWER, id="2", exact_match=0.0, f1=0.5, judge_kind="Wrong", judge_accepted=False,
                            reply="I do not know", stopped=False, tokens=6)]
    s = setup_summary(rows)
    assert (s["exact_match"], s["f1"], s["judged_right"], s["refusals"], s["stopped"], s["tokens"]) == \
        (0.5, 0.75, 0.5, 0.5, 0.5, 4.0)


def test_agreement_is_the_share_of_questions_every_setup_judges_alike():
    rows = [ANSWER, replace(ANSWER, prompt="bare"),
            replace(ANSWER, id="2"), replace(ANSWER, id="2", prompt="bare", judge_kind="Wrong", judge_accepted=False)]
    assert agreement(rows) == 0.5
