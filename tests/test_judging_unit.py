"""The model as a judge: blind to who answered, reasoning before two closing lines, empty answers never asked."""

import inspect

import pytest

from foqlens import judging
from foqlens.extractive import NO_ANSWER
from foqlens.generation import Reply
from foqlens.judging import (ACCEPTED, EXAM, GARBAGE, GRADES, MAX_REFERENCES, NOT_READ, ModelJudge, NotJudged, Verdict,
                             judge_prompt, judge_question, read_verdict)
from foqlens.prompting import ChatFormat

CORRECT = "Correct answers (any one is enough): "


def test_the_question_shows_the_exam_the_references_and_the_fenced_answer_and_ends_with_the_closing_lines():
    prompt = judge_question("Who wrote Hamlet?", "Shakespeare", ["William Shakespeare", "Shakespeare"])
    assert prompt.startswith(EXAM) and "do not answer the question yourself" in prompt
    assert "Question: Who wrote Hamlet?\n" in prompt
    assert f"{CORRECT}William Shakespeare | Shakespeare\n" in prompt
    assert "Examinee's answer:\n---\nShakespeare\n---\n" in prompt
    assert prompt.endswith("**Kind:** Garbage|Noise|Wrong|Related|Partial|Nearly|Correct\n**Accepted:** Yes|No")


def test_the_kinds_go_from_the_general_to_the_particular_and_only_the_top_two_are_accepted():
    prompt = judge_question("q", "a", ["r"])
    lines = [f"{word} - {meaning}" for word, meaning in GRADES]
    assert [prompt.index(line) for line in lines] == sorted(prompt.index(line) for line in lines)
    assert [word for word, _ in GRADES][:2] == ["Garbage", "Noise"] and ACCEPTED == {"Nearly", "Correct"}
    assert "accepted only if its kind is Nearly or Correct." in prompt


def test_an_empty_answer_stays_between_its_fences():
    assert "\n---\n\n---\n" in judge_question("q", "", ["r"])


class Template:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "<turn>" + messages[-1]["content"] + "<model>"


def test_a_chat_judge_is_asked_in_the_models_own_form_and_its_reply_is_left_to_it():
    assert judge_prompt("q", "a", ["a"], ChatFormat(Template())).endswith("**Accepted:** Yes|No<model>")


def test_a_question_without_an_answer_shows_that_as_the_correct_answer():
    """SQuAD v2: no reference answers means the right response is to say there is none."""
    assert f"{CORRECT}{NO_ANSWER}\n" in judge_question("q", "a", [])


def test_only_the_first_references_are_shown_and_a_repeated_one_once():
    prompt = judge_question("q", "a", ["Rhine", "Rhine", "the Rhine", *[f"alias{i}" for i in range(MAX_REFERENCES)]])
    assert f"{CORRECT}Rhine | the Rhine | alias0 | alias1 | alias2\n" in prompt


def test_the_question_can_carry_nothing_about_who_answered():
    assert list(inspect.signature(judge_question).parameters) == ["question", "answer", "references"]


@pytest.mark.parametrize("reply, kind, accepted", [
    ("It is the capital.\n**Kind:** Correct\n**Accepted:** Yes", "Correct", True),
    ("Empty.\n**Kind:** Garbage\n**Accepted:** No", "Garbage", False),
    # the reasoning may quote the format: the last lines count
    ("I must end with **Kind:** Garbage|...\nIt names the city.\n**Kind:** Nearly\n**Accepted:** yes", "Nearly", True),
    ("It is right.", NOT_READ, False),
    ("**Kind:** Excellent\n**Accepted:** Yes", NOT_READ, False),
    ("**Kind:** Correct", NOT_READ, False),
])
def test_only_the_two_closing_lines_are_read(reply, kind, accepted):
    assert read_verdict(reply) == Verdict(kind, accepted, reply)


class Controller:
    def __init__(self):
        self.levels = None

    def set_all(self, level):
        self.levels = level


def test_the_judge_reads_at_bf16_asks_shortest_first_and_keeps_the_answers_order(monkeypatch):
    batches = []

    def replies(model, tok, prompts, max_new_tokens, stop, decoder):
        batches.append(prompts)
        return [Reply("**Kind:** Correct\n**Accepted:** Yes" if "Paris" in p else "**Kind:** Wrong\n**Accepted:** No", 9, True)
                for p in prompts]

    monkeypatch.setattr(judging, "generate_replies", replies)
    ctl = Controller()
    judge = ModelJudge(None, None, ctl, batch_size=2)
    got = judge.verdicts(["Capital of France?", "Capital of Germany, the long one?", "q3"], ["Paris", "Munich", "Paris"],
                         [["Paris"], ["Berlin"], ["Paris"]])
    assert [(v.kind, v.accepted) for v in got] == [("Correct", True), ("Wrong", False), ("Correct", True)]
    assert [len(b) for b in batches] == [2, 1]
    assert len(batches[0][0]) <= len(batches[0][1]) <= len(batches[1][0])
    assert ctl.levels is not None and ctl.levels.name == "BF16"


def test_an_empty_answer_is_garbage_without_being_asked(monkeypatch):
    asked = []
    monkeypatch.setattr(judging, "generate_replies",
                        lambda model, tok, prompts, n, stop, decoder: asked.extend(prompts) or
                        [Reply("**Kind:** Correct\n**Accepted:** Yes", 9, True) for _ in prompts])
    got = ModelJudge(None, None, Controller()).verdicts(["q1", "q2", "q3"], ["", " \n\t", "Paris"], [["r"], ["r"], ["Paris"]])
    assert got[:2] == [Verdict(GARBAGE, False, "")] * 2 and got[2].accepted
    assert len(asked) == 1


def test_a_reply_the_limit_cut_is_asked_once_more_with_the_retry_limit(monkeypatch):
    calls = []

    def replies(model, tok, prompts, max_new_tokens, stop, decoder):
        calls.append((len(prompts), max_new_tokens))
        if max_new_tokens == 10:  # the first pass: the long one is cut at the limit, the short one stops unread
            return [Reply("I work it through", 10, False) if "long" in p else Reply("No closing lines.", 3, True)
                    for p in prompts]
        return [Reply("Worked through.\n**Kind:** Correct\n**Accepted:** Yes", 40, True) for _ in prompts]

    monkeypatch.setattr(judging, "generate_replies", replies)
    got = ModelJudge(None, None, Controller(), max_new_tokens=10, retry_tokens=100).verdicts(
        ["a long question", "short"], ["a1", "a2"], [["r"], ["r"]])
    assert calls == [(2, 10), (1, 100)]
    assert (got[0].kind, got[0].accepted) == ("Correct", True) and got[1].kind == NOT_READ


def test_a_model_that_cannot_judge_leaves_its_answers_unjudged():
    assert NotJudged().verdicts(["q1", "q2"], ["a1", "a2"], [["r"], ["r"]]) == [None, None]
