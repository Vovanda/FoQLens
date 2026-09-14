"""The model as a judge: Yes or No at bf16, blind to who answered, with the references and without."""

import inspect

import numpy as np
import pytest

from foqlens import judging
from foqlens.extractive import NO_ANSWER
from foqlens.judging import MAX_REFERENCES, ModelJudge, judge_prompt, judge_question, verdict_ids
from foqlens.prompting import ChatFormat


def test_the_prompt_with_references_asks_to_compare_and_ends_where_the_verdict_goes():
    prompt = judge_prompt("Who wrote Hamlet?", "Shakespeare", ["William Shakespeare", "Shakespeare"])
    assert "Reference answer: William Shakespeare; Shakespeare\n" in prompt
    assert "Proposed answer: Shakespeare\n" in prompt
    assert "mean the same as the reference answer" in prompt
    assert prompt.endswith("\nAnswer:")  # " Yes" or " No" comes next


def test_the_prompt_without_references_asks_whether_it_is_correct():
    prompt = judge_prompt("q", "a", None)
    assert "Reference" not in prompt and "Is the proposed answer correct?" in prompt


class Template:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "<turn>" + messages[-1]["content"] + "<model>"


def test_a_chat_judge_opens_its_reply_with_the_cue_so_the_next_word_is_the_verdict():
    """Left to itself an -it model writes "**Answer: Yes**", and its first token is no verdict at all."""
    assert judge_prompt("q", "a", ["a"], ChatFormat(Template())).endswith("<model>Answer:")


def test_a_question_without_an_answer_shows_that_as_the_reference():
    """SQuAD v2: no reference answers means the right response is to say there is none."""
    assert f"Reference answer: {NO_ANSWER}\n" in judge_prompt("q", "a", [])


def test_only_the_first_references_are_shown():
    prompt = judge_prompt("q", "a", [f"alias{i}" for i in range(MAX_REFERENCES + 3)])
    assert f"alias{MAX_REFERENCES - 1}" in prompt and f"alias{MAX_REFERENCES}" not in prompt


def test_the_question_can_carry_nothing_about_who_answered():
    assert list(inspect.signature(judge_question).parameters) == ["question", "answer", "references"]


class OneToken:
    def __init__(self, width):
        self.width, self.seen = width, []

    def __call__(self, text, add_special_tokens):
        self.seen.append(text)
        return type("Enc", (), {"input_ids": list(range(self.width))})()


def test_the_verdicts_are_single_tokens_or_refused():
    assert verdict_ids(OneToken(1)) == (0, 0)
    with pytest.raises(ValueError):
        verdict_ids(OneToken(2))


def test_the_verdicts_are_read_as_the_word_after_the_cue():
    tok = OneToken(1)
    verdict_ids(tok)
    assert tok.seen == [" Yes", " No"]


class Controller:
    def __init__(self):
        self.levels = None

    def set_all(self, level):
        self.levels = level


def test_the_judge_reads_at_bf16_in_batches_and_keeps_the_order(monkeypatch):
    seen = []
    p = {"a1": 0.9, "a2": 0.2, "a3": 0.6}

    def logprobs(model, tok, prompts, ids):
        seen.append(prompts)
        yes = np.array([next(v for k, v in p.items() if f"Proposed answer: {k}\n" in pr) for pr in prompts])
        return np.log(np.stack([yes, 1 - yes], axis=1))

    monkeypatch.setattr(judging, "letter_logprobs_batch", logprobs)
    ctl = Controller()
    judge = ModelJudge((1, 2), None, None, ctl, batch_size=2)
    got = judge.p_yes(["q1", "q2", "q3"], ["a1", "a2", "a3"], [["r1"], ["r2"], []])
    assert np.allclose(got, [0.9, 0.2, 0.6])
    assert [len(b) for b in seen] == [2, 1]
    assert all("Reference answer" in pr for b in seen for pr in b)
    assert ctl.levels is not None and ctl.levels.name == "BF16"
    judge.p_yes(["q1"], ["a1"], None)
    assert "Reference" not in seen[-1][0]
