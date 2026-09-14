"""Gemma 4 E2B-it, the model the corpus is selected on: its chat prompts through the batched loop, and it as a judge."""

import numpy as np
import pytest

from foqlens import model as fm
from foqlens.extractive import first_line, qa_prompt
from foqlens.generation import END_OF_TURN, FIRST_LINE, answer_texts, end_ids, generate_answers
from foqlens.judging import ModelJudge
from foqlens.quant import Level

pytestmark = pytest.mark.gpu

QUESTIONS = ["What is the capital of France?", "How many legs does a spider have?", "What is the chemical symbol for gold?",
             "Who wrote the play Hamlet?", "What planet is known as the Red Planet?", "What is the boiling point of water in Celsius?"]
RIGHT = ["Paris", "Eight", "Au", "William Shakespeare", "Mars", "100 degrees"]
WRONG = ["Berlin", "Six", "Ag", "Charles Dickens", "Venus", "50 degrees"]
REFERENCES = [["Paris"], ["8", "eight"], ["Au"], ["Shakespeare"], ["Mars"], ["100", "100 °C"]]
TOKENS = 48  # a reply of a sentence or two, the length an -it model gives without being asked to be brief


@pytest.fixture(scope="module")
def chat(e2b_it_sdpa):
    model, tokenizer, ctl = e2b_it_sdpa
    ctl.set_all(Level.BF16)
    fmt = fm.prompt_format(fm.E2B_IT, tokenizer)
    return model, tokenizer, ctl, fmt, [qa_prompt(None, q, (), fmt) for q in QUESTIONS]


def generate_batch(model, tokenizer, prompts):
    enc = fm.encode_left(tokenizer, prompts, model.device)
    out = model.generate(**enc, max_new_tokens=TOKENS, do_sample=False)
    return answer_texts(tokenizer, out[:, enc["input_ids"].shape[1]:], end_ids(model))


def test_the_loop_writes_what_generate_writes_on_the_same_chat_batch(chat):
    model, tokenizer, _, _, prompts = chat
    full = generate_batch(model, tokenizer, prompts)
    for text in full:
        print(repr(text))
    assert generate_answers(model, tokenizer, prompts, TOKENS, END_OF_TURN) == full
    assert [t.strip() for t in generate_answers(model, tokenizer, prompts, TOKENS, FIRST_LINE)] == [first_line(t) for t in full]


@pytest.mark.parametrize("with_references", [True, False])
def test_right_answers_get_yes_and_wrong_ones_no(chat, with_references):
    model, tokenizer, ctl, fmt, _ = chat
    judge = ModelJudge.build(model, tokenizer, ctl, fmt)
    refs = REFERENCES if with_references else None
    right, wrong = judge.p_yes(QUESTIONS, RIGHT, refs), judge.p_yes(QUESTIONS, WRONG, refs)
    print(f"references={with_references}: right {np.round(right, 3)}, wrong {np.round(wrong, 3)}")
    assert (wrong < 0.5).all() and (right > wrong).all()
    # A right answer worded as its reference is a plain Yes; a paraphrase ("100 degrees" against "100 °C")
    # may be a near tie - such answers go to Claude, the third judge.
    literal = np.array([a.lower() in {r.lower() for r in refs} for a, refs in zip(RIGHT, REFERENCES)])
    assert (right[literal] > 0.5).all()


def test_the_judge_reads_at_bf16_whatever_layout_was_set(chat):
    model, tokenizer, ctl, fmt, _ = chat
    judge = ModelJudge.build(model, tokenizer, ctl, fmt)
    at_bf16 = judge.p_yes(QUESTIONS, RIGHT, REFERENCES)
    ctl.set_all(Level.D4)
    assert np.array_equal(judge.p_yes(QUESTIONS, RIGHT, REFERENCES), at_bf16)
