"""Gemma 4 E2B-it, the model the corpus is selected on: its chat prompts through the batched loop, and it as a judge."""

import pytest

from foqlens import model as fm
from foqlens.attention import SPLIT
from foqlens.extractive import first_line, qa_prompt
from foqlens.generation import END_OF_TURN, FIRST_LINE, answer_texts, end_ids, generate_answers
from foqlens.graph_decode import StaticDecoder
from foqlens.judging import GARBAGE, ModelJudge
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


# Garbage as D2 wrote it (E001): symbols and repeated fragments with no answer in them.
GARBAGE_ANSWERS = ["... | ** | ** ** **", r"$\text{** $\text{** $\text{", "(17. 17.** | ** **("]


def judge_of(model, tokenizer, ctl, fmt):
    return ModelJudge(model, tokenizer, ctl, fmt, StaticDecoder(attention=SPLIT))


def test_right_answers_are_accepted_and_wrong_ones_are_not(chat):
    model, tokenizer, ctl, fmt, _ = chat
    judge = judge_of(model, tokenizer, ctl, fmt)
    right, wrong = judge.verdicts(QUESTIONS, RIGHT, REFERENCES), judge.verdicts(QUESTIONS, WRONG, REFERENCES)
    print([v.kind for v in right], [v.kind for v in wrong])
    assert not any(v.accepted for v in wrong)
    # A right answer worded as its reference is plainly accepted; a paraphrase ("100 degrees" against "100 °C")
    # may be graded down - such answers go to Claude, the third judge.
    literal = [a.lower() in {r.lower() for r in refs} for a, refs in zip(RIGHT, REFERENCES)]
    assert all(v.accepted for v, lit in zip(right, literal) if lit)


def test_garbage_is_called_garbage_and_never_accepted(chat):
    model, tokenizer, ctl, fmt, _ = chat
    got = judge_of(model, tokenizer, ctl, fmt).verdicts(QUESTIONS[:3], GARBAGE_ANSWERS, REFERENCES[:3])
    print([(v.kind, v.reply[-80:]) for v in got])
    assert not any(v.accepted for v in got) and all(v.kind == GARBAGE for v in got)


def test_the_judge_reads_at_bf16_whatever_layout_was_set(chat):
    model, tokenizer, ctl, fmt, _ = chat
    judge = judge_of(model, tokenizer, ctl, fmt)
    at_bf16 = judge.verdicts(QUESTIONS, RIGHT, REFERENCES)
    ctl.set_all(Level.D4)
    assert judge.verdicts(QUESTIONS, RIGHT, REFERENCES) == at_bf16
