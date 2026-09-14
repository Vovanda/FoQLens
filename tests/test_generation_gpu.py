"""The batched greedy loop against generate(): the same batch exactly, one prompt alone within bf16's reach."""

from pathlib import Path

import pytest
import torch

from foqlens import model as fm
from foqlens.extractive import first_line
from foqlens.generation import (
    answer_texts, end_ids, generate_answer, generate_answers, greedy_tokens, newline_ids, question_prompt,
)
from foqlens.io import read_jsonl
from foqlens.quant import Level

pytestmark = pytest.mark.gpu

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
DOMAINS = ("biology", "chemistry", "math", "physics")  # never the held-out topics
PER_DOMAIN = 8  # 32 prompts of different lengths: the batch the corpus runs use (QA_BATCH)
TOKENS = 24
# bf16 is not batch-invariant: token states move up to ~2% with the batch, and a near tie between two
# tokens can go the other way, after which the texts part. Most rows must still write the same text.
MIN_SAME_ALONE = 0.9


@pytest.fixture(scope="module")
def prompts() -> list[str]:
    return [question_prompt(q["text"]) for d in DOMAINS for q in read_jsonl(PROMPTS / f"{d}.jsonl", PER_DOMAIN)]


@pytest.fixture(autouse=True)
def bf16(e2b_sdpa):
    e2b_sdpa[2].set_all(Level.BF16)
    yield
    e2b_sdpa[2].set_all(Level.BF16)


def generate_batch(model, tokenizer, prompts: list[str]) -> list[str]:
    """The first line of what generate() itself writes on the same left-padded batch, run to the end."""
    enc = fm.encode_left(tokenizer, prompts, model.device)
    out = model.generate(**enc, max_new_tokens=TOKENS, do_sample=False)
    return [first_line(t) for t in answer_texts(tokenizer, out[:, enc["input_ids"].shape[1]:], end_ids(model))]


@pytest.mark.parametrize("level", [Level.BF16, Level.D4])
def test_the_loop_writes_the_first_line_generate_writes_on_the_same_batch(e2b_sdpa, prompts, level):
    """The same calls in the same batch: positions, mask and cache as generate() builds them; a row stops at its line."""
    model, tokenizer, ctl = e2b_sdpa
    ctl.set_all(level)
    ours = [t.strip() for t in generate_answers(model, tokenizer, prompts, TOKENS)]
    assert ours == generate_batch(model, tokenizer, prompts)


def test_a_prompt_in_the_batch_writes_what_it_writes_alone(e2b_sdpa, prompts):
    model, tokenizer, _ = e2b_sdpa
    batched = [t.strip() for t in generate_answers(model, tokenizer, prompts, TOKENS)]
    alone = [first_line(generate_answer(model, tokenizer, p, TOKENS)) for p in prompts]
    parted = [(i, a, b) for i, (a, b) in enumerate(zip(alone, batched)) if a != b]
    for i, a, b in parted:
        print(f"prompt {i}: alone {a!r} | batch {b!r}")
    print(f"same text: {len(prompts) - len(parted)}/{len(prompts)}")
    assert 1 - len(parted) / len(prompts) >= MIN_SAME_ALONE


def test_the_same_batch_twice_writes_the_same_tokens(e2b_sdpa, prompts):
    model, tokenizer, _ = e2b_sdpa
    enc = fm.encode_left(tokenizer, prompts, model.device)
    stop = torch.tensor(end_ids(model) + newline_ids(tokenizer), device=model.device)
    first = greedy_tokens(model, enc["input_ids"], enc["attention_mask"], TOKENS, stop)
    assert torch.equal(first, greedy_tokens(model, enc["input_ids"], enc["attention_mask"], TOKENS, stop))
