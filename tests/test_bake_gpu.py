"""A uniform level baked into the weights of E2B-it answers as the same level read by unpacking its slices.

A file of its own: baking cannot be undone, so it loads its own model rather than the shared fixture.
"""

import pytest
import torch

from foqlens import model as fm
from foqlens.extractive import qa_prompt
from foqlens.generation import END_OF_TURN, generate_replies
from foqlens.graph_decode import STATIC
from foqlens.precision import install
from foqlens.quant import Level

pytestmark = pytest.mark.gpu

QUESTIONS = ["What is the capital of France?", "How many legs does a spider have?", "What is the chemical symbol for gold?",
             "Who wrote the play Hamlet?", "What planet is known as the Red Planet?", "What is the boiling point of water in Celsius?"]
TOKENS = 48


@pytest.fixture(scope="module")
def e2b_it():
    model, tokenizer = fm.load(fm.E2B_IT, attn_implementation="sdpa")
    yield model, tokenizer, install(model)
    del model
    torch.cuda.empty_cache()


def test_a_baked_level_answers_bit_for_bit_as_unpacking_at_less_memory(e2b_it):
    model, tokenizer, ctl = e2b_it
    fmt = fm.prompt_format(fm.E2B_IT, tokenizer)
    prompts = [qa_prompt(None, q, (), fmt) for q in QUESTIONS]
    ctl.set_all(Level.D4)
    unpacked = generate_replies(model, tokenizer, prompts, TOKENS, END_OF_TURN, STATIC)
    before = torch.cuda.memory_allocated()
    ctl.bake(Level.D4)
    assert torch.cuda.memory_allocated() < before  # the bf16 weight and the sliced copy leave, one weight stays
    ctl.set_all(Level.D4)
    assert generate_replies(model, tokenizer, prompts, TOKENS, END_OF_TURN, STATIC) == unpacked
    with pytest.raises(ValueError):
        ctl.set_all(Level.BF16)
