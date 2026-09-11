"""Free-text answers of the model under a precision layout: the reference points to read by eye.

One prompt at a time: batched generation would need left padding, and Gemma 4 derives positions
from arange(seq), so left padding would shift them (see model.load).

Invariant: greedy decoding - the same layout and prompt give the same text.
"""

from __future__ import annotations

import torch

from foqlens import model as fm


def question_prompt(question: str) -> str:
    """A plain question for a free answer, without the options."""
    return f"Question: {question}\nAnswer:"


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = fm.encode(tokenizer, [prompt], model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
