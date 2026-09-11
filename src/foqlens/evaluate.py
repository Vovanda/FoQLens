"""Step 3 quality: MMLU multiple choice, scored by the letter the model puts most probability on.

The standard format: the question, the options under A-D, then "Answer:"; the log-probabilities
of " A" .. " D" at the last real token are compared. Batches are right-padded, and only the
logits of the last real positions are computed (logits_to_keep).

Invariants:
- Invariant: the same prompts in the same batch give identical log-probabilities.
- Invariant (approximate, bf16): a prompt scored in a batch is within 0.25 of its score alone
  (measured 0.09-0.19); comparisons between layouts therefore always use the same batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from foqlens import model as fm

LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Question:
    domain: str
    prompt: str
    answer: int


def mc_prompt(question: str, choices: list[str]) -> str:
    assert len(choices) == len(LETTERS), choices
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices))
    return f"{question}\n{options}\nAnswer:"


def letter_ids(tokenizer) -> list[int]:
    ids = [tokenizer(" " + letter, add_special_tokens=False).input_ids for letter in LETTERS]
    assert all(len(i) == 1 for i in ids), ids
    return [i[0] for i in ids]


@torch.no_grad()
def letter_logprobs_batch(model, tokenizer, prompts: list[str], ids: list[int]) -> np.ndarray:
    """Log-probabilities over the four answer letters, renormalized among them: [batch, 4]."""
    enc = fm.encode(tokenizer, prompts, model.device)
    last = enc["attention_mask"].sum(dim=1) - 1
    keep = torch.unique(last)  # sorted distinct last positions: logits for these only
    logits = model(**enc, logits_to_keep=keep).logits  # [batch, len(keep), vocab]
    picked = logits[torch.arange(len(prompts), device=logits.device), torch.searchsorted(keep, last)]
    return torch.log_softmax(picked[:, ids].float(), dim=-1).cpu().numpy()


def letter_logprobs(model, tokenizer, prompt: str, ids: list[int]) -> np.ndarray:
    return letter_logprobs_batch(model, tokenizer, [prompt], ids)[0]
