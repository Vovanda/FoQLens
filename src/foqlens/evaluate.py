"""Step 3 quality: how well the model answers, behind one interface.

A QualityMetric scores a batch of questions - one row of named values per question - and names the
value layouts are compared on (`primary`, higher is better). Evaluation, summaries and comparisons
read the values by name, so a new metric is a new class.

LetterChoice is MMLU multiple choice: the question, the options under A-D, then "Answer:"; the
log-probabilities of " A" .. " D" at the last real token are compared. Batches are right-padded, and
only the logits of the last real positions are computed (logits_to_keep).

Invariants:
- Invariant: the same prompts in the same batch give identical log-probabilities.
- Invariant (approximate, bf16): a prompt scored in a batch is within 0.25 of its score alone
  (measured 0.09-0.19); comparisons between layouts therefore always use the same batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from foqlens import model as fm

LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Question:
    domain: str
    prompt: str
    answer: int


class QualityMetric(Protocol):
    name: str
    primary: str  # the value layouts are compared on; higher is better

    def score(self, model, tokenizer, questions: list[Question]) -> list[dict[str, float]]:
        """One row of named values per question, in order."""
        ...


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


@dataclass(frozen=True)
class LetterChoice:
    """MMLU multiple choice: the log-probability of the right letter among A-D, and whether it is the top one.

    `picked` is the index of the letter that won, kept so that a run can tell a model that answers from
    one that leans on a letter: a lean shows as a skew in how often each index comes out on top.
    """

    ids: tuple[int, ...]
    name: str = "letter_choice"
    primary: str = "logprob"

    @classmethod
    def for_tokenizer(cls, tokenizer) -> LetterChoice:
        return cls(tuple(letter_ids(tokenizer)))

    def score(self, model, tokenizer, questions: list[Question]) -> list[dict[str, float]]:
        logprobs = letter_logprobs_batch(model, tokenizer, [q.prompt for q in questions], list(self.ids))
        return [{"accuracy": float(lp.argmax() == q.answer), "logprob": float(lp[q.answer]), "picked": float(lp.argmax())}
                for lp, q in zip(logprobs, questions)]
