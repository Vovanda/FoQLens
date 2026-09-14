"""Matching a free answer to the options it was never shown - the second step of the two-step metric.

The first step asks the model the bare question under a precision layout and keeps what it writes.
This module is the second: given that text and the four options, which option says the same thing.

The matcher is deliberately separate from the layout under test. It always runs at bf16 and sees the
same prompt for every layout, so whatever lean it has is a constant of the measurement rather than
something a layout can exploit - which is the whole point, since the lean toward a letter is what the
one-step metric turned out to be measuring.

Invariant: the matcher never sees which layout produced an answer, and never reads the right option.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from foqlens.evaluate import LETTERS, letter_logprobs_batch
from foqlens.quant import Level

ANSWER_TOKENS = 32  # a free answer to a quiz question is a short phrase; beyond this the model explains itself
ANSWER_BATCH = 32   # a decoding step costs the same for 1 row and for 32 (14.09 profile); short prompts, small cache


class AnswerMatcher(Protocol):
    name: str

    def match(self, answers: list[str], questions: list[str], options: list[list[str]]) -> np.ndarray:
        """The index of the option each answer means, one per answer."""
        ...


def match_prompt(question: str, answer: str, options: list[str]) -> str:
    listed = "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, options, strict=True))
    return (f"Question: {question}\n"
            f"Someone answered: {answer}\n"
            f"Which option below says the same thing as that answer?\n"
            f"{listed}\n"
            f"Option:")


@dataclass(frozen=True)
class LetterMatcher:
    """The model itself reads the answer and points at an option, always at bf16."""

    ids: tuple[int, ...]
    model: object
    tokenizer: object
    ctl: object
    batch_size: int = 16
    name: str = "letter_matcher"

    def match(self, answers: list[str], questions: list[str], options: list[list[str]]) -> np.ndarray:
        prompts = [match_prompt(q, a, o) for q, a, o in zip(questions, answers, options, strict=True)]
        self.ctl.set_all(Level.BF16)
        picked = np.empty(len(prompts), dtype=int)
        for start in range(0, len(prompts), self.batch_size):
            chunk = slice(start, start + self.batch_size)
            logprobs = letter_logprobs_batch(self.model, self.tokenizer, prompts[chunk], list(self.ids))
            picked[chunk] = logprobs.argmax(axis=1)
        return picked


def first_answer(text: str) -> str:
    """The answer itself, without the next question the base model goes on to invent.

    These are base checkpoints with no chat template, so after answering they simply keep writing -
    usually "\\n\\nQuestion: ..." of their own. An MMLU answer is one line, so the first line is the
    answer and everything after it belongs to a question nobody asked.
    """
    return text.split("\n", 1)[0].strip()


def free_answers(model, tokenizer, prompts: list[str], max_new_tokens: int,
                 log: object = None, batch_size: int = ANSWER_BATCH) -> list[str]:
    """What the model writes for each prompt, under whatever layout is currently set, batch_size at a time."""
    from foqlens.generation import generate_answers

    out = []
    for start in range(0, len(prompts), batch_size):
        written = generate_answers(model, tokenizer, prompts[start:start + batch_size], max_new_tokens)
        out += [first_answer(text) for text in written]
        if log is not None:
            log(f"  generated {len(out)}/{len(prompts)}")
    return out
