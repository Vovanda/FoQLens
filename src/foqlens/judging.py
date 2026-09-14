"""The full model as a judge of free answers: shown a question and an answer, it says Yes or No.

The second of the corpus's three judges (docs/corpus.md): exact match and F1 miss an answer that is
right in other words, and Claude reads what the first two leave open. The model judges in one forward
pass, without generating: its reply is opened with the cue "Answer:" and the log-probabilities of
" Yes" and " No" as the next word are renormalized between the two, so a model that would go on to
write something else still gives a verdict.

It asks in two forms: with the dataset's reference answers (does the answer mean the same as them) and
without (is it right, as far as the model knows). The first is the verdict; the second shows how much
of it the reference carries. The question is rendered in the model's own form (foqlens.prompting).

Invariant: the judge never sees who wrote an answer - its question holds the question, the answer and,
in one form, the references, and nothing else.
Invariant: the judge reads at bf16 whatever layout is under test, and leaves the model at bf16.
Invariant: the same prompts in the same batch give the same verdicts (evaluate.letter_logprobs_batch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foqlens.evaluate import letter_logprobs_batch
from foqlens.extractive import NO_ANSWER
from foqlens.prompting import PLAIN, USER, PromptFormat
from foqlens.quant import Level

VERDICTS = (" Yes", " No")  # the word after the cue "Answer:" (prompting.PromptFormat.cue)
# TriviaQA lists up to dozens of aliases per answer; the first few say what the answer is, the rest
# only lengthen the prompt.
MAX_REFERENCES = 5


def judge_question(question: str, answer: str, references: list[str] | None) -> str:
    """What the judge is asked; `references=None` asks without them. An empty list means there is no answer.

    With the reference the judge compares, it does not solve: asked "is it correct" an -it model judged
    the question itself and said No to "Princess Jessica" against the reference "Princess Jessica"
    (2026-09-14, HotpotQA).
    """
    if references is None:
        return (f"Question: {question}\nProposed answer: {answer}\n"
                f"Is the proposed answer correct? Answer Yes or No.")
    shown = "; ".join((references or [NO_ANSWER])[:MAX_REFERENCES])
    return (f"Question: {question}\nReference answer: {shown}\nProposed answer: {answer}\n"
            f"Does the proposed answer mean the same as the reference answer? Answer Yes or No.")


def judge_prompt(question: str, answer: str, references: list[str] | None, fmt: PromptFormat = PLAIN) -> str:
    """The judge's question in the model's own form, its reply opened with the cue: the next word is the verdict."""
    return fmt.render([{"role": USER, "content": judge_question(question, answer, references)}]) + fmt.cue


def verdict_ids(tokenizer) -> tuple[int, int]:
    """The two verdicts as the word after the cue."""
    ids = [tokenizer(v, add_special_tokens=False).input_ids for v in VERDICTS]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f"a verdict must be one token: {dict(zip(VERDICTS, ids))}")
    return ids[0][0], ids[1][0]


@dataclass(frozen=True)
class ModelJudge:
    """The model at bf16 says whether each answer is right: the probability of Yes, one per answer."""

    ids: tuple[int, int]
    model: object
    tokenizer: object
    ctl: object
    fmt: PromptFormat = PLAIN
    batch_size: int = 16
    name: str = "model_judge"

    @classmethod
    def build(cls, model, tokenizer, ctl, fmt: PromptFormat, **kwargs) -> ModelJudge:
        return cls(verdict_ids(tokenizer), model, tokenizer, ctl, fmt, **kwargs)

    def p_yes(self, questions: list[str], answers: list[str], references: list[list[str]] | None) -> np.ndarray:
        """`references=None` judges without them; otherwise one list per answer."""
        refs = [None] * len(answers) if references is None else references
        prompts = [judge_prompt(q, a, r, self.fmt) for q, a, r in zip(questions, answers, refs, strict=True)]
        self.ctl.set_all(Level.BF16)
        out = np.empty(len(prompts))
        for start in range(0, len(prompts), self.batch_size):
            chunk = slice(start, start + self.batch_size)
            out[chunk] = np.exp(letter_logprobs_batch(self.model, self.tokenizer, prompts[chunk], list(self.ids))[:, 0])
        return out
