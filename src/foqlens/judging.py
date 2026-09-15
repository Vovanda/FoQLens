"""The full model as a judge of free answers: shown a question and an answer, it says Yes or No.

The second of the corpus's three judges (docs/corpus.md): exact match and F1 miss an answer that is
right in other words, and Claude reads what the first two leave open. The model judges in one forward
pass, without generating: its reply is opened with the cue "Answer:" and the log-probabilities of
" Yes" and " No" as the next word are renormalized between the two, so a model that would go on to
write something else still gives a verdict.

It asks in two forms: with the dataset's reference answers - an exam whose correct answers are known,
where the judge checks the examinee's answer against them and does not answer the question itself - and
without (is it right, as far as the model knows). The first is the verdict; the second shows how much
of it the reference carries. The question is rendered in the model's own form (foqlens.prompting).

Shown the reference, the judge also says what kind of answer it is (GRADES): its reply goes on as
"Yes, Correct", so the verdict it gave and a comma are written in and the kinds are read as the next
word - one more pass, again without generating.

Invariant: the judge never sees who wrote an answer - its question holds the question, the answer and,
in one form, the references, and nothing else.
Invariant: the kind of answer is read after the verdict the judge gave, never after the other one.
Invariant: the judge reads at bf16 whatever layout is under test, and leaves the model at bf16.
Invariant: the same prompts in the same batch give the same verdicts (evaluate.letter_logprobs_batch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.nn.attention import SDPBackend, sdpa_kernel

from foqlens.attention import MATH
from foqlens.evaluate import letter_logprobs_batch
from foqlens.extractive import NO_ANSWER
from foqlens.prompting import PLAIN, USER, PromptFormat
from foqlens.quant import Level
from foqlens.selection import JUDGE_YES

VERDICTS = (" Yes", " No")  # the word after the cue "Answer:" (prompting.PromptFormat.cue)
GRADE_CUE = ","  # between the verdict and the kind of answer, as the prompt asks the reply to be written
# TriviaQA lists up to dozens of aliases per answer; the first few say what the answer is, the rest
# only lengthen the prompt.
MAX_REFERENCES = 5


# Shown the question, the -it judge answers it itself and grades its own answer, not the one given: asked
# whether the answer "means the same as the reference", it said No to 503 of the 12,588 exact matches
# Claude reads right (4%; "Captain Flint" against "Captain Flint", P(Yes) 0.0001). Told the answers are
# known and not to solve, and asked to accept or not with the kind of answer as the reason, it says Yes to
# 96% of those 503 and still No to 53 of 60 wrong answers it had called wrong (stage 1, 2026-09-15).
EXAM = ("You are grading an exam. The correct answers are already known, so do not answer the question yourself: "
        "only grade the examinee's answer against the correct answers.")
# The kinds of answer, best to worst: the reason after the verdict (Volodya 2026-09-15). Few words, much
# meaning: "less precise" let a wrong neighbour into the second grade (Cambridge for Soham), and the word
# "Hint" was never chosen. Kept a list with the rule in words: the same ladder as a compact CSV table made
# the judge accept 22 of those 60 wrong answers.
GRADES = (("Correct", "right"), ("Nearly", "almost right"), ("Partial", "partly right"),
          ("Related", "wrong, but related"), ("Wrong", "wrong"))
ACCEPTED_GRADES = 2  # the answer counts from "Nearly": the model knows the question (Volodya 2026-09-15)


def judge_question(question: str, answer: str, references: list[str] | None) -> str:
    """What the judge is asked; `references=None` asks without them. An empty list means there is no answer.

    The question stays in view: it is what the answers are about. The references are a list of which any
    one is enough. SQuAD repeats a span once per annotator: each is shown once.
    """
    if references is None:
        return (f"Question: {question}\nProposed answer: {answer}\n"
                f"Is the proposed answer correct? Answer Yes or No.")
    shown = " | ".join(list(dict.fromkeys(references or [NO_ANSWER]))[:MAX_REFERENCES])
    kinds = "".join(f"{word} - {meaning}\n" for word, meaning in GRADES)
    accepted = " or ".join(word for word, _ in GRADES[:ACCEPTED_GRADES])
    return (f"{EXAM}\n\nQuestion: {question}\nCorrect answers (any one is enough): {shown}\n"
            f"Examinee's answer: {answer}\n\n"
            f"First say whether the answer is accepted - the examinee passes this question: Yes or No. "
            f"Then give the reason, the kind of answer it is:\n{kinds}"
            f"The answer is accepted (Yes) only if it is {accepted}.\nReply as: Yes, {GRADES[0][0]}")


def judge_prompt(question: str, answer: str, references: list[str] | None, fmt: PromptFormat = PLAIN) -> str:
    """The judge's question in the model's own form, its reply opened with the cue: the next word is the verdict."""
    return fmt.render([{"role": USER, "content": judge_question(question, answer, references)}]) + fmt.cue


def verdict_ids(tokenizer) -> tuple[int, int]:
    """The two verdicts as the word after the cue."""
    ids = [tokenizer(v, add_special_tokens=False).input_ids for v in VERDICTS]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f"a verdict must be one token: {dict(zip(VERDICTS, ids))}")
    return ids[0][0], ids[1][0]


def grade_ids(tokenizer) -> tuple[int, ...]:
    """The kinds of answer, best first, as the word after the verdict and its comma."""
    ids = [tokenizer(f" {word}", add_special_tokens=False).input_ids for word, _ in GRADES]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f"a kind of answer must be one token: {dict(zip((w for w, _ in GRADES), ids))}")
    return tuple(i[0] for i in ids)


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
    grade_tokens: tuple[int, ...] = ()
    kernels: tuple[SDPBackend, ...] = MATH  # the sdpa kernels its forward may use (foqlens.attention)

    @classmethod
    def build(cls, model, tokenizer, ctl, fmt: PromptFormat, **kwargs) -> ModelJudge:
        return cls(verdict_ids(tokenizer), model, tokenizer, ctl, fmt, grade_tokens=grade_ids(tokenizer), **kwargs)

    def p_yes(self, questions: list[str], answers: list[str], references: list[list[str]] | None) -> np.ndarray:
        """`references=None` judges without them; otherwise one list per answer."""
        refs = [None] * len(answers) if references is None else references
        prompts = [judge_prompt(q, a, r, self.fmt) for q, a, r in zip(questions, answers, refs, strict=True)]
        return np.exp(self._read(prompts, list(self.ids))[:, 0])

    def grades(self, questions: list[str], answers: list[str], references: list[list[str]],
               p_yes: np.ndarray) -> np.ndarray:
        """The probabilities of the kinds of answer after the verdict given with the references: [answers, GRADES]."""
        given = [VERDICTS[0] if p > JUDGE_YES else VERDICTS[1] for p in p_yes]
        prompts = [judge_prompt(q, a, r, self.fmt) + v + GRADE_CUE
                   for q, a, r, v in zip(questions, answers, references, given, strict=True)]
        return np.exp(self._read(prompts, list(self.grade_tokens)))

    def _read(self, prompts: list[str], ids: list[int]) -> np.ndarray:
        """The next word's probabilities among `ids`, renormalized, at bf16 in batches: log, [prompts, ids]."""
        self.ctl.set_all(Level.BF16)
        out = np.empty((len(prompts), len(ids)))
        with sdpa_kernel(list(self.kernels)):
            for start in range(0, len(prompts), self.batch_size):
                chunk = slice(start, start + self.batch_size)
                out[chunk] = letter_logprobs_batch(self.model, self.tokenizer, prompts[chunk], ids)
        return out


class NotJudged:
    """The judge of a model that cannot judge: a level baked into the weights holds no bf16 weight.

    Its answers are written with NaN verdicts and no kinds of answer, and the bf16 judge of another
    process reads them (foqlens.answering.rejudge).
    """

    def p_yes(self, questions: list[str], answers: list[str], references: list[list[str]] | None) -> np.ndarray:
        return np.full(len(answers), np.nan)

    def grades(self, questions: list[str], answers: list[str], references: list[list[str]],
               p_yes: np.ndarray) -> list[tuple[float, ...]]:
        return [()] * len(answers)
