"""Answering a batch of questions in one setup: what the model writes, the answer taken from it, the automatic judges.

One step shared by every run that collects answers - choosing the prompt, stage 1, stage 2 - so that
an answer line means the same thing whichever run wrote it. Answers written earlier are read again by
the present judge the same way (`rejudge`): the reply stays, the verdicts are taken anew.

Invariant: a batch is written at the Asking's reading - its level, or every question's own layout - and judged at bf16;
the reading is set again before every batch.
Invariant: one Answer per row, in the rows' order.
Invariant: rejudging changes only the judge's fields of an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from foqlens.corpora import Row
from foqlens.extractive import exact_match, token_f1
from foqlens.generation import DYNAMIC, Decoder, generate_replies
from foqlens.gpu_share import FULL, Pacer
from foqlens.judging import Verdict
from foqlens.prompt_variants import Variant
from foqlens.prompting import PromptFormat
from foqlens.quality import token_batches
from foqlens.quant import Level
from foqlens.selection import Answer

BATCH = 128  # a decoding step costs the same at any batch here: E2B-it answered 3.7 / 7.3 / 14.8 questions a second at 32 / 64 / 128
# Padded prompt tokens a batch may hold. At 65k (128 prompts of HotpotQA passages) the prefill ran out of
# the card with 16.2 GiB allocated: the double-wide MLP asked 1.5 GiB for one activation (2026-09-14).
# Half of that keeps the peak near 12 GiB of the 19.2 the share allows, and a HotpotQA batch holds ~24
# passages of ~1300 tokens instead of ~12: a batch costs its longest reply in steps, so twice the rows
# per batch is half the time on HotpotQA.
BATCH_TOKENS = 32768


def level_label(level: Level) -> str:
    """The level as answer files and lines name it: bf16, d8, d6, d4, d2."""
    return level.name.lower()


class Reading(Protocol):
    """How the weights are read for a batch: one level for all, or a layout of every question's own."""

    label: str  # what answer lines name the level by

    def apply(self, ctl, corpus: str, rows: list[Row]) -> None:
        """Set the controller for this batch, one row of the batch per sample."""
        ...


@dataclass(frozen=True)
class UniformReading:
    """Every block at one level - the uniform ladder."""

    level: Level

    @property
    def label(self) -> str:
        return level_label(self.level)

    def apply(self, ctl, corpus: str, rows: list[Row]) -> None:
        ctl.set_all(self.level)


@dataclass(frozen=True)
class Asking:
    """How a corpus is asked in a run: its setup and examples, the model and how its weights are read - the level, or
    `reading` where each question has a layout of its own (regulator.RegulatedReading)."""

    corpus: str
    revision: str        # the corpus's pinned revision
    model: str           # checkpoint@revision
    level: Level
    setup: Variant
    examples: tuple = ()
    reading: Reading | None = None

    @property
    def read(self) -> Reading:
        return UniformReading(self.level) if self.reading is None else self.reading

    def prompts(self, fmt: PromptFormat, rows: list[Row]) -> list[str]:
        return [fmt.render(self.setup.messages(r.context, r.question, self.examples)) for r in rows]

    def batches(self, fmt: PromptFormat, tokenizer, rows: list[Row], max_tokens: int | None = None) -> list[list[Row]]:
        """The rows in batches by prompt length: at most BATCH prompts and `max_tokens` (BATCH_TOKENS) padded tokens each.

        No rows is no batches: a run that resumes over a layout whose file it already holds whole asks for nothing here,
        and the tokenizer raises IndexError on an empty call instead of returning an empty encoding.
        """
        if not rows:
            return []
        lengths = [len(ids) for ids in tokenizer(self.prompts(fmt, rows))["input_ids"]]
        return [[rows[i] for i in idx.tolist()] for idx in token_batches(lengths, max_tokens or BATCH_TOKENS, BATCH)]

    def answer(self, model, tokenizer, ctl, fmt: PromptFormat, judge, rows: list[Row], pacer: Pacer = FULL,
               decoder: Decoder = DYNAMIC) -> list[Answer]:
        reading = self.read
        reading.apply(ctl, self.corpus, rows)
        with pacer.batch():
            written = generate_replies(model, tokenizer, self.prompts(fmt, rows), self.setup.max_new_tokens, self.setup.stop,
                                       decoder)
        parts = [self.setup.extract(w.text) for w in written]
        verdicts = judge_answers(judge, rows, [a for _, a in parts], pacer)
        return [with_verdict(Answer(corpus=self.corpus, id=r.id, revision=self.revision, model=self.model,
                                    level=reading.label, prompt=self.setup.name, reply=w.text, answer=a,
                                    reasoning=reasoning, exact_match=exact_match(a, list(r.answers)),
                                    f1=token_f1(a, list(r.answers)), tokens=w.tokens, stopped=w.stopped), v)
                for r, w, (reasoning, a), v in zip(rows, written, parts, verdicts, strict=True)]


def judge_answers(judge, rows: list[Row], answers: list[str | None], pacer: Pacer = FULL) -> list[Verdict | None]:
    """The judge's verdict on each answer against its question's references; None where the judge cannot judge."""
    with pacer.batch():
        return judge.verdicts([r.question for r in rows], answers, [list(r.answers) for r in rows])


def with_verdict(answer: Answer, verdict: Verdict | None) -> Answer:
    """The answer carrying this verdict and no other: the one-token judge's fields are cleared."""
    if verdict is None:
        return replace(answer, judge_kind=None, judge_accepted=None, judge_reply=None)
    return replace(answer, judge_kind=verdict.kind, judge_accepted=verdict.accepted, judge_reply=verdict.reply,
                   judge_with_reference=float("nan"), judge_without_reference=float("nan"), judge_grades=())


def cut_ids(answers: list[Answer]) -> set[str]:
    """The questions whose answer ran into the token cap instead of ending on its own."""
    return {a.id for a in answers if not a.stopped}


def rejudge(judge, answers: list[Answer], rows: list[Row], pacer: Pacer = FULL) -> list[Answer]:
    """Answers written earlier, read again by the present judge; `rows` are their questions, in the same order."""
    if [a.id for a in answers] != [r.id for r in rows]:
        raise ValueError("each answer is rejudged against its own question")
    verdicts = judge_answers(judge, rows, [a.answer for a in answers], pacer)
    return [with_verdict(a, v) for a, v in zip(answers, verdicts, strict=True)]
