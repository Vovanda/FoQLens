"""Answering a batch of questions in one setup: what the model writes, the answer taken from it, both automatic judges.

One step shared by every run that collects answers - choosing the prompt, stage 1, stage 2 - so that
an answer line means the same thing whichever run wrote it. Answers written earlier are read again by
the present judge the same way (`rejudge`): the reply stays, the verdicts are taken anew.

Invariant: a batch is written at the Asking's level and judged at bf16; the level is set again before every batch.
Invariant: one Answer per row, in the rows' order.
Invariant: rejudging changes only the judge's fields of an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from foqlens.corpora import Row
from foqlens.extractive import exact_match, token_f1
from foqlens.generation import DYNAMIC, Decoder, generate_replies
from foqlens.gpu_share import FULL, Pacer
from foqlens.prompt_variants import Variant
from foqlens.prompting import PromptFormat
from foqlens.quality import token_batches
from foqlens.quant import Level
from foqlens.selection import Answer

JUDGE_BATCH = 64  # the judge is one forward pass per prompt; a larger batch fills the card better than the default 16
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


@dataclass(frozen=True)
class Asking:
    """How a corpus is asked in a run: its setup and examples, the model and the level it is read at."""

    corpus: str
    revision: str        # the corpus's pinned revision
    model: str           # checkpoint@revision
    level: Level
    setup: Variant
    examples: tuple = ()

    def prompts(self, fmt: PromptFormat, rows: list[Row]) -> list[str]:
        return [fmt.render(self.setup.messages(r.context, r.question, self.examples)) for r in rows]

    def batches(self, fmt: PromptFormat, tokenizer, rows: list[Row]) -> list[list[Row]]:
        """The rows in batches by prompt length: at most BATCH prompts and BATCH_TOKENS padded tokens each."""
        lengths = [len(ids) for ids in tokenizer(self.prompts(fmt, rows))["input_ids"]]
        return [[rows[i] for i in idx.tolist()] for idx in token_batches(lengths, BATCH_TOKENS, BATCH)]

    def answer(self, model, tokenizer, ctl, fmt: PromptFormat, judge, rows: list[Row], pacer: Pacer = FULL,
               decoder: Decoder = DYNAMIC) -> list[Answer]:
        ctl.set_all(self.level)
        with pacer.batch():
            written = generate_replies(model, tokenizer, self.prompts(fmt, rows), self.setup.max_new_tokens, self.setup.stop,
                                       decoder)
        parts = [self.setup.extract(w.text) for w in written]
        judged = judge_answers(judge, rows, [a for _, a in parts], pacer)
        return [Answer(self.corpus, r.id, self.revision, self.model, level_label(self.level), self.setup.name,
                       w.text, a, reasoning, exact_match(a, list(r.answers)), token_f1(a, list(r.answers)),
                       yes, no_ref, w.tokens, w.stopped, grades)
                for r, w, (reasoning, a), (yes, no_ref, grades) in zip(rows, written, parts, judged, strict=True)]


def judge_answers(judge, rows: list[Row], answers: list[str | None],
                  pacer: Pacer = FULL) -> list[tuple[float, float, tuple[float, ...]]]:
    """Each answer's P(Yes) with the references and without, and the kinds of answer after the first verdict."""
    questions, references = [r.question for r in rows], [list(r.answers) for r in rows]
    with pacer.batch():
        with_ref = judge.p_yes(questions, answers, references)
        without_ref = judge.p_yes(questions, answers, None)
        grades = judge.grades(questions, answers, references, with_ref)
    return [(float(yes), float(no_ref), tuple(float(g) for g in kinds))
            for yes, no_ref, kinds in zip(with_ref, without_ref, grades, strict=True)]


def rejudge(judge, answers: list[Answer], rows: list[Row], pacer: Pacer = FULL) -> list[Answer]:
    """Answers written earlier, read again by the present judge; `rows` are their questions, in the same order."""
    if [a.id for a in answers] != [r.id for r in rows]:
        raise ValueError("each answer is rejudged against its own question")
    judged = judge_answers(judge, rows, [a.answer for a in answers], pacer)
    return [replace(a, judge_with_reference=yes, judge_without_reference=no_ref, judge_grades=grades)
            for a, (yes, no_ref, grades) in zip(answers, judged, strict=True)]
