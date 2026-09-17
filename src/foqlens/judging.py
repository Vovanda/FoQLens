"""The full model as a judge of free answers: shown a question, the correct answers and an answer, it grades it.

The second of the corpus's three judges (docs/corpus.md): exact match and F1 miss an answer that is
right in other words, and Claude reads what the first two leave open. The model reads an exam whose
correct answers are known and grades the examinee's answer against them without answering the question
itself. It reasons as it needs and ends its reply with two lines, the kind of answer (GRADES) and whether
the answer is accepted; only those two lines are read (read_verdict). The question is rendered in the
model's own form (foqlens.prompting).

A reply read in one token could not tell an answer from garbage: it accepted 45-85% of D2's empty and
random-symbol answers whatever the instructions, and called most of them Correct. Given room to reason,
it calls them Garbage (47 of 48 on held-out answers, E016, 2026-09-16). An answer of whitespace alone is
not shown to it: it is Garbage without a question asked.

Invariant: the judge never sees who wrote an answer - its question holds the question, the correct answers
and the answer, and nothing else.
Invariant: an answer of whitespace alone is Garbage and not accepted, and costs no generation.
Invariant: a reply without the two closing lines is N/A and not accepted; one the limit cut is asked once more with a larger one.
Invariant: the judge reads at bf16 whatever layout is under test, and leaves the model at bf16.
Invariant: one verdict per answer, in the answers' order, whatever order the batches run in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from foqlens.extractive import NO_ANSWER
from foqlens.generation import END_OF_TURN, Decoder, generate_replies
from foqlens.graph_decode import STATIC
from foqlens.prompting import PLAIN, USER, PromptFormat
from foqlens.quant import Level

# TriviaQA lists up to dozens of aliases per answer; the first few say what the answer is, the rest
# only lengthen the prompt.
MAX_REFERENCES = 5
# Told nothing, the -it judge answers the question itself and grades its own answer: asked whether the answer
# "means the same as the reference", it said No to 503 of the 12,588 exact matches Claude reads right
# ("Captain Flint" against "Captain Flint"). Told the answers are known and not to solve, it stops (stage 1, 2026-09-15).
EXAM = ("You are grading an exam. The correct answers are already known, so do not answer the question yourself: "
        "only grade the examinee's answer against the correct answers.")
# The kinds of answer from the general to the particular: first whether it is readable text at all, then whether
# it answers the question, then how close it is (Volodya 2026-09-16). Few words, much meaning: "less precise" let
# a wrong neighbour into a right grade (Cambridge for Soham, 2026-09-15).
GRADES = (("Garbage", "not readable text: empty, random symbols or repeated fragments"),
          ("Noise", "readable words, but not an answer to this question"),
          ("Wrong", "wrong"), ("Related", "wrong, but related"), ("Partial", "partly right"),
          ("Nearly", "almost right"), ("Correct", "right"))
ACCEPTED = frozenset({"Nearly", "Correct"})  # the answer counts from "Nearly": the model knows the question (Volodya 2026-09-15)
GARBAGE, NOT_READ = "Garbage", "N/A"  # N/A: the reply has no closing lines the parser understands
# How a level's answers are reported (Volodya 2026-09-17): excellent is accepted; Partial is good - an incomplete
# answer, not a bad one; wrong ones are bad; the rest is no answer at all.
GRADE_GROUPS = {"excellent": ACCEPTED, "good": frozenset({"Partial"}), "bad": frozenset({"Wrong", "Related"}),
                "incoherent": frozenset({"Noise", GARBAGE})}
# The answer stands between two fence lines. Unfenced, an empty answer was followed by the instructions, and
# the judge took them for the answer (E016, 2026-09-16).
ANSWER_FENCE = "---"
# The reply is free and only its end is fixed. Said "one to three sentences" or opened as JSON, the judge
# held back its reasoning; in its own markdown it let 4% of wrong answers through against 18% in JSON (2026-09-16).
CHECKS = ("Grade from the general to the particular:\n"
          "1. Is the answer readable text? Empty, random symbols or repeated fragments are Garbage.\n"
          "2. Does it answer this question, rather than repeat what the question is about? If not, it is Noise.\n"
          "3. Does it mean the same as one of the correct answers, in any words?\n")
CLOSING = ("Reason as you need, then end your reply with exactly these two lines and nothing after them:\n"
           f"**Kind:** {'|'.join(word for word, _ in GRADES)}\n**Accepted:** Yes|No")
# The judge's replies on 247 labelled answers ran 9-161 tokens (median 33); the limit leaves room and
# bounds a batch, which costs its longest reply.
JUDGE_TOKENS = 200
# A reply cut by that limit is asked once more with this one: on E016's bf16 answers 63 of 20,640 were cut before
# their closing lines, mostly ARC and SQuAD where the judge works the answer through (2026-09-16). Few, so a
# second batch of them costs seconds, where a larger limit for all would slow every batch to its longest reply.
JUDGE_RETRY_TOKENS = 1024
# Replies decoded at once: 64 / 128 / 256 judged a level of 20,640 in 25 / 22 / 22 minutes (2026-09-16).
JUDGE_BATCH = 128
KIND = re.compile(r"\*\*Kind:\*\*\s*(\w+)")
ACCEPTED_LINE = re.compile(r"\*\*Accepted:\*\*\s*(Yes|No)\b", re.IGNORECASE)


def fenced(answer: str) -> str:
    """The answer between fence lines, so that where it ends is seen even when it is empty."""
    return f"{ANSWER_FENCE}\n{answer}\n{ANSWER_FENCE}"


def judge_question(question: str, answer: str, references: list[str]) -> str:
    """What the judge is asked. An empty list of references means the question has no answer.

    The question stays in view: it is what the answers are about. The references are a list of which any
    one is enough. SQuAD repeats a span once per annotator: each is shown once.
    """
    shown = " | ".join(list(dict.fromkeys(references or [NO_ANSWER]))[:MAX_REFERENCES])
    kinds = "".join(f"{word} - {meaning}\n" for word, meaning in GRADES)
    accepted = " or ".join(word for word, _ in GRADES if word in ACCEPTED)
    return (f"{EXAM}\n\nQuestion: {question}\nCorrect answers (any one is enough): {shown}\n"
            f"Examinee's answer:\n{fenced(answer)}\n\n{CHECKS}"
            f"The kind of answer is one of:\n{kinds}The answer is accepted only if its kind is {accepted}.\n{CLOSING}")


def judge_prompt(question: str, answer: str, references: list[str], fmt: PromptFormat = PLAIN) -> str:
    """The judge's question in the model's own form, ready for its reply."""
    return fmt.render([{"role": USER, "content": judge_question(question, answer, references)}])


@dataclass(frozen=True)
class Verdict:
    """What the judge said of one answer: the kind of answer, whether it is accepted, and its whole reply."""

    kind: str
    accepted: bool
    reply: str


def read_verdict(reply: str) -> Verdict:
    """The two closing lines of a reply; the last of each counts, since the reasoning may quote the format.

    Whether the answer is accepted follows from its kind, as the rule says: the judge's own Accepted line
    contradicted its kind on 0.9% of E016's verdicts, 658 of them "Partial, Yes" (2026-09-16).
    """
    kinds, accepted = KIND.findall(reply), ACCEPTED_LINE.findall(reply)
    known = {word for word, _ in GRADES}
    if not kinds or not accepted or kinds[-1] not in known:
        return Verdict(NOT_READ, False, reply)
    return Verdict(kinds[-1], kinds[-1] in ACCEPTED, reply)


def is_empty(answer: str | None) -> bool:
    return answer is None or not answer.strip()


@dataclass(frozen=True)
class ModelJudge:
    """The model at bf16 grades each answer, reasoning before its verdict."""

    model: object
    tokenizer: object
    ctl: object
    fmt: PromptFormat = PLAIN
    decoder: Decoder = STATIC
    batch_size: int = JUDGE_BATCH
    max_new_tokens: int = JUDGE_TOKENS
    retry_tokens: int = JUDGE_RETRY_TOKENS

    def verdicts(self, questions: list[str], answers: list[str | None], references: list[list[str]]) -> list[Verdict]:
        """One verdict per answer, in order. Empty answers are Garbage unasked; the rest are asked shortest prompt
        first, and a reply the limit cut before its closing lines is asked once more with the retry limit."""
        out: list[Verdict | None] = [Verdict(GARBAGE, False, "") if is_empty(a) else None for a in answers]
        prompts = {i: judge_prompt(q, a, r, self.fmt)
                   for i, (q, a, r, v) in enumerate(zip(questions, answers, references, out, strict=True)) if v is None}
        self.ctl.set_all(Level.BF16)
        cut = self._ask(prompts, list(prompts), self.max_new_tokens, out)
        self._ask(prompts, cut, self.retry_tokens, out)
        return out

    def _ask(self, prompts: dict[int, str], asked: list[int], max_new_tokens: int, out: list[Verdict | None]) -> list[int]:
        """The verdicts of the answers `asked`, written into `out`; returns those whose reply the limit cut unread."""
        order = sorted(asked, key=lambda i: len(prompts[i]))
        cut = []
        for start in range(0, len(order), self.batch_size):
            chunk = order[start:start + self.batch_size]
            replies = generate_replies(self.model, self.tokenizer, [prompts[i] for i in chunk], max_new_tokens,
                                       END_OF_TURN, self.decoder)
            for i, reply in zip(chunk, replies, strict=True):
                out[i] = read_verdict(reply.text)
                if out[i].kind == NOT_READ and not reply.stopped:
                    cut.append(i)
        return cut


class NotJudged:
    """The judge of a model that cannot judge: a level baked into the weights holds no bf16 weight.

    Its answers are written without a verdict, and the bf16 judge of another process reads them
    (foqlens.answering.rejudge).
    """

    def verdicts(self, questions: list[str], answers: list[str | None], references: list[list[str]]) -> list[None]:
        return [None] * len(answers)
