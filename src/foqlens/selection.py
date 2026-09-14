"""What the corpus selection records and how it decides: one answer per line, three judges, a frozen file.

Every answer the model writes is kept as an Answer - one line of answers/<level>/<corpus>.jsonl, one
file per corpus and per quantization level of the model that answered. A line carries everything a
judge or a later reader needs: the question's number at the pinned revision, the prompt it was asked
with, the raw reply and the answer taken from it, and the scores of the two automatic judges.

Three judges decide whether the model knows a question (docs/corpus.md): exact match and F1, the
model itself asked Yes or No against the reference (foqlens.judging), and Claude, who reads the
answers. Claude's verdict decides wherever it is given; without it a refusal is a question the model
does not know, and otherwise the two automatic judges decide only where they agree, the rest staying
open for Claude.

The selection ends in a FrozenCorpus: the numbers kept and excluded, each with its reason, the
revisions, the prompts, and the questions spent on choosing the prompt. Every later run reads that
file; how it was assembled is recorded, not re-derived.

Invariant: a verdict never depends on the order of answers or on which level wrote an answer other than its own line.
Invariant: an Answer survives a round trip through its JSON line unchanged.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import numpy as np

# The share of each corpus spent on choosing its prompt, drawn at random (Volodya 14.09: 1%), and the
# fewest questions that can still tell two prompts apart: 1% of ARC-Challenge is 10, of NQ-open 36.
TUNING_SHARE = 0.01
TUNING_FLOOR = 50
# The model judge's probability of Yes above which it says the answer is right. Its verdicts on E2B-it
# sit at 0 and 1 (tests/test_it_gpu.py), so the threshold is the middle rather than a tuned value.
JUDGE_YES = 0.5
# Where a reply writes its reasoning first, the answer follows "Answer:" on a line of its own, whatever
# markdown the model wraps around it.
MARKUP = "*_#`"
ANSWER_LINE = re.compile(rf"^[\s{re.escape(MARKUP)}]*answer[\s{re.escape(MARKUP)}]*:(.*)$", re.IGNORECASE)
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


class Verdict(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    OPEN = "open"  # the automatic judges disagree and Claude has not read it yet


class Reason(StrEnum):
    CLAUDE = "claude"            # Claude read the answer
    REFUSED = "refused"          # the model said it does not know
    JUDGES_AGREE = "judges_agree"
    JUDGES_DISAGREE = "judges_disagree"


# A refusal is not an answer, whatever the judges say of it: shown the reference, the model judge said
# Yes to "I do not have specific information about..." against "Justin Timberlake" (NQ-open, 2026-09-14).
REFUSAL = re.compile(r"\b(i (do not|don't|cannot|can't) (know|have|find|answer|provide)|not (sure|able to)|"
                     r"no (information|way to know))\b", re.IGNORECASE)


def is_refusal(text: str) -> bool:
    """The model says it does not know, rather than answering - a heuristic over the usual phrasings."""
    return bool(REFUSAL.search(text))


@dataclass(frozen=True)
class Answer:
    """One question answered once: a line of answers/<level>/<corpus>.jsonl."""

    corpus: str
    id: str
    revision: str        # the corpus's pinned revision
    model: str           # the checkpoint and its revision, e.g. google/gemma-4-E2B-it@3e22461f
    level: str           # the quantization level it was read at: bf16, D8, D6, D4, D2
    prompt: str          # which prompt variant it was asked with
    reply: str           # what the model wrote, as it wrote it
    answer: str          # the answer taken from the reply
    reasoning: str | None  # the solution or justification before the answer, where one was asked for
    exact_match: float
    f1: float
    judge_with_reference: float     # the model judge's P(Yes) shown the reference
    judge_without_reference: float  # and not shown it
    tokens: int
    stopped: bool        # it ended on its own rather than at the token limit

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, row: dict) -> Answer:
        return cls(**row)


def tuning_sample(ids: list[str], seed: int, share: float = TUNING_SHARE, floor: int = TUNING_FLOOR) -> list[str]:
    """The questions spent on choosing the prompt: a seeded random draw, kept in the corpus's own order."""
    count = min(len(ids), max(floor, round(share * len(ids))))
    picked = np.random.default_rng(seed).choice(len(ids), size=count, replace=False)
    return [ids[i] for i in sorted(picked.tolist())]


def strip_markup(text: str) -> str:
    """The answer without the dress an -it model puts on it: HTML tags ("<h2>Chaplin</h2>"), emphasis, headings."""
    return HTML_TAG.sub("", text).strip(MARKUP + " ")


def split_reasoning(reply: str) -> tuple[str | None, str | None]:
    """The reasoning and the answer of a reply that ends with "Answer: ..." on a line of its own.

    An -it model marks the line up as it likes ("**Answer:** east"), so the cue is found through the
    markup. A reply without the line gives no answer: it ran out of tokens or never concluded, and
    what it wrote is reasoning, not an answer to be scored.
    """
    lines = reply.strip().split("\n")
    for i in range(len(lines) - 1, -1, -1):
        found = ANSWER_LINE.match(lines[i])
        if found:
            return "\n".join(lines[:i]).strip() or None, strip_markup(found.group(1))
    return reply.strip() or None, None


def verdict(answer: Answer, claude: Verdict | None = None) -> tuple[Verdict, Reason]:
    """Whether the model knows this question, and which judge said so."""
    if claude is not None:
        return claude, Reason.CLAUDE
    if is_refusal(answer.reply):
        return Verdict.UNKNOWN, Reason.REFUSED
    match = answer.exact_match == 1.0
    judged = answer.judge_with_reference > JUDGE_YES
    if match == judged:
        return (Verdict.KNOWN if match else Verdict.UNKNOWN), Reason.JUDGES_AGREE
    return Verdict.OPEN, Reason.JUDGES_DISAGREE


@dataclass(frozen=True)
class FrozenCorpus:
    """The selected corpus as a file of numbers: what every later run reads."""

    corpus: str
    revision: str
    model: str
    prompt: str
    kept: tuple[str, ...]                       # the model knows these
    excluded: dict[str, str] = field(default_factory=dict)  # id -> the reason it is out
    tuning: tuple[str, ...] = ()                # spent on choosing the prompt, never measured on
    unknown_share: tuple[str, ...] = ()          # the 10% of excluded questions stage 2 asks anyway, marked

    def to_json(self) -> dict:
        return {**asdict(self), "kept": list(self.kept), "tuning": list(self.tuning),
                "unknown_share": list(self.unknown_share)}

    @classmethod
    def from_json(cls, row: dict) -> FrozenCorpus:
        return cls(**{**row, "kept": tuple(row["kept"]), "tuning": tuple(row["tuning"]),
                      "unknown_share": tuple(row["unknown_share"])})
