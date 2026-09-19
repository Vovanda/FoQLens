"""What the corpus selection records and how it decides: one answer per line, three judges, a frozen file.

Every answer the model writes is kept as an Answer - one line of answers/<level>/<corpus>.jsonl, one
file per corpus and per quantization level of the model that answered. A line carries everything a
judge or a later reader needs: the question's number at the pinned revision, the prompt it was asked
with, the raw reply and the answer taken from it, and the scores of the two automatic judges.

Three judges decide whether the model knows a question (docs/corpus.md): exact match and F1, the
model itself grading the answer as an exam against the reference (foqlens.judging), and Claude, who reads the
answers. Claude's verdict decides wherever it is given; without it a reply with no answer, an answer
to a question the passage does not answer, and a refusal are questions the model does not know, and
otherwise the two automatic judges decide only where they agree, the rest staying open for Claude.

Claude reads in turns (Volodya 15.09): first where the judges disagree, then the noes the judges are
not sure of, then the sure noes, and the agreed yeses last - every answer the rule does not decide
without reading. A reading is one of four: right, right in other words, wrong, no answer.

The selection ends in a FrozenCorpus: the numbers kept and excluded, each with its reason, the
revisions, the prompts, and the questions spent on choosing the prompt. Every later run reads that
file; how it was assembled is recorded, not re-derived.

Invariant: a verdict never depends on the order of answers or on which level wrote an answer other than its own line.
Invariant: an Answer and a ClaudeVerdict survive a round trip through their JSON lines unchanged.
Invariant: every answer either waits in exactly one turn or is decided by the rule without reading, never both.
Invariant: the draws of split_shares never share a question, and the same seed gives the same draws.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum

import numpy as np

# The share of each corpus spent on choosing its prompt, drawn at random (Volodya 14.09: 1%), and the
# fewest questions that can still tell two prompts apart: 1% of ARC-Challenge is 10, of NQ-open 36.
TUNING_SHARE = 0.01
TUNING_FLOOR = 50
# The model judge's probability of Yes above which it says the answer is right. Its verdicts on E2B-it
# sit at 0 and 1 (tests/test_it_gpu.py), so the threshold is the middle rather than a tuned value.
JUDGE_YES = 0.5
# Below this P(Yes) the judge is sure of its No: 55% of TriviaQA's answers and 77% of NQ-open's sit
# under it, and only 1.5-6.6% of any corpus between it and JUDGE_YES (stage 1, 2026-09-15).
JUDGE_SURE_NO = 0.05
# The kinds of a refused answer the judge is not sure of: close to right, unlike Wrong, Noise and Garbage.
DOUBTFUL_KINDS = frozenset({"Related", "Partial"})
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
    NO_ANSWER = "no_answer"      # the reply gives no answer: a solution that never reached its answer line
    UNANSWERABLE = "unanswerable"  # the passage has no answer (SQuAD v2) and the model gave one anyway
    REFUSED = "refused"          # the model said it does not know
    JUDGES_AGREE = "judges_agree"
    JUDGES_DISAGREE = "judges_disagree"


class Reading(StrEnum):
    """What Claude reads in an answer: two ways of knowing it, two of not."""

    RIGHT = "right"
    OTHER_WORDS = "other_words"  # right, but not in the reference's words
    WRONG = "wrong"
    NO_ANSWER = "no_answer"      # an evasion, a refusal the heuristic missed, a category instead of the thing


KNOWING = frozenset({Reading.RIGHT, Reading.OTHER_WORDS})


class Turn(IntEnum):
    """The order Claude reads in: where the automatic judges settle least, first."""

    DISAGREE = 1     # exact match and the model judge say different things
    DOUBTFUL_NO = 2  # both say No, but the answer shares words with the reference or the judge is not sure
    SURE_NO = 3
    AGREED_YES = 4


# A refusal is not an answer, whatever the judges say of it: shown the reference, the model judge said
# Yes to "I do not have specific information about..." against "Justin Timberlake" (NQ-open, 2026-09-14).
REFUSAL = re.compile(r"\b(i (do not|don't|cannot|can't) (know|have|find|answer|provide)|not (sure|able to)|"
                     r"no (information|way to know))\b", re.IGNORECASE)


def is_refusal(text: str) -> bool:
    """The model says it does not know, rather than answering - a heuristic over the usual phrasings."""
    return bool(REFUSAL.search(text))


# A question that asks yes or no, or names two candidates and asks which ("Who is older, A or B?"), is
# answered right half the time by a guess. It stays in the corpus, marked, and is counted apart
# (Volodya 15.09): its chance floor is 0.5, and a coarse model may lean to one side - "yes", or the
# first name.
TWO_WAY = re.compile(r"\bor\b", re.IGNORECASE)
YES_NO = frozenset({"yes", "no"})
WORD = re.compile(r"\w+")
NAMED_SHARE = 0.5  # more than half the answer's words stand in the question: it is one of the options named there
SHORT_WORD = 2     # "of", "a", "de" name nothing


def two_way_choice(question: str, references: tuple[str, ...]) -> bool:
    """Yes or no, or one of two options the question itself names: a guess is right half the time."""
    if any(r.strip().lower() in YES_NO for r in references):
        return True
    if not TWO_WAY.search(question):
        return False
    asked = set(WORD.findall(question.lower()))
    for reference in references:
        named = [w for w in WORD.findall(reference.lower()) if len(w) > SHORT_WORD]
        if named and sum(w in asked for w in named) / len(named) > NAMED_SHARE:
            return True
    return False


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
    tokens: int
    stopped: bool        # it ended on its own rather than at the token limit
    # The model judge's verdict (foqlens.judging): the kind of answer, whether it is accepted, its whole reply.
    # None where no judge has read the answer yet, and on the lines of the one-token judge before 2026-09-16.
    judge_kind: str | None = None
    judge_accepted: bool | None = None
    judge_reply: str | None = None
    # The one-token judge, until 2026-09-16: its P(Yes) shown the reference and not shown it, and its
    # probabilities of the kinds of answer then (best first). Kept so that the lines it wrote still read.
    judge_with_reference: float = float("nan")
    judge_without_reference: float = float("nan")
    judge_grades: tuple[float, ...] = ()

    @property
    def accepted(self) -> bool:
        """Whether the model judge accepts the answer, whichever judge wrote the line."""
        if self.judge_accepted is not None:
            return self.judge_accepted
        return self.judge_with_reference > JUDGE_YES

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, row: dict) -> Answer:
        return cls(**{**row, "judge_grades": tuple(row.get("judge_grades", ()))})


def tuning_sample(ids: list[str], seed: int, share: float = TUNING_SHARE, floor: int = TUNING_FLOOR) -> list[str]:
    """The questions spent on choosing the prompt: a seeded random draw, kept in the corpus's own order."""
    count = min(len(ids), max(floor, round(share * len(ids))))
    picked = np.random.default_rng(seed).choice(len(ids), size=count, replace=False)
    return [ids[i] for i in sorted(picked.tolist())]


def split_shares(ids: list[str], seed: int, shares: tuple[float, ...], floor: int = 0) -> list[list[str]]:
    """Disjoint seeded draws: the i-th holds shares[i] of `ids`, at least `floor` of them, in the ids' own order."""
    counts = [min(len(ids), max(floor, round(share * len(ids)))) for share in shares]
    if sum(counts) > len(ids):
        raise ValueError(f"shares {shares} with floor {floor} ask {sum(counts)} of {len(ids)} questions")
    order = np.random.default_rng(seed).permutation(len(ids))
    starts = np.cumsum([0, *counts])
    return [[ids[i] for i in sorted(order[a:b].tolist())] for a, b in zip(starts[:-1], starts[1:])]


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


def decided_unread(answer: Answer, answerable: bool = True) -> Reason | None:
    """Why the model does not know this question with nothing to read, or None if it waits for a reading.

    Against no reference the exact match is 1 only for the reply that says the passage has no answer.
    """
    if not answer.answer:
        return Reason.NO_ANSWER
    if not answerable and answer.exact_match != 1.0:
        return Reason.UNANSWERABLE
    if is_refusal(answer.reply):
        return Reason.REFUSED
    return None


def verdict(answer: Answer, claude: Verdict | None = None, answerable: bool = True) -> tuple[Verdict, Reason]:
    """Whether the model knows this question, and which judge said so."""
    if claude is not None:
        return claude, Reason.CLAUDE
    unread = decided_unread(answer, answerable)
    if unread is not None:
        return Verdict.UNKNOWN, unread
    match = answer.exact_match == 1.0
    judged = answer.accepted
    if match == judged:
        return (Verdict.KNOWN if match else Verdict.UNKNOWN), Reason.JUDGES_AGREE
    return Verdict.OPEN, Reason.JUDGES_DISAGREE


def turn(answer: Answer, answerable: bool = True) -> Turn | None:
    """The turn this answer waits in for Claude's reading, or None where the rule decides it unread."""
    if decided_unread(answer, answerable) is not None:
        return None
    match = answer.exact_match == 1.0
    if match != answer.accepted:
        return Turn.DISAGREE
    if match:
        return Turn.AGREED_YES
    if answer.f1 > 0 or doubtful_refusal(answer):
        return Turn.DOUBTFUL_NO
    return Turn.SURE_NO


def doubtful_refusal(answer: Answer) -> bool:
    """A refused answer the judge is not sure of: close to right by its kind, or by P(Yes) on the one-token judge's lines."""
    if answer.judge_accepted is not None:
        return answer.judge_kind in DOUBTFUL_KINDS
    return answer.judge_with_reference >= JUDGE_SURE_NO


@dataclass(frozen=True)
class ClaudeVerdict:
    """Claude's reading of one answer: a line of verdicts/<level>/<corpus>.jsonl."""

    corpus: str
    id: str
    level: str      # the level of the answer it reads
    reading: Reading
    date: str       # the day it was read, ISO
    note: str = ""  # why, where the reading is not obvious

    @property
    def verdict(self) -> Verdict:
        return Verdict.KNOWN if self.reading in KNOWING else Verdict.UNKNOWN

    def to_json(self) -> dict:
        return {**asdict(self), "reading": str(self.reading)}

    @classmethod
    def from_json(cls, row: dict) -> ClaudeVerdict:
        return cls(**{**row, "reading": Reading(row["reading"])})


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
    two_way: tuple[str, ...] = ()                # kept questions a guess between two answers right half the time

    def asked(self) -> tuple[str, ...]:
        """What stage 2 asks: the kept questions and the unknown share, never the ones spent on the prompt."""
        return self.kept + self.unknown_share

    def check(self, model: str, revision: str, prompt: str) -> None:
        """A run reads the file only for the model, the corpus revision and the prompt it was frozen with."""
        if (self.model, self.revision, self.prompt) != (model, revision, prompt):
            raise ValueError(f"{self.corpus} was frozen for {(self.model, self.revision, self.prompt)}, "
                             f"not {(model, revision, prompt)}")

    def to_json(self) -> dict:
        return {**asdict(self), **{name: list(getattr(self, name)) for name in LISTS}}

    @classmethod
    def from_json(cls, row: dict) -> FrozenCorpus:
        return cls(**{**row, **{name: tuple(row.get(name, ())) for name in LISTS}})


LISTS = ("kept", "tuning", "unknown_share", "two_way")  # the FrozenCorpus fields written as JSON lists
# Stage 2 asks 90% questions the full model knew and 10% it did not, drawn at random (Volodya 14.09).
UNKNOWN_SHARE = 0.1


def freeze(answers: list[Answer], claude: dict[str, Verdict], questions: dict[str, tuple[str, tuple[str, ...]]],
           tuning: tuple[str, ...], seed: int) -> FrozenCorpus:
    """The corpus as a file of numbers, by the same rule as every verdict (`verdict`).

    `questions` maps every question's number, in the corpus's own order, to its text and references; the
    questions spent on the prompt have no answer. An answer the rule leaves open is an error: the corpus
    is frozen only once every answer is decided.
    """
    by_id = {a.id: a for a in answers}
    fields = {(a.corpus, a.revision, a.model, a.prompt) for a in answers}
    if len(fields) != 1:
        raise ValueError(f"a corpus is frozen from one corpus, revision, model and prompt: {sorted(fields)}")
    corpus, revision, model, prompt = fields.pop()
    kept, excluded, two_way = [], {}, []
    for i, (question, references) in questions.items():
        if i not in by_id:
            continue
        v, reason = verdict(by_id[i], claude.get(i), bool(references))
        if v == Verdict.OPEN:
            raise ValueError(f"{corpus} {i}: the judges disagree and Claude has not read it")
        if v == Verdict.KNOWN:
            kept.append(i)
            if references and two_way_choice(question, references):
                two_way.append(i)
        else:
            excluded[i] = f"{v}:{reason}"
    pool = list(excluded)
    count = min(len(pool), round(len(kept) * UNKNOWN_SHARE / (1 - UNKNOWN_SHARE)))
    picked = np.random.default_rng(seed).choice(len(pool), size=count, replace=False)
    unknown = tuple(pool[k] for k in sorted(picked.tolist()))
    return FrozenCorpus(corpus, revision, model, prompt, tuple(kept), excluded, tuple(tuning), unknown, tuple(two_way))
