"""Infrastructure: the free-answer corpora, every question with a stable number, at a pinned revision.

The corpus is a frozen artifact (docs/corpus.md): what is kept and what is excluded is recorded as
question numbers, never as texts, so every row carries an id that names the same question on every
reading of the same revision. The revision is pinned here rather than read off the hub at run time:
a dataset updated upstream must not change the corpus under a run.

TriviaQA's rc.nocontext validation lists 7,984 of its 9,960 questions twice (17,944 rows; two of
them differ between their rows); a question is kept once, at its first row. NQ-open has no id of its
own, so its number is the row at the pinned revision.

Invariant: within a corpus the ids are unique, and the same revision gives the same ids in the same order.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class Row:
    """One question: its number in the corpus, every accepted answer, and the passage where the corpus gives one.

    No answers means the question is unanswerable (SQuAD v2).
    """

    id: str
    question: str
    answers: tuple[str, ...]
    context: str | None = None


@dataclass(frozen=True)
class Source:
    repo: str
    path: str
    revision: str

    def local(self) -> str:
        return hf_hub_download(self.repo, self.path, repo_type="dataset", revision=self.revision)

    def __str__(self) -> str:
        return f"{self.repo}@{self.revision[:8]}"


def trivia_answers(record: dict) -> tuple[str, ...]:
    """TriviaQA names one entity under its aliases: the value first, then every alias once."""
    return tuple(dict.fromkeys([record["answer"]["value"], *record["answer"]["aliases"]]))


def trivia_rows(records: Iterable[dict]) -> list[Row]:
    """Every question once, at its first row: the validation file lists most of them twice."""
    seen: set[str] = set()
    rows = []
    for r in records:
        if r["question_id"] not in seen:
            seen.add(r["question_id"])
            rows.append(Row(r["question_id"], r["question"], trivia_answers(r)))
    return rows


def nq_rows(records: Iterable[dict]) -> list[Row]:
    """NQ-open has no id of its own: a question's number is its row at the pinned revision."""
    return [Row(str(i), r["question"], tuple(r["answer"])) for i, r in enumerate(records)]


def squad_rows(records: Iterable[dict]) -> list[Row]:
    return [Row(r["id"], r["question"], tuple(r["answers"]["text"]), r["context"]) for r in records]


def hotpot_passage(context: dict) -> str:
    """The paragraphs under their titles: two carry the answer and the step between them, eight are distractors."""
    return "\n\n".join(title + ": " + "".join(sentences)
                       for title, sentences in zip(context["title"], context["sentences"], strict=True))


def hotpot_rows(records: Iterable[dict]) -> list[Row]:
    return [Row(r["id"], r["question"], (r["answer"],), hotpot_passage(r["context"])) for r in records]


# A question that points at its options has no answer without them.
POINTS_AT_OPTIONS = re.compile(r"\b(which of (the following|these)|the following|listed below)\b", re.IGNORECASE)


def arc_closed_rows(records: Iterable[dict]) -> list[Row]:
    """ARC asked without its options: the model writes its answer, the right option's text is the reference.

    Nothing matches the answer to the options - that matching read letters and moved its pick with
    their order on a fifth of ARC-Challenge (runs/reference/corpus-knowledge). The options are never
    shown, so a question keeps its place whatever their number.
    """
    rows = []
    for r in records:
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        if r["answerKey"] in labels and not POINTS_AT_OPTIONS.search(r["question"]):
            rows.append(Row(r["id"], r["question"], (texts[labels.index(r["answerKey"])],)))
    return rows


@dataclass(frozen=True)
class Corpus:
    source: Source
    columns: tuple[str, ...]
    build: Callable[[list[dict]], list[Row]]
    passage: bool = False  # the answer is read from a given passage rather than recalled from the weights


CORPORA = {
    "triviaqa": Corpus(
        Source("mandarjoshi/trivia_qa", "rc.nocontext/validation-00000-of-00001.parquet",
               "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"),
        ("question_id", "question", "answer"), trivia_rows),
    "nq_open": Corpus(
        Source("google-research-datasets/nq_open", "nq_open/validation-00000-of-00001.parquet",
               "5dd9790a83002ad084ddeb7c420dc716852c6f28"),
        ("question", "answer"), nq_rows),
    "arc_challenge_closed": Corpus(
        Source("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet",
               "210d026faf9955653af8916fad021475a3f00453"),
        ("id", "question", "choices", "answerKey"), arc_closed_rows),
    "squad_v2": Corpus(
        Source("rajpurkar/squad_v2", "squad_v2/validation-00000-of-00001.parquet",
               "3ffb306f725f7d2ce8394bc1873b24868140c412"),
        ("id", "question", "context", "answers"), squad_rows, passage=True),
    "hotpotqa": Corpus(
        Source("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet",
               "1908d6afbbead072334abe2965f91bd2709910ab"),
        ("id", "question", "answer", "context"), hotpot_rows, passage=True),
}


def read(name: str, limit: int | None = None) -> tuple[list[Row], Source]:
    """One corpus at its pinned revision, in its own order; `limit` keeps the first rows, for a smoke check."""
    corpus = CORPORA[name]
    rows = corpus.build(pq.read_table(corpus.source.local(), columns=list(corpus.columns)).to_pylist())
    return (rows[:limit] if limit else rows), corpus.source
