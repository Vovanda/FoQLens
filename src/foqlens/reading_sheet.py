"""Infrastructure: a sheet of answers laid out for Claude to read, and the marks Claude writes back.

A sheet numbers its answers from 1 and stores only their question numbers, so a reading costs the
answers of one sheet and nothing else. Claude marks only the answers whose reading differs from the
sheet's default, one line per reading: "w 3 7 12", "o 5: the reference names the town", "n 1-4". A
line is a letter (r right, o right in other words, w wrong, n no answer), the numbers or ranges it
covers and, after a colon, a note for all of them. Where the judges disagree there is no default and
every answer is marked.

The sheet shows the question, the references and the answer, never the judges' scores: a reading
that saw them would measure its agreement with them against itself.

Invariant: every answer of a sheet gets exactly one reading; a number marked twice, out of range or,
without a default, left unmarked is an error.
"""

from __future__ import annotations

from dataclasses import dataclass

from foqlens.corpora import Row
from foqlens.selection import Answer, Reading, Turn

LETTERS = {"r": Reading.RIGHT, "o": Reading.OTHER_WORDS, "w": Reading.WRONG, "n": Reading.NO_ANSWER}
# The reading an unmarked answer takes: the one both judges gave. Where they disagree there is none.
DEFAULTS = {Turn.DISAGREE: None, Turn.DOUBTFUL_NO: Reading.WRONG, Turn.SURE_NO: Reading.WRONG,
            Turn.AGREED_YES: Reading.RIGHT}
# Enough of a question, an answer and a passage to read them by; beyond this a text is cut with a mark.
QUESTION_CHARS = 400
ANSWER_CHARS = 400
PASSAGE_CHARS = 2000
REFERENCES_SHOWN = 6  # TriviaQA lists up to dozens of aliases; the first few name the entity
CUT = "…"


@dataclass(frozen=True)
class Sheet:
    """The answers on one sheet, by question number, in the order they are numbered."""

    corpus: str
    level: str
    turn: Turn
    ids: tuple[str, ...]

    def to_json(self) -> dict:
        return {"corpus": self.corpus, "level": self.level, "turn": int(self.turn), "ids": list(self.ids)}

    @classmethod
    def from_json(cls, row: dict) -> Sheet:
        return cls(row["corpus"], row["level"], Turn(row["turn"]), tuple(row["ids"]))


def flat(text: str, limit: int) -> str:
    """One line, cut to the limit."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + CUT


def render(answers: list[Answer], rows: dict[str, Row], passage: bool = False) -> str:
    """The numbered lines Claude reads: question, references, answer, and the passage where asked."""
    lines = []
    for k, answer in enumerate(answers, 1):
        row = rows[answer.id]
        # SQuAD lists a reference once per annotator, most of them the same span.
        distinct = tuple(dict.fromkeys(row.answers))
        references = " / ".join(distinct[:REFERENCES_SHOWN]) or "(the passage has no answer)"
        lines.append(f"{k}. Q: {flat(row.question, QUESTION_CHARS)} | REF: {references} | "
                     f"A: {flat(answer.answer or '', ANSWER_CHARS)}")
        if passage and row.context:
            lines.append(f"   P: {flat(row.context, PASSAGE_CHARS)}")
    return "\n".join(lines)


def numbers(token: str) -> range:
    first, _, last = token.partition("-")
    return range(int(first), int(last or first) + 1)


def parse_marks(text: str, size: int, default: Reading | None) -> list[tuple[Reading, str]]:
    """The reading and note of every answer of a sheet of `size`, in its order."""
    marked: dict[int, tuple[Reading, str]] = {}
    for line in filter(None, (line.strip() for line in text.splitlines())):
        head, _, note = line.partition(":")
        letter, *tokens = head.split()
        if letter not in LETTERS:
            raise ValueError(f"unknown reading {letter!r} in {line!r}: one of {', '.join(LETTERS)}")
        for k in (k for token in tokens for k in numbers(token)):
            if not 1 <= k <= size:
                raise ValueError(f"answer {k} is not on a sheet of {size}")
            if k in marked:
                raise ValueError(f"answer {k} is marked twice")
            marked[k] = (LETTERS[letter], note.strip())
    unmarked = [k for k in range(1, size + 1) if k not in marked]
    if unmarked and default is None:
        raise ValueError(f"this sheet has no default reading, and these are unmarked: {unmarked}")
    return [marked.get(k, (default, "")) for k in range(1, size + 1)]
