"""Infrastructure: reading the query sets and writing run outputs."""

from __future__ import annotations

import json
from pathlib import Path

from foqlens.evaluate import Question, mc_prompt
from foqlens.selection import Answer, ClaudeVerdict, FrozenCorpus


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def read_questions(prompts_dir: Path, spec: str, limit: int | None = None) -> list[Question]:
    """Multiple-choice questions of one domain file; spec may carry a subdirectory (heldout/history)."""
    domain = Path(spec).name
    return [Question(domain, mc_prompt(r["text"], r["choices"]), int(r["answer"])) for r in read_jsonl(prompts_dir / f"{spec}.jsonl", limit)]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def answers_path(root: Path, level: str, corpus: str) -> Path:
    """answers/<level>/<corpus>.jsonl: one file per corpus and per quantization level of the model that answered."""
    return root / level / f"{corpus}.jsonl"


def append_answers(path: Path, answers: list[Answer]) -> None:
    """A batch goes to the file as soon as it is written, so a stopped run loses at most the batch in flight."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(a.to_json(), ensure_ascii=False) + "\n" for a in answers)


def read_answers(path: Path) -> list[Answer]:
    return [Answer.from_json(row) for row in read_jsonl(path)] if path.exists() else []


def written_ids(path: Path) -> set[str]:
    """The questions already answered in this file: a restart skips them."""
    return {a.id for a in read_answers(path)}


def read_frozen(path: Path) -> FrozenCorpus:
    """A frozen corpus file (scripts/freeze_corpus.py): the questions every later run asks."""
    return FrozenCorpus.from_json(json.loads(path.read_text(encoding="utf-8")))


def append_verdicts(path: Path, verdicts: list[ClaudeVerdict]) -> None:
    """Claude's readings, laid out like the answers they read: verdicts/<level>/<corpus>.jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(v.to_json(), ensure_ascii=False) + "\n" for v in verdicts)


def read_verdicts(path: Path) -> dict[str, ClaudeVerdict]:
    """By question; a question read again keeps its last reading."""
    return {v.id: v for v in (ClaudeVerdict.from_json(row) for row in read_jsonl(path))} if path.exists() else {}
