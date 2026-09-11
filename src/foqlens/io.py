"""Infrastructure: reading the query sets and writing run outputs."""

from __future__ import annotations

import json
from pathlib import Path

from foqlens.evaluate import Question, mc_prompt


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
