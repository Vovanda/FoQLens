"""Infrastructure: reading the query sets and writing run outputs."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from collections.abc import Collection
from pathlib import Path

import numpy as np

from foqlens.evaluate import Question, mc_prompt
from foqlens.selection import Answer, ClaudeVerdict, FrozenCorpus


def save_npz_atomic(path: Path, **arrays) -> None:
    """np.savez to a temporary file beside `path`, then a rename over it: a process killed while writing leaves the
    previous file whole, never a torn one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class Checkpoint:
    """The partial state of a long pass, kept beside its output: what is done is read back after a stop or a crash,
    and the pass goes on from there. The pass's work units are fixed before it starts, so a resumed pass computes
    exactly what an unbroken one would. `plan` names the units (a hash of their layout); a checkpoint of another plan
    is refused, not resumed.

    Invariant: a checkpoint read back holds bit for bit the arrays saved, and only for the same plan."""

    def __init__(self, path: Path, plan: str):
        self.path, self.plan = path, plan

    def load(self) -> dict[str, np.ndarray] | None:
        if not self.path.exists():
            return None
        with np.load(self.path, allow_pickle=False) as z:
            if str(z["plan"]) != self.plan:
                raise ValueError(f"{self.path} is a checkpoint of another pass; move it away to start over")
            return {k: z[k] for k in z.files if k != "plan"}

    def save(self, **arrays) -> None:
        save_npz_atomic(self.path, plan=self.plan, **arrays)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def plan_of(*parts) -> str:
    """A short fingerprint of a pass's work units (its questions, batches, settings) for a Checkpoint."""
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def read_npz_parts(pattern: str, per_run: frozenset[str] = frozenset(),
                   only: Collection[str] | None = None,
                   skip: Collection[str] = ()) -> dict[str, np.ndarray]:
    """One .npz, or the shards a glob names joined in their order: every array per question concatenated; the 0-d
    values and the `per_run` arrays (what describes the run, as its group names) taken once, refused if parts differ.

    `only` names the fields the caller needs. Shards written by different versions of a run hold different fields -
    one that has since dropped a field still sits on disk beside a new one - and joining them whole fails on a field
    the older shard alone has. With `only`, what the caller does not ask for is not read, and a field missing from a
    shard is named with the file it is missing from.

    `skip` drops named fields instead, for a caller that wants the rest whole. A `per_run` field measured inside its
    own shard - the rungs' ratios, read on that shard's questions - honestly differs between shards, and joining them
    is refused; a caller that does not use it skips it rather than reading one shard's value as if it were both.
    """
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    if not paths:
        raise FileNotFoundError(pattern)
    wanted = None if only is None else set(only) | set(per_run)
    dropped = set(skip)
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            if wanted is not None and (lacks := wanted - dropped - set(z.files)):
                raise ValueError(f"{path} holds no {', '.join(sorted(lacks))}")
            parts.append({k: z[k] for k in z.files if (wanted is None or k in wanted) and k not in dropped})
    joined = {}
    for key, value in parts[0].items():
        if value.ndim == 0 or key in per_run:
            if any(not np.array_equal(value, p[key]) for p in parts[1:]):
                raise ValueError(f"the parts of {pattern} differ in {key}")
            joined[key] = value
        elif missing := [path for path, p in zip(paths[1:], parts[1:]) if key not in p]:
            raise ValueError(f"{missing[0]} holds no {key}, which {paths[0]} does: name the fields you need in `only`")
        else:
            joined[key] = np.concatenate([p[key] for p in parts])
    return joined


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


def judge_run_path(root: Path, level: str, corpus: str, run: str) -> Path:
    """judge/<level>/<corpus>/<run>.jsonl: one pass of the judge over one answers file; a later pass is a new run."""
    return root / level / corpus / f"{run}.jsonl"


def judge_run_summary_path(root: Path, level: str, corpus: str, run: str) -> Path:
    return root / level / corpus / f"{run}.summary.json"


def parse_rows(spec: str) -> list[int]:
    """Line numbers of a file, counted from 1: "1-500,812,901-950" -> 1..500, 812, 901..950, in order, each once."""
    rows: list[int] = []
    for part in spec.split(","):
        first, dash, last = part.strip().partition("-")
        start, stop = int(first), int(last if dash else first)
        if start < 1 or stop < start:
            raise ValueError(f"a row range counts from 1 upwards: {part!r}")
        rows.extend(range(start, stop + 1))
    return list(dict.fromkeys(rows))


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
