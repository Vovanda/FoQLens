"""Build the run 1 query sets in prompts/ from pinned dataset revisions (see docs/data-sources.md).

Deterministic: the same revisions give the same files, byte for byte.

- prompts/<domain>.jsonl          debugging domains, from MMLU-Redux-2.0, error_type == "ok" only
- prompts/heldout/<domain>.jsonl  held-out domains - not opened and not run while the score is debugged
- prompts/calibration/*.jsonl     PAWS paraphrase pairs and random unrelated pairs
- prompts/biophysics.jsonl        hand-written, committed as is, not touched by this script

    uv run python scripts/build_prompts.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REDUX = "edinburgh-dawg/mmlu-redux-2.0"
REDUX_REVISION = "372ea425445d51e1ba1188c56e5e893f8138621f"
PAWS = "google-research-datasets/paws"
PAWS_REVISION = "161ece9501cf0a11f3e48bd356eaa82de46d6a09"

DEBUG = {
    "biology": "high_school_biology",
    "math": "high_school_mathematics",
    "chemistry": "high_school_chemistry",
    "physics": "high_school_physics",
}
HELDOUT = {
    "history": "prehistory",
    "geography": "high_school_geography",
}
CALIBRATION_PAIRS = 100
SEED = 0


def redux_questions(subject: str) -> list[dict]:
    path = hf_hub_download(REDUX, f"{subject}/data-00000-of-00001.arrow", repo_type="dataset", revision=REDUX_REVISION)
    with open(path, "rb") as f:
        table = pa.ipc.open_stream(f).read_all()
    rows = table.select(["question", "error_type"]).to_pylist()
    return [
        {"text": r["question"].strip(), "source": f"{REDUX}/{subject}@{REDUX_REVISION[:8]}", "row": i}
        for i, r in enumerate(rows)
        if r["error_type"] == "ok"
    ]


def paws_pairs() -> tuple[list[dict], list[dict]]:
    path = hf_hub_download(PAWS, "labeled_final/test-00000-of-00001.parquet", repo_type="dataset", revision=PAWS_REVISION)
    rows = pq.read_table(path).to_pylist()
    rng = random.Random(SEED)
    positives = [r for r in rows if r["label"] == 1]
    rng.shuffle(positives)
    paraphrases = [
        {"a": r["sentence1"], "b": r["sentence2"], "source": f"{PAWS}@{PAWS_REVISION[:8]}", "id": r["id"]}
        for r in positives[:CALIBRATION_PAIRS]
    ]
    # unrelated: first sentences of two different pairs, never from the same pair
    pool = rng.sample(rows, 2 * CALIBRATION_PAIRS)
    unrelated = [
        {"a": x["sentence1"], "b": y["sentence1"], "source": f"{PAWS}@{PAWS_REVISION[:8]}", "ids": [x["id"], y["id"]]}
        for x, y in zip(pool[::2], pool[1::2])
    ]
    return paraphrases, unrelated


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{path}: {len(items)}")


def main() -> None:
    out = Path("prompts")
    for domain, subject in DEBUG.items():
        write_jsonl(out / f"{domain}.jsonl", redux_questions(subject))
    for domain, subject in HELDOUT.items():
        write_jsonl(out / "heldout" / f"{domain}.jsonl", redux_questions(subject))
    paraphrases, unrelated = paws_pairs()
    write_jsonl(out / "calibration" / "paraphrase.jsonl", paraphrases)
    write_jsonl(out / "calibration" / "unrelated.jsonl", unrelated)


if __name__ == "__main__":
    main()
