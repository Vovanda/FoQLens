"""The answers at every layout read as one table (foqlens.runs): how often a layout's reply is word for word the reply
of the whole network at the top rung, per corpus too, and what the layout costs in bytes (the summaries the answers
were written with). No judge and no model - the files as they lie.

    uv run python scripts/precision_answers.py --answers runs/oracles/e2b-it/answers \
        --summaries runs/oracles/e2b-it --reference everything-d8-bartowski-Q2_K --out runs/oracles/e2b-it/answers.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foqlens.io import write_json
from foqlens.runs import RunFiles


def bytes_of(summaries: Path, label: str) -> dict | None:
    """What a layout spent, from the summary files of that label (the shards joined by their mean)."""
    found = sorted(summaries.glob(f"summary-{label}*.json"))
    if not found:
        return None
    read = [json.loads(p.read_text(encoding="utf-8")) for p in found]
    mean = sum(r["bytes"]["per_question_mean"] for r in read) / len(read)
    uniform = read[0]["bytes"]["uniform"]
    return {"per_question_mean": mean, "of_uniform_d8": mean / uniform["d8"], "of_uniform_d4": mean / uniform["d4"],
            "of_uniform_d2": mean / uniform["d2"]}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True, help="where the summary-*.json of the run lie")
    parser.add_argument("--reference", required=True, help="the label every other is compared to")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    files = RunFiles(args.answers)
    rows = files.same_reply(args.reference)
    by_corpus = files.same_reply_by_corpus(args.reference)
    table = []
    for row in rows:
        table.append(row | {"bytes": bytes_of(args.summaries, row["level"]),
                            "by_corpus": {r["corpus"]: r["same"] for r in by_corpus if r["level"] == row["level"]}})
    found = {"reference": args.reference, "layouts": table}
    write_json(args.out, found)
    for row in table:
        share = row["bytes"]["of_uniform_d8"] if row["bytes"] else float("nan")
        print(f"{row['level']:<60} same {row['same']:.3f} (n {row['n']}) bytes {share:.3f} of D8")
    return args.out


if __name__ == "__main__":
    main()
