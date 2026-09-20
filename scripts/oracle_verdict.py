"""The oracles' answers cut by the shape of the layout they answered on (foqlens.field_analysis.shapes, foqlens.runs):
a question whose whole network sits at the base or at the top holds the reference reply for a reason that has nothing
to do with zones, and read together with the mapped questions it flatters every oracle. This prints the three groups
apart - how many questions, how often the reply is the whole network's, and what the layout spent.

    uv run python scripts/oracle_verdict.py --fields "runs/oracles/e2b-it/precision-fields-*-shard*of5.npz" \
        --answers runs/oracles/e2b-it/answers --summaries runs/oracles/e2b-it \
        --reference everything-d8-bartowski-Q2_K --out runs/oracles/e2b-it/oracle-verdict.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foqlens.field_analysis import shapes
from foqlens.group_oracle import RUN_FIELDS
from foqlens.io import read_npz_parts, write_json
from foqlens.quant import Level
from foqlens.runs import RunFiles

SHAPES = ("base", "map", "top")


def layout_bytes(summaries: Path, label: str) -> tuple[dict[tuple[str, str], int], int]:
    """Every question's bytes at a layout, from the summary files of that label, and the bytes of the uniform top."""
    per_question, top = {}, 0
    for path in sorted(summaries.glob(f"summary-{label}-shard*.json")) or sorted(summaries.glob(f"summary-{label}.json")):
        read = json.loads(path.read_text(encoding="utf-8"))
        top = read["bytes"]["uniform"]["d8"]
        for row in read["questions"]:
            per_question[(row["corpus"], row["id"])] = row["bytes"]
    return per_question, top


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", required=True, help="the precision fields' files (a glob joins shards)")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--reference", required=True, help="the label every other is compared to")
    parser.add_argument("--label", default="precision-{source}-bartowski-Q2_K-t0.02",
                        help="how a source's layout is named among the answers")
    parser.add_argument("--uniform", nargs="*", default=["everything-d2-bartowski-Q2_K", "everything-d4-bartowski-Q2_K",
                                                         "everything-d6-bartowski-Q2_K"],
                        help="the rungs every source's mapped questions are also read at, at the same questions")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    head = read_npz_parts(args.fields, RUN_FIELDS | {"sources"}, only={"sources"})
    sources = head["sources"].tolist() + ["reference", "common", "middle", "together"]
    z = read_npz_parts(args.fields, RUN_FIELDS | {"sources"}, only={"ids", "corpus", "sources"}
                       | {f"levels_{s}" for s in sources})
    keys = list(zip(z["corpus"].tolist(), z["ids"].tolist()))
    shape_of = {s: dict(zip(keys, shapes(z[f"levels_{s}"], int(Level.D2), int(Level.D8)).tolist())) for s in sources}

    run = RunFiles(args.answers)
    same = f"({RunFiles.SAME_FORM.format('a.reply')} = {RunFiles.SAME_FORM.format('r.reply')})::int"
    rows = run.query(
        rf"""select a.level, a.corpus, a.id, {same} same, r.exact_match ref_exact
             from answers a join answers r using (corpus, id) where r.level = ? and a.level != r.level""",
        [args.reference])

    report: dict[str, dict] = {"reference": args.reference, "layouts": {}}
    for source in sources:
        label = args.label.format(source=source)
        mine = [row for row in rows if row["level"] == label]
        if not mine:
            continue
        spent, top = layout_bytes(args.summaries, label)
        entry = {}
        for shape in SHAPES:
            part = [row for row in mine if shape_of[source].get((row["corpus"], row["id"])) == shape]
            if not part:
                continue
            correct = [row for row in part if row["ref_exact"]]
            cost = [spent[(row["corpus"], row["id"])] for row in part if (row["corpus"], row["id"]) in spent]
            entry[shape] = {
                "questions": len(part),
                "same_as_top": float(np.mean([row["same"] for row in part])),
                "questions_the_top_answers": len(correct),
                "same_where_the_top_is_right": float(np.mean([row["same"] for row in correct])) if correct else None,
                "bytes_of_uniform_top": float(np.mean(cost) / top) if cost and top else None,
            }
        # the same questions read at the uniform rungs: what a deployment would do instead of the map, so that the map
        # is judged against the ladder on its own questions and not on all of them
        mapped = {(row["corpus"], row["id"]) for row in mine
                  if shape_of[source].get((row["corpus"], row["id"])) == "map"}
        entry["uniform_on_the_mapped"] = {}
        for rung in args.uniform:
            spent, top = layout_bytes(args.summaries, rung)
            part = [row for row in rows if row["level"] == rung and (row["corpus"], row["id"]) in mapped]
            correct = [row for row in part if row["ref_exact"]]
            cost = [spent[(row["corpus"], row["id"])] for row in part if (row["corpus"], row["id"]) in spent]
            if part:
                entry["uniform_on_the_mapped"][rung] = {
                    "questions": len(part),
                    "same_where_the_top_is_right": float(np.mean([row["same"] for row in correct])) if correct else None,
                    "bytes_of_uniform_top": float(np.mean(cost) / top) if cost and top else None,
                }
        report["layouts"][source] = entry

    write_json(args.out, report)
    print(f"{'source':24s} {'shape':5s} {'n':>4s} {'right':>6s} {'held':>6s} {'bytes':>6s}")
    for source, entry in report["layouts"].items():
        for shape, row in entry.items():
            if shape == "uniform_on_the_mapped":
                for rung, at in row.items():
                    held = "-" if at["same_where_the_top_is_right"] is None else f"{at['same_where_the_top_is_right']:.3f}"
                    cost = "-" if at["bytes_of_uniform_top"] is None else f"{at['bytes_of_uniform_top']:.3f}"
                    print(f"{source:24s} {rung.split('-')[1]:5s} {at['questions']:4d} {'':6s} {held:>6s} {cost:>6s}")
                continue
            held = "-" if row["same_where_the_top_is_right"] is None else f"{row['same_where_the_top_is_right']:.3f}"
            cost = "-" if row["bytes_of_uniform_top"] is None else f"{row['bytes_of_uniform_top']:.3f}"
            print(f"{source:24s} {shape:5s} {row['questions']:4d} {row['questions_the_top_answers']:6d} "
                  f"{held:>6s} {cost:>6s}")
    return args.out


if __name__ == "__main__":
    main()
