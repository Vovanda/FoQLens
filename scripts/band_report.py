"""The grid of field scales, read back from its answers: every oracle's own scale and the uniform one (foqlens.maps).

scripts/band_grid.py writes every point of the grid as a layout and scripts/oracle_answers.py answers them all. This
reads those answers against the top rung's, question by question (foqlens.runs), counts what each point held - the
questions and the hard ones among them - and names the cheapest point that holds the answer for each oracle, then
the one scale whose worst loss over the oracles is the smallest.

    uv run python scripts/band_report.py --grid runs/E006-oracle-masks-that-hold/grid \
        --hard runs/E006-oracle-masks-that-hold/sample/hard.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from foqlens.maps import ScalePoint, least_worst_scale, own_scale
from foqlens.quant import Level
from foqlens.runs import RunFiles

# a layout of the grid: precision-<oracle>_b<bounds in hundredths>-<base>-t<tolerance>
POINT = re.compile(r"^precision-(?P<oracle>.+)_b(?P<b4>\d{3})(?P<b6>\d{3})(?P<b8>\d{3})-")


def run_cost(grid: Path) -> dict[str, float]:
    """What each layout of the run actually read, as a share of the top rung's - from the summaries beside it.

    cost.json holds the nominal bits of a layout (the base at 2 bits of 8 is 0.25); what the bench reads is not
    that, because the sensitive module classes lie on Q4_K and the base costs 0.342 of D8. Only the summaries say
    what a layout cost, and they are the ones comparable with the flat rungs.
    """
    costs = {}
    for summary in grid.glob("summary-*.json"):
        held = json.loads(summary.read_text(encoding="utf-8"))
        level = summary.name[len("summary-"):-len(".json")]
        costs[level] = round(held["bytes"]["per_question_mean"] / held["bytes"]["uniform"]["d8"], 4)
    return costs


def read_points(grid: Path, hard: set[tuple[str, str]], reference: str) -> list[ScalePoint]:
    """Every point of the grid as it answered: the oracle, its bounds, its cost and what it held."""
    cost = json.loads((grid / "cost.json").read_text(encoding="utf-8"))
    held: dict[str, list[tuple[str, str, int]]] = {}
    for row in RunFiles(grid / "answers").same_reply_by_question(reference):
        held.setdefault(str(row["level"]), []).append((str(row["corpus"]), str(row["id"]), int(row["same"])))
    points = []
    for level, answers in held.items():
        found = POINT.match(level)
        if not found:
            continue
        bounds = tuple(int(found.group(b)) / 100 for b in ("b4", "b6", "b8"))
        name = found.group("oracle")
        points.append(ScalePoint(oracle=name, bounds=bounds, cost=cost[f"{name}_b" + level.split("_b")[1][:9]],
                                 answered=sum(same for _, _, same in answers),
                                 hard=sum(same for corpus, qid, same in answers if (corpus, qid) in hard)))
    return points


def rung_shares(grid: Path, oracle: str, bounds: tuple[float, float, float]) -> dict[str, float]:
    """What share of the groups a scale leaves on each rung - the layout the grid answered, read back from its file.

    A scale that puts half the network on D4 is not the base's scale: the point of the base is that most of the
    network stays on it, and the shares say whether a cheap point is cheap for that reason.
    """
    name = "levels_%s_b%03d%03d%03d" % (oracle, *[round(b * 100) for b in bounds])
    kept = np.load(next(grid.glob("precision-fields-*.npz")), allow_pickle=True)
    levels = kept[name]
    return {rung.name.lower(): round(float((levels == int(rung)).mean()), 3)
            for rung in (Level.D2, Level.D4, Level.D6, Level.D8)}


def matrix(grid: Path, hard: set[tuple[str, str]], searched: set[tuple[str, str]], reference: str) -> list[dict]:
    """Every layout of a run against the top rung: what it cost and what it held, field by field and scale by scale.

    The questions of the search are counted apart from the rest: a scale found on ten questions and shown on a
    hundred has to say what it does on the ninety it never saw, or the number is its own search read back.
    """
    kept = np.load(next(grid.glob("precision-fields-*.npz")), allow_pickle=True)
    keys = list(zip(kept["corpus"].tolist(), kept["ids"].tolist()))
    cost = run_cost(grid)
    same: dict[str, dict[tuple[str, str], int]] = {}
    for row in RunFiles(grid / "answers").same_reply_by_question(reference):
        same.setdefault(str(row["level"]), {})[(str(row["corpus"]), str(row["id"]))] = int(row["same"])
    rows = []
    for key in sorted(k for k in kept.files if k.startswith("levels_")):
        name = key[len("levels_"):]
        field, bounds = name.rsplit("_b", 1)
        level = f"precision-{name}-bartowski-Q2_K-t0.03"
        levels = kept[key]
        # a map that lifted the whole network matches the top rung for free; those questions are counted apart
        whole = (levels == int(Level.D8)).all(axis=1)
        live = [q for q, w in zip(keys, whole) if not w]
        held = same[level]
        parts = {"all": live, "hard": [q for q in live if q in hard],
                 "searched": [q for q in live if q in searched], "rest": [q for q in live if q not in searched]}
        rows.append({"field": field, "bounds": tuple(int(bounds[i:i + 3]) / 100 for i in (0, 3, 6)),
                     "cost": cost[level], "degenerate": int(whole.sum()),
                     **{f"{part}_held": sum(held[q] for q in qs) for part, qs in parts.items()},
                     **{f"{part}_of": len(qs) for part, qs in parts.items()}})
    for rung in ("d2", "d4", "d6"):
        level = f"everything-{rung}-bartowski-Q2_K"
        held = same[level]
        rows.append({"field": f"flat {rung}", "bounds": None, "cost": cost[level], "degenerate": 0,
                     "all_held": sum(held.values()), "all_of": len(held),
                     "hard_held": sum(v for q, v in held.items() if q in hard),
                     "hard_of": sum(q in hard for q in held),
                     "searched_held": sum(v for q, v in held.items() if q in searched),
                     "searched_of": sum(q in searched for q in held),
                     "rest_held": sum(v for q, v in held.items() if q not in searched),
                     "rest_of": sum(q not in searched for q in held)})
    return sorted(rows, key=lambda r: -r["all_held"] / max(r["all_of"], 1))


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=Path, required=True, help="the grid run: answers/ and cost.json")
    parser.add_argument("--hard", type=Path, required=True, help="a json list of [corpus, id]: the hard questions")
    parser.add_argument("--reference", default="everything-d8-bartowski-Q2_K", help="the layout answered against")
    parser.add_argument("--answered", type=int, default=9, help="questions a scale must hold")
    parser.add_argument("--hard-held", type=int, default=5, help="hard questions a scale must hold")
    parser.add_argument("--show", type=int, default=6, help="points printed per oracle, cheapest first")
    args = parser.parse_args(argv)

    hard = {tuple(pair) for pair in json.loads(args.hard.read_text(encoding="utf-8"))}
    points = read_points(args.grid, hard, args.reference)
    by_oracle: dict[str, list[ScalePoint]] = {}
    for point in points:
        by_oracle.setdefault(point.oracle, []).append(point)

    scales = {}
    for name in sorted(by_oracle):
        group = sorted(by_oracle[name], key=lambda p: (p.cost, p.bounds))
        mine = own_scale(group, args.answered, args.hard_held)
        scales[name] = mine
        best = max(p.answered for p in group)
        # the oracle's best scale whatever the thresholds: most answers, then most hard, then cheapest
        top = max(group, key=lambda p: (p.answered, p.hard, -p.cost))
        shares = rung_shares(args.grid, name, top.bounds)
        print("\n%s: best %.2f / %.2f / %.2f  cost %.3f  answers %d/10  hard %d/5  "
              "base %.0f%% d4 %.0f%% d6 %.0f%% d8 %.0f%%"
              % (name, *top.bounds, top.cost, top.answered, top.hard,
                 *[100 * shares[r] for r in ("d2", "d4", "d6", "d8")]))
        for point in [p for p in group if p.answered >= best - 1][:args.show]:
            mark = " <- own" if point == mine else ""
            print("  %.2f / %.2f / %.2f  cost %.3f  answers %2d/10  hard %d/5%s"
                  % (*point.bounds, point.cost, point.answered, point.hard, mark))

    bounds, losses = least_worst_scale(points, args.answered, args.hard_held)
    print("\nuniform scale:", "none" if bounds is None else "%.2f / %.2f / %.2f" % bounds)
    for name in sorted(losses):
        lost, lost_hard, overpaid = losses[name]
        print(f"  {name:16} loses {lost} answers, {lost_hard} hard, overpays {overpaid:.3f}")
    return {"scales": scales, "uniform": bounds, "losses": losses}


if __name__ == "__main__":
    main()
