"""Every candidate for the regulator's score, every correction and every pair of them, against every oracle - in one
pass over a written trace.

A trace is arrays (scripts/trace_question.py), so a formula over it costs nothing: the whole grid is computed at once
instead of one candidate at a time. What it prints is two things - how far each candidate stands from an oracle over
all the groups, and the walk forward, group by group, where it is visible who was near the oracle's own highest at
that point of the pass and who was lying.

The candidates are the score's half of a strategy; how a score becomes a rung is the reading's half and is not here.

    uv run python scripts/strategy_grid.py --trace runs/traces/e2b-it/<q>-bartowski-Q2_K-d2.npz \\
        --fields runs/oracles/e2b-it/unified-fields-shard1of5.npz --question <q>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from foqlens.layerwise import Activity
from foqlens.oracle_overlay import block_group_ids, overlay, to_groups

# how much of the depth's own growth is divided out: 0 leaves the score as it is, 1 divides by the norm of the stream
# the layer carries, and the middle values are between. The response grows with the stream, the stream grows with the
# depth, and an oracle's mass sits early - so the correction is a knob of the strategy, not a detail of the code.
DEPTHS = (0.0, 0.5, 1.0)
PAIRS = ("product", "mean")  # how two candidates are laid over one another, as the oracles' fields are
# For now a strategy is read as a boolean - the highest share to the top rung, the rest at the base (Volodya 21.09:
# that is fine to start with). What it owes next is the rung itself, D4 / D6 / D8, and with it the two directions of
# a miss counted apart: reading finer than the oracle is paid in bytes, coarser in the answer, and one number over
# both hides which of them the candidate is making.
#
# Measured 21.09 on one hard question: the exact injection ||(W_high - W_low)_b x||, one matmul a module, ranks the
# groups no better than the product of the two norms (0.73 against 0.75 on the energy oracle, 0.25 against 0.17 on
# the gradient) - the blocks differ by magnitudes, not by the directions agreeing, so the cheap product is enough
# and the matmul buys nothing.


def raw_candidates(trace: np.lib.npyio.NpzFile, groups: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Every candidate over the blocks, and the norm of the stream at every block's layer."""
    modules = trace["modules"].tolist()
    layers = np.array([int(name.split(".")[1]) for name in modules])
    kinds = np.array([name.split(".", 2)[2] for name in modules])
    blocks = [trace[f"gap_{i}"].shape[0] for i in range(len(modules))]
    at_of = block_group_ids(np.repeat(layers, blocks), np.repeat(kinds, blocks), groups)
    made: dict[str, list[np.ndarray]] = {"response": [], "damage": [], "share": [], "activity": []}
    streams = []
    for i, module in enumerate(modules):
        response = trace[f"response_{i}"].mean(axis=0)
        gap, norms, weights = trace[f"gap_{i}"], trace[f"norms_{i}"], trace[f"weights_{i}"]
        state = trace[f"state_{int(module.split('.')[1])}"]
        stream = float(np.linalg.norm(state, axis=-1).mean())
        made["response"].append(response)
        made["damage"].append(response * gap)
        made["share"].append(response / stream)
        made["activity"].append(
            Activity().scores(None, torch.as_tensor(state)[None], torch.as_tensor(norms ** 2 / weights)).numpy()[0])
        streams.append(np.full(len(response), stream))
    per_group = {name: to_groups(np.concatenate(rows)[None], at_of, len(groups))[0] for name, rows in made.items()}
    stream_of = to_groups(np.concatenate(streams)[None], at_of, len(groups))[0]
    return per_group, stream_of


def corrected(value: np.ndarray, stream: np.ndarray, power: float) -> np.ndarray:
    """The candidate with the depth's growth divided out to the given power."""
    return value / np.maximum(stream, 1e-9) ** power


def rank_of(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rho without scipy: the correlation of the ranks."""
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb) + 1e-30))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--share", type=float, default=0.1, help="the share of the groups an oracle's highest are")
    parser.add_argument("--oracles", nargs="*", default=["lift", "drop", "answer_gradient", "error_energy", "pooled"])
    args = parser.parse_args(argv)

    z = np.load(args.fields)
    ids, groups = z["ids"].tolist(), z["groups"].tolist()
    q = ids.index(args.question)
    fields = {s: z[f"field_{s}"][q] for s in args.oracles}
    trace = np.load(args.trace)
    plain, stream = raw_candidates(trace, groups)

    # One regulator with a vector of coefficients instead of a heap of strategies (Volodya 21.09): the score is a
    # product of the factors in powers, which in the logarithm is a straight sum. Every strategy written by hand is
    # then one row of the coefficients, and the fit below says which components an oracle actually wants non-zero.
    factors = {"response": plain["response"], "gap": plain["damage"] / np.maximum(plain["response"], 1e-30),
               "stream": stream, "activity": plain["activity"]}
    logs = np.stack([np.log(np.maximum(v, 1e-30)) for v in factors.values()], axis=1)
    logs = np.column_stack([logs - logs.mean(axis=0), np.ones(len(groups))])
    print(f"{args.question}: the coefficients every oracle wants, fitted on log of the factors")
    print(f"{'oracle':18} " + " ".join(f"{name:>10}" for name in factors) + f"{'fit':>8}")
    for name, values in fields.items():
        target = np.log(np.maximum(values, 1e-6))
        coefficients, *_ = np.linalg.lstsq(logs, target - target.mean(), rcond=None)
        made = logs @ coefficients
        print(f"{name:18} " + " ".join(f"{c:>10.2f}" for c in coefficients[:-1]) + f"{rank_of(made, values):>8.2f}")

    grid: dict[str, np.ndarray] = {}
    for name, value in plain.items():
        for power in DEPTHS:
            grid[f"{name}^{power:g}" if power else name] = corrected(value, stream, power)
    names = sorted(grid)
    for i, one in enumerate(names):  # every pair of the corrected candidates, laid over one another
        for other in names[i + 1:]:
            if one.split("^")[0] == other.split("^")[0]:
                continue  # the same candidate at two corrections is not a pair
            for how in PAIRS:
                a, b = (grid[k] / max(grid[k].max(), 1e-30) for k in (one, other))
                grid[f"{one} {how} {other}"] = overlay([a, b], how)

    print(f"{args.question}: {len(groups)} groups, {len(grid)} strategies against {len(fields)} oracles")
    print(f"{'strategy':44} " + " ".join(f"{o[:12]:>13}" for o in args.oracles) + "   best")
    rows = []
    for name, value in grid.items():
        rho = {o: rank_of(value, f) for o, f in fields.items()}
        rows.append((max(rho.values()), name, rho))
    for best, name, rho in sorted(rows, reverse=True)[:20]:
        print(f"{name:44} " + " ".join(f"{rho[o]:>13.2f}" for o in args.oracles) + f"   {best:.2f}")

    top = sorted(rows, reverse=True)[0][1]
    print(f"\nthe walk forward: who stood in an oracle's highest {args.share:.0%} at every group, for {top}")
    order = np.argsort([int(g.split(".")[0]) for g in groups], kind="stable")
    take = max(1, int(round(args.share * len(groups))))
    highest = {o: set(np.argsort(f)[-take:].tolist()) for o, f in fields.items()}
    mine = set(np.argsort(grid[top])[-take:].tolist())
    hits = {o: 0 for o in fields}
    for at in order:
        marks = "".join("+" if at in highest[o] else "." for o in args.oracles)
        for o in args.oracles:
            hits[o] += int(at in highest[o] and at in mine)
        if at in mine or marks.count("+"):
            print(f"  {groups[at]:14} {'mine' if at in mine else '    '}  oracles {marks}")
    print("  " + ", ".join(f"{o}: {hits[o]} of {take}" for o in args.oracles))


if __name__ == "__main__":
    main()
