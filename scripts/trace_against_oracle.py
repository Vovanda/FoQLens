"""One question's pass against the oracles' maps of the same question: what the raised groups look like in the pass.

The trace holds what every layer was given and what its blocks answered (scripts/trace_question.py); the fields hold
the rung every oracle reads for every group. Both are on the host, so a candidate for the signal of the regulator is
tried here over arrays, as many as one wants, without the model.

Candidates so far, per group: what the group put out (`response`), that times the norm of the error the rungs make in
it (`response x gap`), and the score the layer-wise regulator uses today (`activity`).

    uv run python scripts/trace_against_oracle.py --trace runs/traces/e2b-it/tc_1516-bartowski-Q2_K-d2.npz \\
        --fields "runs/oracles/e2b-it/precision-fields-*-shard1of5.npz" --question tc_1516
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from foqlens.group_oracle import RUN_FIELDS
from foqlens.io import read_npz_parts
from foqlens.layerwise import Activity
from foqlens.oracle_overlay import block_group_ids, to_groups
from foqlens.quant import Level


def candidates(trace: np.lib.npyio.NpzFile, groups: list[str]) -> dict[str, np.ndarray]:
    """Every candidate signal of the pass, summed into the oracles' groups: name -> [groups].

    The blocks meet the groups through the bench's own rule (oracle_overlay.block_group_ids), and `activity` is the
    layer-wise regulator's own score (layerwise.Activity), not a second copy of it written here.
    """
    modules = trace["modules"].tolist()
    layers = np.array([int(name.split(".")[1]) for name in modules])
    kinds = np.array([name.split(".", 2)[2] for name in modules])
    blocks = [trace[f"gap_{i}"].shape[0] for i in range(len(modules))]
    at_of = block_group_ids(np.repeat(layers, blocks), np.repeat(kinds, blocks), groups)
    per_block = {name: [] for name in ("response", "response x gap", "activity")}
    for i, module in enumerate(modules):
        response = trace[f"response_{i}"].mean(axis=0)  # [blocks], the mean over the question's tokens
        gap, norms, weights = trace[f"gap_{i}"], trace[f"norms_{i}"], trace[f"weights_{i}"]
        state = torch.as_tensor(trace[f"state_{int(module.split('.')[1])}"])[None]
        static = torch.as_tensor(norms ** 2 / weights)
        per_block["response"].append(response)
        per_block["response x gap"].append(response * gap)
        per_block["activity"].append(Activity().scores(None, state, static).numpy()[0])
    return {name: to_groups(np.concatenate(rows)[None], at_of, len(groups))[0] for name, rows in per_block.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--fields", required=True, help="the oracles' precision fields (a glob joins shards)")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top", type=int, default=12, help="how many groups of the ranking to print")
    args = parser.parse_args(argv)

    head = read_npz_parts(args.fields, RUN_FIELDS | {"sources"}, only={"ids", "sources"})
    sources = head["sources"].tolist()
    z = read_npz_parts(args.fields, RUN_FIELDS | {"sources"},
                       only={"ids", "sources", "groups"} | {f"levels_{s}" for s in sources})
    ids, groups = z["ids"].tolist(), z["groups"].tolist()
    if args.question not in ids:
        raise SystemExit(f"no {args.question} in the fields: {len(ids)} questions")
    q = ids.index(args.question)
    levels = {s: z[f"levels_{s}"][q] for s in sources}
    raised = {s: np.array([code != int(Level.D2) and code != 255 for code in levels[s]]) for s in sources}

    trace = np.load(args.trace)
    signals = candidates(trace, groups)

    print(f"{args.question}: {len(groups)} groups, oracles {', '.join(sources)}")
    print(f"{'oracle':20} {'raised':>7} " + " ".join(f"{name:>16}" for name in signals))
    for s in sources:
        if not raised[s].any():
            print(f"{s:20} {0:>7}")
            continue
        parts = []
        for name, value in signals.items():
            up, base = value[raised[s]], value[~raised[s]]
            parts.append(f"{up.mean() / base.mean():>16.2f}" if base.mean() else f"{'-':>16}")
        print(f"{s:20} {int(raised[s].sum()):>7} " + " ".join(parts))
    print("\nthe ratio of a candidate's mean over the groups an oracle raised to its mean over the rest: 1.00 means "
          "the candidate does not tell them apart at all on this question")

    order = np.argsort(-signals["response x gap"])
    print(f"\nthe {args.top} groups the strongest candidate puts highest, and what every oracle read there:")
    print(f"{'group':14} {'resp x gap':>12} " + " ".join(f"{s[:8]:>9}" for s in sources))
    for at in order[: args.top]:
        rungs = " ".join(f"{Level(int(levels[s][at])).name if levels[s][at] != 255 else '.':>9}" for s in sources)
        print(f"{groups[at]:14} {signals['response x gap'][at]:>12.4g} {rungs}")


if __name__ == "__main__":
    main()
