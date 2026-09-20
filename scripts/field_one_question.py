"""One question's precision fields read side by side (foqlens.field_analysis), from the kept files: the level every
oracle gives every group, the band they leave it, and where their antinodes fall. The aggregate report averages all of
this over the questions; a regulator has to work on one question, and that is what this prints.

    uv run python scripts/field_one_question.py --fields "runs/oracles/e2b-it/precision-fields-*-shard*of5.npz" \
        --question tc_1516
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from foqlens.field_analysis import centres, jaccard
from foqlens.group_oracle import RUN_FIELDS
from foqlens.io import read_npz_parts, write_json
from foqlens.oracle_overlay import block_group_ids
from foqlens.precision_field import FIELD_VALUE, values
from foqlens.quant import Level
from foqlens.small_corpus import StoredMasks

BASE = FIELD_VALUE[Level.D2]
ORACLES = ["lift", "drop", "answer_gradient", "answer_quant_gap", "error_energy", "error_energy_d4", "pooled"]
DERIVED = ["reference", "common", "middle", "together"]
SHORT = {"lift": "lift", "drop": "drop", "answer_gradient": "grad", "answer_quant_gap": "qgap",
         "error_energy": "enD2", "error_energy_d4": "enD4", "pooled": "pool", "reference": "ref",
         "common": "comm", "middle": "midl", "together": "tgth"}


def rung(code: int) -> str:
    """A level code as the rung's digit, so that a row of levels reads as a profile: 2, 4, 6, 8, or `.` if unread."""
    return Level(code).name[1:] if code != 255 else "."


def bits_a_weight(levels: np.ndarray, weights: np.ndarray) -> float:
    """The nominal bits a weight of one question's layout [groups] of level codes."""
    return float(sum(Level(int(c)).bits * w for c, w in zip(levels, weights)) / weights.sum())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", required=True, help="the precision fields' files (a glob joins shards)")
    parser.add_argument("--weights", default="runs/masks/e2b-it/error_energy-file-bartowski-Q2_K-small-corpus.npz",
                        help="a kept masks file: the weights of the blocks, summed into groups")
    parser.add_argument("--question", nargs="*", default=[], help="the ids to print; none means --corpus's first")
    parser.add_argument("--corpus", default=None, help="take the first --take questions of this corpus instead")
    parser.add_argument("--take", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None, help="where to also write the printed maps as json")
    args = parser.parse_args(argv)

    head = read_npz_parts(args.fields, RUN_FIELDS | {"sources"}, only={"ids", "corpus", "sources"})
    sources = [s for s in head["sources"].tolist() if s in ORACLES]
    kept = [s for s in DERIVED if s not in sources]
    z = read_npz_parts(args.fields, RUN_FIELDS | {"sources"},
                       only={"ids", "corpus", "sources"} | {f"levels_{s}" for s in sources + kept})
    names = z["groups"].tolist()
    layers = np.array([int(n.split(".")[0]) for n in names])
    stored = StoredMasks.read(args.weights)
    weights = np.bincount(block_group_ids(stored.block_layer, stored.block_kind, names),
                          weights=stored.block_weights, minlength=len(names))
    ids, corpus = z["ids"].tolist(), z["corpus"]

    if args.question:
        picked = [ids.index(q) for q in args.question]
    else:
        where = np.flatnonzero(corpus == args.corpus) if args.corpus else np.arange(len(ids))
        picked = where[: args.take].tolist()

    shown = sources + [s for s in kept if f"levels_{s}" in z]
    out = {}
    for q in picked:
        levels = {s: z[f"levels_{s}"][q] for s in shown}
        found = {s: centres(values(levels[s]), layers, BASE) for s in shown}
        stack = np.stack([levels[s] for s in sources]).astype(int)
        low, high = stack.min(axis=0), stack.max(axis=0)

        print(f"\n== {ids[q]} ({corpus[q]}), the oracles {', '.join(sources)}")
        print("group".ljust(13) + "".join(SHORT[s].rjust(5) for s in shown) + "   band")
        for g, name in enumerate(names):
            row = "".join((rung(int(levels[s][g])) + ("*" if found[s][g] else " ")).rjust(5) for s in shown)
            band = f"{rung(int(low[g]))}-{rung(int(high[g]))}" if high[g] != low[g] else f" {rung(int(low[g]))} "
            print(name.ljust(13) + row + "   " + band)

        fixed = low == high
        summary = {
            "id": ids[q], "corpus": str(corpus[q]),
            "band": {"groups_fixed": int(fixed.sum()), "of_groups": len(names),
                     "fixed_above_base": int((fixed & (low > int(Level.D2))).sum()),
                     "raised_by_every_oracle": int((low > int(Level.D2)).sum()),
                     "raised_by_any_oracle": int((high > int(Level.D2)).sum()),
                     "bits_a_weight_lowest": bits_a_weight(low, weights),
                     "bits_a_weight_highest": bits_a_weight(high, weights)},
            "fields": {s: {"raised": int((levels[s] > int(Level.D2)).sum()), "antinodes": int(found[s].sum()),
                           "antinode_layers": sorted({int(layers[i]) for i in np.flatnonzero(found[s])}),
                           "bits_a_weight": bits_a_weight(levels[s], weights)} for s in shown},
            "antinodes_shared": {f"{a}|{b}": float(jaccard(found[a][None, :], found[b][None, :])[0])
                                 for i, a in enumerate(shown) for b in shown[i + 1:]},
        }
        print(f"\nband: {summary['band']['groups_fixed']} of {len(names)} groups fixed, "
              f"{summary['band']['raised_by_every_oracle']} raised by every oracle, "
              f"{summary['band']['raised_by_any_oracle']} by any; "
              f"bits a weight {summary['band']['bits_a_weight_lowest']:.2f}-"
              f"{summary['band']['bits_a_weight_highest']:.2f}")
        for s in shown:
            f = summary["fields"][s]
            print(f"  {SHORT[s]:5s} raised {f['raised']:3d}, antinodes {f['antinodes']:2d} at layers "
                  f"{f['antinode_layers']}, {f['bits_a_weight']:.2f} bits a weight")
        pairs = sorted(summary["antinodes_shared"].items(), key=lambda kv: -kv[1])
        print("  shared antinodes: " + ", ".join(f"{k} {v:.2f}" for k, v in pairs[:6]))
        out[ids[q]] = summary

    if args.out:
        write_json(args.out, out)
        print(f"\n{args.out}")


if __name__ == "__main__":
    main()
