"""E011, what a depth cap costs (experiments/E011-depth-caps/PREREG.md).

A layout is read per query, but storage is one copy: a block keeps the deepest slice any query of the
set asks of it. This measures that storage against the number of questions and against their variety,
for the graded zone layouts of E010 - the reading is unchanged by a cap, so quality is not evaluated
again and the only GPU work is the mask pass.

Writes runs/E011-depth-caps/<model>/summary.json.

    uv run python scripts/depth_caps.py
    uv run python scripts/depth_caps.py --limit 8 --out /tmp/caps    # smoke check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from foqlens import budget as bg
from foqlens import model as fm
from foqlens.gpu_share import default_share
from foqlens.io import read_questions, write_json
from foqlens.layouts import FixedZones, OwnZones, RandomZones, TopicMeans, TopicZones, graded_zone_layout
from foqlens.pipeline import GRADIENT_BATCH, POOLED_BATCH, Bench, subtract_background
from foqlens.quant import Level
from foqlens.topics import PAIR_NAMES, PAIRS, SPECS
from foqlens.weight_map import coactivation_map
from foqlens.zones import find_zones

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
MASK_SOURCE = "gradient"
FLOORS = {"d4": Level.D4, "zero": Level.ZERO}
FOCUS_AREAS = [0.2, 0.5, 0.8]
SUBSET_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 395]
DRAWS = 20  # draws per subset size; the summary keeps the mean and the range
FULL_COPY_BITS = float(Level.D8.bits)  # what storing every slice of every block costs
SAVING = 0.3  # the saving S3 asks about


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--floor", nargs="+", choices=sorted(FLOORS), default=sorted(FLOORS))
    parser.add_argument("--focus-area", nargs="+", type=float, default=FOCUS_AREAS)
    parser.add_argument("--focus-strength", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="questions per topic, for a smoke check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-batch", type=int, default=POOLED_BATCH)
    parser.add_argument("--gradient-batch", type=int, default=GRADIENT_BATCH)
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--out", type=Path, default=Path("runs/E011-depth-caps"))
    return parser.parse_args(argv)


def bits_per_code() -> np.ndarray:
    """Bits of every level code, as a table indexed by the code itself."""
    table = np.zeros(256)
    for level in Level:
        table[int(level)] = level.bits
    return table


def mean_bits(codes: np.ndarray, weights: np.ndarray, bits: np.ndarray) -> float:
    """Bits per weight of one layout or one set of caps, weighted by block size."""
    return float(bits[codes] @ weights / weights.sum())


def caps_of(levels: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The cap of every block over a subset of questions: the deepest level any of them reads.

    Level codes grow with depth (quant.Level), so the deepest read is the largest code.
    """
    return levels[rows].max(axis=0)


def storage_curve(levels: np.ndarray, weights: np.ndarray, bits: np.ndarray, pool: np.ndarray,
                  sizes: list[int], rng: np.random.Generator) -> list[dict]:
    """Storage against the number of questions drawn from `pool`, DRAWS draws per size."""
    out = []
    for k in sizes:
        if k > len(pool):
            continue
        drawn = [mean_bits(caps_of(levels, rng.choice(pool, size=k, replace=False)), weights, bits) for _ in range(DRAWS)]
        out.append({"k": k, "mean": float(np.mean(drawn)), "min": float(np.min(drawn)), "max": float(np.max(drawn))})
    return out


def first_k_below(curve: list[dict], limit: float) -> int | None:
    """The largest k whose mean storage is still at or below `limit`; None if even k = 1 is above it."""
    below = [row["k"] for row in curve if row["mean"] <= limit]
    return max(below) if below else None


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    questions = [q for spec in SPECS.values() for q in read_questions(args.prompts_dir, spec, args.limit)]
    domains = np.array([q.domain for q in questions])
    bench = Bench.load(MODELS[args.model], gpu_share=args.gpu_share)
    raw = bench.masks([q.prompt for q in questions], [bench.sources(args.pooled_batch, args.gradient_batch)[1]])
    coords = coactivation_map(raw[MASK_SOURCE])
    topics = TopicZones(TopicMeans(subtract_background(raw)[MASK_SOURCE], domains.tolist()), PAIRS, coords)
    backbone = find_zones(raw[MASK_SOURCE].mean(axis=0), coords)
    weights = bg.block_weights(bench.ctl).astype(np.float64)
    bits = bits_per_code()
    index = np.arange(len(questions))
    rng = np.random.default_rng(args.seed)
    sizes = [k for k in SUBSET_SIZES if k <= len(questions)] or [len(questions)]
    limit = FULL_COPY_BITS * (1 - SAVING)

    sets = {"all": index} | {f"topic:{d}": index[domains == d] for d in dict.fromkeys(domains.tolist())}
    sets |= {f"pair:{'-'.join(pair)}": index[np.isin(domains, pair)] for pair in PAIR_NAMES}

    cells = {}
    for floor_name in args.floor:
        floor = FLOORS[floor_name]
        halo = floor is Level.ZERO
        for area in args.focus_area:
            def layout(kind: str, source):
                return graded_zone_layout(kind, source, area, args.focus_strength, coords, floor=floor, halo=halo)

            own = layout("own", OwnZones(topics)).levels(index)
            cell = {
                "read_bits": float(np.mean([mean_bits(row, weights, bits) for row in own])),
                "storage_all": mean_bits(caps_of(own, index), weights, bits),
                "curves": {name: storage_curve(own, weights, bits, rows, sizes, rng) for name, rows in sets.items()},
                "references": {
                    "backbone": mean_bits(caps_of(layout("backbone", FixedZones(backbone)).levels(index), index), weights, bits),
                    "random": mean_bits(caps_of(layout("random", RandomZones(topics, args.seed)).levels(index), index), weights, bits),
                    "full_copy": FULL_COPY_BITS,
                },
            }
            cell["k_below_saving"] = {name: first_k_below(curve, limit) for name, curve in cell["curves"].items()}
            cells[f"{floor_name}|fa{area:.2f}"] = cell
            print(f"{floor_name:4} fa{area:.2f}: reads {cell['read_bits']:.2f}, stores {cell['storage_all']:.2f} "
                  f"of {FULL_COPY_BITS:.0f} bits; k under {limit:.1f} bits: "
                  + ", ".join(f"{n.removeprefix('topic:')} {k}" for n, k in cell["k_below_saving"].items() if not n.startswith("pair:")),
                  flush=True)

    out_dir = args.out / args.model
    summary = {
        "model": MODELS[args.model],
        "revision": fm.REVISIONS[MODELS[args.model]],
        "questions": {d: int((domains == d).sum()) for d in dict.fromkeys(domains.tolist())},
        "focus_strength": args.focus_strength,
        "subset_sizes": sizes,
        "draws": DRAWS,
        "saving": SAVING,
        "cells": cells,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"-> {out_dir / 'summary.json'}")
    return out_dir / "summary.json"


if __name__ == "__main__":
    main()
