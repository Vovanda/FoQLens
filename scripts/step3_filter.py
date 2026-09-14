"""E010, the graded zone layout (experiments/E010-lens-layout/ADDENDUM-11.md; mechanism: docs/quantization-filter.md).

The whole network sits at a floor level; the expert zones of a query are lifted over it up to a
ceiling, along an even profile. There is no budget: what a layout costs is what its zones ask for.

The weight map comes from the raw gradient masks of all questions of the run (co-activation, no
labels). In every cell of floor x focus area x focus strength four layouts are compared: the query's
own zones, the paired topic's zones, the backbone's zones (one set for every question), and the own
layout's levels shuffled over the blocks - the same memory with no mask at all. References, once:
bf16, uniform D4 / D6 / D8, and each floor alone.

The cells run from the most promising to the least and the summary is written after every cell, so a
stopped run still leaves what it has done.

Writes runs/E010-lens-layout/<model>/summary.json and raw per-question results.

    uv run python scripts/step3_filter.py
    uv run python scripts/step3_filter.py --limit 4 --focus-area 0.5 --out /tmp/filter   # smoke check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from foqlens import budget as bg
from foqlens import model as fm
from foqlens.evaluate import LetterChoice
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.io import read_questions, write_json
from foqlens.layouts import FixedZones, MovedZones, OtherZones, OwnZones, RandomZones, ShuffledLevels, TopicMeans, TopicZones, Uniform, graded_zone_layout
from foqlens.pipeline import GRADIENT_BATCH, POOLED_BATCH, Bench, subtract_background
from foqlens.progress import Progress
from foqlens.quality import batches, evaluate_all, layout_pool, submit_layouts, summarize
from foqlens.quant import Level
from foqlens.stats import paired_bootstrap
from foqlens.topics import PAIR_NAMES, PAIRS, SPECS
from foqlens.weight_map import coactivation_map
from foqlens.zones import find_zones

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
METRIC = LetterChoice  # the quality metric; layouts are compared on METRIC.primary
MASK_SOURCE = "gradient"  # the only source here: pooled zones failed in ADDENDUM-07 and E009
# The grid of ADDENDUM-11: three floors x focus areas that halve or double the radius x two strengths.
FLOORS = {"d4": Level.D4, "d2": Level.D2, "zero": Level.ZERO}
FOCUS_AREAS = [0.2, 0.333, 0.5, 0.667, 0.8]
FOCUS_STRENGTHS = [0.5, 1.0]
# Where an address is most likely to show: the zones as found, read to the top of the ladder.
PROMISING = (0.5, 1.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--floor", nargs="+", choices=sorted(FLOORS), default=["d4", "d2", "zero"], help="the level everything outside the zones is read at")
    parser.add_argument("--focus-area", nargs="+", type=float, default=FOCUS_AREAS, help="the size of the zones, 0 ... 1")
    parser.add_argument("--focus-strength", nargs="+", type=float, default=FOCUS_STRENGTHS, help="how far a zone's center rises above the floor, 0 ... 1")
    parser.add_argument("--combine", choices=("sum", "max"), default="sum", help="how the lifts of overlapping zones combine")
    parser.add_argument("--random-zones", action="store_true",
                        help="the floor of the comparison: zones of the same count and radii around random blocks, run on its own")
    parser.add_argument("--moved-zones", action="store_true",
                        help="the honest control: the query's own zones carried elsewhere on the map, at the same cost")
    parser.add_argument("--pair", nargs=2, metavar=("TOPIC", "TOPIC"), default=None,
                        help="run on this pair of topics instead of the two of topics.py; a name is a file in prompts/ or prompts/heldout/")
    parser.add_argument("--limit", type=int, default=None, help="questions per topic, for a smoke check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-batch", type=int, default=POOLED_BATCH)
    parser.add_argument("--gradient-batch", type=int, default=GRADIENT_BATCH)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--gpu-share", type=float, default=default_share(), help="share of the GPU the run takes (foqlens/gpu_share.py)")
    parser.add_argument("--out", type=Path, default=Path("runs/E010-lens-layout"))
    return parser.parse_args(argv)


def cells_by_promise(floors: list[str], areas: list[float], strengths: list[float]) -> list[tuple[str, float, float]]:
    """Every (floor, focus area, focus strength) cell, nearest to PROMISING first, D4 floor before D2 before ZERO."""
    area0, strength0 = PROMISING
    order = {name: i for i, name in enumerate(["d4", "d2", "zero"])}
    cells = [(f, a, g) for f in floors for a in areas for g in strengths]
    return sorted(cells, key=lambda c: (order[c[0]], abs(c[1] - area0) / area0 + abs(c[2] - strength0) / strength0))


def cell_name(kind: str, floor: str, area: float, strength: float) -> str:
    return f"zones_{kind}_{floor}_fa{area:.2f}_fs{strength:.2f}"


def lifted_share(policy, index: np.ndarray, floor: Level, weights: np.ndarray) -> dict:
    """How much of the net a cell moves: the share of blocks lifted over the floor, by count and by weight."""
    levels = policy.levels(index)
    lifted = levels != int(floor)
    return {"blocks": float(lifted.mean()), "weight": float(weights[lifted.any(axis=0)].sum() / weights.sum())}


def cell_policies(cell, coords, topics, backbone_zones, weights, seed, random_zones: bool = False,
                  moved_zones: bool = False) -> list:
    """The layouts of one cell: own zones, the paired topic's, the backbone's and own shuffled - or the random floor."""
    floor_name, area, strength = cell
    floor = FLOORS[floor_name]
    halo = floor is Level.ZERO  # the lowest rung goes past the edge, softening the step into nothing
    def layout(kind: str, source):
        return graded_zone_layout(cell_name(kind, floor_name, area, strength), source, area, strength, coords,
                                  floor=floor, halo=halo)

    own = layout("own", OwnZones(topics))
    if moved_zones:
        reach = area / (1 - area) if area < 1 else 1.0
        return [own, layout("moved", MovedZones(topics, weights, reach, seed))]
    if random_zones:
        return [own, layout("random", RandomZones(topics, seed))]
    return [own, layout("other", OtherZones(topics)), layout("backbone", FixedZones(backbone_zones)),
            ShuffledLevels(cell_name("nomask", floor_name, area, strength), own, weights, seed)]


def comparisons(results: dict, domains: tuple[str, ...], cells, seed: int, random_zones: bool = False,
                moved_zones: bool = False, pair_names: tuple = PAIR_NAMES) -> dict:
    """The predictions of ADDENDUM-11 per pair and cell: L2 the address, L3 no mask, L6 the backbone, H4 against bf16.

    With random zones the run holds only the floor of the comparison, L1 and L4: own against random.
    """
    lp = {name: np.array([r[METRIC.primary] for r in rows]) for name, rows in results.items()}
    out = {}
    for pair in pair_names:
        m = np.array([d in pair for d in domains])
        if not m.any():
            continue
        for floor, area, strength in cells:
            own = lp[cell_name("own", floor, area, strength)][m]
            key = f"{'-'.join(pair)}|{floor}|fa{area:.2f}|fs{strength:.2f}"
            if moved_zones:
                moved = lp[cell_name("moved", floor, area, strength)][m]
                acc = {name: np.array([r["accuracy"] for r in rows]) for name, rows in results.items()}
                out[key] = {
                    "own_minus_moved": paired_bootstrap(own, moved, seed=seed),
                    "own_minus_moved_accuracy": paired_bootstrap(acc[cell_name("own", floor, area, strength)][m],
                                                                 acc[cell_name("moved", floor, area, strength)][m], seed=seed),
                }
                continue
            if random_zones:
                out[key] = {"own_minus_random": paired_bootstrap(own, lp[cell_name("random", floor, area, strength)][m], seed=seed)}
                continue
            out[key] = {
                "own_minus_other": paired_bootstrap(own, lp[cell_name("other", floor, area, strength)][m], seed=seed),
                "own_minus_nomask": paired_bootstrap(own, lp[cell_name("nomask", floor, area, strength)][m], seed=seed),
                "own_minus_backbone": paired_bootstrap(own, lp[cell_name("backbone", floor, area, strength)][m], seed=seed),
                "own_minus_floor": paired_bootstrap(own, lp[f"floor_{floor}"][m], seed=seed),
                "own_minus_bf16": paired_bootstrap(own, lp["uniform_bf16"][m], seed=seed),
            }
    return out


def topic_setup(pair: list[str] | None, prompts_dir: Path) -> tuple[dict, dict, tuple]:
    """The topics of a run: the pairs of topics.py, or one pair named on the command line.

    A topic is a file of questions; the bench keeps some under heldout/, so a name is looked up in
    both places. The pair is what background subtraction and the comparisons are built from.
    """
    if pair is None:
        return SPECS, PAIRS, PAIR_NAMES
    specs = {}
    for name in pair:
        spec = name if (prompts_dir / f"{name}.jsonl").exists() else f"heldout/{name}"
        if not (prompts_dir / f"{spec}.jsonl").exists():
            raise SystemExit(f"no questions for topic {name!r} in {prompts_dir}")
        specs[name] = spec
    return specs, {pair[0]: pair[1], pair[1]: pair[0]}, ((pair[0], pair[1]),)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    with layout_pool() as pool:  # started first: its imports run while the model loads
        return run(args, pool)


def run(args: argparse.Namespace, pool) -> Path:
    specs, pairs, pair_names = topic_setup(args.pair, args.prompts_dir)
    questions = [q for spec in specs.values() for q in read_questions(args.prompts_dir, spec, args.limit)]
    domains = tuple(q.domain for q in questions)
    bench = Bench.load(MODELS[args.model], gpu_share=args.gpu_share)

    with GpuMonitor() as mask_gpu:
        raw = bench.masks([q.prompt for q in questions], bench.sources(args.pooled_batch, args.gradient_batch))
    coords = coactivation_map(raw[MASK_SOURCE])
    topics = TopicZones(TopicMeans(subtract_background(raw)[MASK_SOURCE], domains), pairs, coords)
    backbone_zones = find_zones(raw[MASK_SOURCE].mean(axis=0), coords)
    weights = bg.block_weights(bench.ctl)
    metric = METRIC.for_tokenizer(bench.tokenizer)
    out_dir = args.out / args.model

    groups = list(batches(len(questions), args.eval_batch))

    def evaluate(policies: list, queued: list) -> dict:
        return evaluate_all(bench.model, bench.tokenizer, bench.ctl, questions, policies, metric, args.eval_batch,
                            throttle=bench.throttle, pool=pool, layouts=queued)

    references = [Uniform(Level.BF16, bench.ctl.n_blocks), Uniform(Level.D4, bench.ctl.n_blocks),
                  Uniform(Level.D6, bench.ctl.n_blocks), Uniform(Level.D8, bench.ctl.n_blocks)]
    # each floor alone, without any zones: focus strength 0 is exactly that
    references += [graded_zone_layout(f"floor_{name}", OwnZones(topics), 0.5, 0.0, coords, floor=FLOORS[name])
                   for name in args.floor]
    zones_per_question = {MASK_SOURCE: float(np.mean([len(topics.own(i).radii) for i in range(len(questions))])),
                          "backbone": len(backbone_zones.radii)}

    with GpuMonitor() as eval_gpu:
        cells = cells_by_promise(args.floor, args.focus_area, args.focus_strength)
        cell_pols = [cell_policies(c, coords, topics, backbone_zones, weights, args.seed, args.random_zones, args.moved_zones)
                     for c in cells]
        # the worker queue always holds the next cell, so no cell starts by waiting for its layouts
        ref_queue = submit_layouts(pool, references, groups)
        queued = submit_layouts(pool, cell_pols[0], groups) if cells else []
        results = evaluate(references, ref_queue)
        done: list[tuple[str, float, float]] = []
        comps: dict = {}
        strength: dict = {}
        progress = Progress(len(cells), "cell")
        for k, cell in enumerate(cells):
            upcoming = submit_layouts(pool, cell_pols[k + 1], groups) if k + 1 < len(cells) else []
            results |= evaluate(cell_pols[k], queued)
            queued = upcoming
            done.append(cell)
            comps |= comparisons(results, domains, [cell], args.seed, args.random_zones, args.moved_zones, pair_names)  # only the new cell
            strength[cell_name("own", *cell)] = lifted_share(cell_pols[k][0], np.arange(len(questions)), FLOORS[cell[0]], weights)
            summary = {
                "model": MODELS[args.model],
                "revision": fm.REVISIONS[MODELS[args.model]],
                "questions": {d: domains.count(d) for d in specs},
                "topics": list(specs),
                "floor": args.floor,
                "focus_area": args.focus_area,
                "focus_strength": args.focus_strength,
                "combine": args.combine,
                "random_zones": args.random_zones,
                "moved_zones": args.moved_zones,
                "cells_done": [list(c) for c in done],
                "zones_per_question": zones_per_question,
                "gpu_share": args.gpu_share,
                "gpu": {"masks": mask_gpu.summary()},
                "comparisons": comps,
                "lifted": strength,
                "configs": summarize(results, questions),
            }
            write_json(out_dir / "summary.json", summary)
            print(progress.step(f"{cell[0]} fa{cell[1]:.2f} fs{cell[2]:.2f}"), flush=True)
    summary["gpu"]["eval"] = eval_gpu.summary()
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "raw" / "per_question.json", {"questions": [q.__dict__ for q in questions], "results": results})
    np.save(out_dir / "raw" / "weight_map.npy", coords)
    for key, c in summary["comparisons"].items():
        line = "  ".join(f"{k.removeprefix('own_minus_')} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" for k, v in c.items())
        print(f"{key:34} {line}")
    print(f"zones {summary['zones_per_question']}  gpu {summary['gpu']}\n-> {out_dir / 'summary.json'}")
    return out_dir / "summary.json"


if __name__ == "__main__":
    main()
