"""Expert zones (experiments/E008-zones-fixed-budget/ADDENDUM-07.md; terms: ADDENDUM-08 to 10): a topic's zones against random zones.

The weight map comes from the raw gradient masks of all questions of the run (co-activation, no
labels). For every question, the expert zones of its own topic's mask and of the paired topic's mask
(background subtracted, leave-one-out) give a layout of D4 / D6 / D8 for every cell of precision share x
focus area; random zones keep the own zones' count and radii around random blocks; the precision
share spent evenly, without a mask, is the reference of its row. The generic importance backbone's zones are a reference too.

The cells run from the most promising to the least - nearest to PROMISING first - and the summary is
written after every cell, so a stopped run still leaves what it has done.

Writes runs/E008-zones-fixed-budget/<model>/summary.json and raw per-question results.

Measured 2026-09-12 on an RTX 3090 Ti, Gemma 4 E2B, with the layout worker, its queue across cells and
all slices unpacked in one go:
- one batch of 32 questions with per-question D4 / D6 / D8 layouts: 1.26 s (1.48 s before the one-go
  unpacking), GPU 93% busy; uniform D8: 0.72 s; bf16: 0.37 s - the slices are still unpacked on every batch;
- one fixed-budget layout: 26 ms of Python and numpy, built in the layout worker process;
- --limit 32 (128 questions), one cell: 95 s; evaluation utilization 89% (median 99%) at --gpu-share 1,
  80% (median 98%) at 0.8; masks 39-51%; with batches by tokens (pipeline.GRADIENT_BATCH) masks 38% at 0.8
  and 50% at 1, peak 16.6 GiB reserved at both - the lengths of these questions are close, and the mask
  pass stays bound by kernel launches;
- the full matrix of E009 (395 questions, 24 cells x 7 policies) took 1 h 42 min before these changes,
  at 40% evaluation utilization.

    uv run python scripts/step3_zones.py
    uv run python scripts/step3_zones.py --limit 4 --focus-area 0.5 --out /tmp/zones   # smoke check
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
from foqlens.layouts import FixedZones, NoZones, OtherZones, OwnZones, RandomZones, TopicMeans, TopicZones, Uniform, legacy_zone_layout
from foqlens.pipeline import GRADIENT_BATCH, MASK_SOURCES, POOLED_BATCH, Bench, subtract_background
from foqlens.progress import Progress
from foqlens.quality import batches, evaluate_all, layout_pool, submit_layouts, summarize
from foqlens.quant import Level
from foqlens.stats import paired_bootstrap
from foqlens.topics import PAIR_NAMES, PAIRS, SPECS
from foqlens.weight_map import coactivation_map
from foqlens.zones import find_zones

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
METRIC = LetterChoice  # the quality metric; layouts are compared on METRIC.primary
# The matrix of ADDENDUM-09: 4.5 / 5 / 6 / 7 bits x focus areas that halve or double the radius (f / (1 - f) = 0 ... 4).
# ADDENDUM-07 is --precision-share 0.25 --focus-area 0.8 0.65 0.5 0.35 0.2 0.0.
PRECISION_SHARES = [0.125, 0.25, 0.5, 0.75]
FOCUS_AREAS = [0.0, 0.2, 0.333, 0.5, 0.667, 0.8]
# Where an address is most likely to show: a mid precision share (room above and below) and the zones as found.
PROMISING = (0.375, 0.5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--precision-share", nargs="+", type=float, default=PRECISION_SHARES, help="share of the precision range spent, 0 ... 1")
    parser.add_argument("--focus-area", nargs="+", type=float, default=FOCUS_AREAS, help="0 the zones' centers only, 0.5 as found, 1 the whole map (no mask)")
    parser.add_argument("--limit", type=int, default=None, help="questions per topic, for a smoke check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-batch", type=int, default=POOLED_BATCH)
    parser.add_argument("--gradient-batch", type=int, default=GRADIENT_BATCH)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--gpu-share", type=float, default=default_share(), help="share of the GPU the run takes (foqlens/gpu_share.py)")
    parser.add_argument("--out", type=Path, default=Path("runs/E008-zones-fixed-budget"))
    return parser.parse_args(argv)


def cells_by_promise(shares: list[float], areas: list[float]) -> list[tuple[float, float]]:
    """Every (precision share, focus area) cell, nearest to PROMISING first (distance relative to each axis' center)."""
    p0, s0 = PROMISING
    cells = [(p, s) for p in shares for s in areas]
    return sorted(cells, key=lambda c: abs(c[0] - p0) / p0 + abs(c[1] - s0) / s0)


def zone_name(kind: str, source: str, focus_area: float, precision_share: float) -> str:
    return f"zone_{kind}_{source}_fa{focus_area:.2f}_ps{precision_share:.3f}"


def cell_policies(cell, coords, weights, topics, backbone_zones, seed) -> list:
    p, s = cell
    out = [legacy_zone_layout(zone_name("fixed", "backbone", s, p), FixedZones(backbone_zones), s, p, coords, weights, seed)]
    for src in MASK_SOURCES:
        kinds = {"own": OwnZones(topics[src]), "other": OtherZones(topics[src]), "random": RandomZones(topics[src], seed)}
        out += [legacy_zone_layout(zone_name(k, src, s, p), zs, s, p, coords, weights, seed) for k, zs in kinds.items()]
    return out


def comparisons(results: dict, domains: tuple[str, ...], cells, seed: int) -> dict:
    lp = {name: np.array([r[METRIC.primary] for r in rows]) for name, rows in results.items()}
    out = {}
    for pair in PAIR_NAMES:
        m = np.array([d in pair for d in domains])
        if not m.any():
            continue
        for src in MASK_SOURCES:
            for p, s in cells:
                own = lp[zone_name("own", src, s, p)][m]
                out[f"{'-'.join(pair)}|{src}|ps{p:.3f}|fa{s:.2f}"] = {
                    "own_minus_random": paired_bootstrap(own, lp[zone_name("random", src, s, p)][m], seed=seed),
                    "own_minus_other": paired_bootstrap(own, lp[zone_name("other", src, s, p)][m], seed=seed),
                    "own_minus_uniform": paired_bootstrap(own, lp[f"zone_uniform_ps{p:.3f}"][m], seed=seed),
                }
    return out


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    with layout_pool() as pool:  # started first: its imports run while the model loads
        return run(args, pool)


def run(args: argparse.Namespace, pool) -> Path:
    questions = [q for spec in SPECS.values() for q in read_questions(args.prompts_dir, spec, args.limit)]
    domains = tuple(q.domain for q in questions)
    bench = Bench.load(MODELS[args.model], gpu_share=args.gpu_share)

    with GpuMonitor() as mask_gpu:
        raw = bench.masks([q.prompt for q in questions], bench.sources(args.pooled_batch, args.gradient_batch))
    coords = coactivation_map(raw["gradient"])
    topics = {src: TopicZones(TopicMeans(m, domains), PAIRS, coords) for src, m in subtract_background(raw).items()}
    backbone_zones = find_zones(raw["gradient"].mean(axis=0), coords)
    weights = bg.block_weights(bench.ctl)
    metric = METRIC.for_tokenizer(bench.tokenizer)
    out_dir = args.out / args.model

    groups = list(batches(len(questions), args.eval_batch))

    def evaluate(policies: list, queued: list) -> dict:
        return evaluate_all(bench.model, bench.tokenizer, bench.ctl, questions, policies, metric, args.eval_batch,
                            throttle=bench.throttle, pool=pool, layouts=queued)

    references = [Uniform(Level.BF16, bench.ctl.n_blocks), Uniform(Level.D4, bench.ctl.n_blocks), Uniform(Level.D8, bench.ctl.n_blocks)]
    references += [legacy_zone_layout(f"zone_uniform_ps{p:.3f}", NoZones(coords.shape[1]), 1.0, p, coords, weights, args.seed)
                   for p in args.precision_share]
    zones_per_question = {src: float(np.mean([len(t.own(i).radii) for i in range(len(questions))]))
                          for src, t in topics.items()} | {"backbone": len(backbone_zones.radii)}
    with GpuMonitor() as eval_gpu:
        cells = cells_by_promise(args.precision_share, args.focus_area)
        cell_pols = [cell_policies(c, coords, weights, topics, backbone_zones, args.seed) for c in cells]
        # the worker queue always holds the next cell, so no cell starts by waiting for its layouts
        ref_queue = submit_layouts(pool, references, groups)
        queued = submit_layouts(pool, cell_pols[0], groups) if cells else []
        results = evaluate(references, ref_queue)
        done: list[tuple[float, float]] = []
        comps: dict = {}
        progress = Progress(len(cells), "cell")
        for k, cell in enumerate(cells):
            upcoming = submit_layouts(pool, cell_pols[k + 1], groups) if k + 1 < len(cells) else []
            results |= evaluate(cell_pols[k], queued)
            queued = upcoming
            done.append(cell)
            comps |= comparisons(results, domains, [cell], args.seed)  # only the new cell: the others are done
            summary = {
                "model": MODELS[args.model],
                "revision": fm.REVISIONS[MODELS[args.model]],
                "questions": {d: domains.count(d) for d in SPECS},
                "precision_share": args.precision_share,
                "focus_area": args.focus_area,
                "cells_done": [list(c) for c in done],
                "zones_per_question": zones_per_question,
                "gpu_share": args.gpu_share,
                "gpu": {"masks": mask_gpu.summary()},
                "comparisons": comps,
                "configs": summarize(results, questions),
            }
            write_json(out_dir / "summary.json", summary)
            print(progress.step(f"ps{cell[0]:.3f} fa{cell[1]:.2f}"), flush=True)
    summary["gpu"]["eval"] = eval_gpu.summary()
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "raw" / "per_question.json", {"questions": [q.__dict__ for q in questions], "results": results})
    np.save(out_dir / "raw" / "weight_map.npy", coords)
    for key, c in summary["comparisons"].items():
        line = "  ".join(f"{k} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" for k, v in c.items())
        print(f"{key:36} {line}")
    print(f"zones {summary['zones_per_question']}  gpu {summary['gpu']}\n-> {out_dir / 'summary.json'}")
    return out_dir / "summary.json"


if __name__ == "__main__":
    main()
