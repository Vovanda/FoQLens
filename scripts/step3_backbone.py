"""Backbone + topic (prereg/ADDENDUM-05.md): the address on top of the scale.

Generic block importance (the mean raw gradient x activation mask over all questions) is kept
sharp for a share of the aperture; the rest of the aperture is filled by the question's topic
mask ("own"), the paired topic's ("other") or random blocks. Outside the aperture: ZERO.
Paired bootstrap of the right-letter log-probability: own - other, own - random fill,
backbone alone - random blocks.

Writes runs/backbone/<model>/summary.json and raw per-question results.

    uv run python scripts/step3_backbone.py
    uv run python scripts/step3_backbone.py --limit 4 --apertures 0.95 --shares 0.8 --out /tmp/bb   # smoke check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from foqlens import budget as bg
from foqlens import model as fm
from foqlens.evaluate import letter_ids
from foqlens.gpu_monitor import GpuMonitor
from foqlens.io import read_questions, write_json
from foqlens.layouts import Backbone, BackboneFill, Random, TopicMeans, Uniform
from foqlens.pipeline import MASK_SOURCES, Bench, subtract_background
from foqlens.quality import evaluate_all, summarize
from foqlens.quant import Level
from foqlens.stats import paired_bootstrap
from foqlens.topics import PAIR_NAMES, PAIRS, SPECS

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
APERTURES = [0.99, 0.98, 0.97, 0.95, 0.9, 0.8, 0.5]
SHARES = [0.5, 0.8, 0.95]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--apertures", nargs="+", type=float, default=APERTURES)
    parser.add_argument("--shares", nargs="+", type=float, default=SHARES, help="backbone share of the aperture")
    parser.add_argument("--limit", type=int, default=None, help="questions per topic, for a smoke check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-batch", type=int, default=32)
    parser.add_argument("--gradient-batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("runs/backbone"))
    return parser.parse_args(argv)


def policies_for(args, backbone, means, weights, n_blocks) -> list:
    z = Level.ZERO
    out = [Uniform(Level.BF16, n_blocks), Uniform(z, n_blocks)]
    for a in args.apertures:
        out += [Random(a, weights, args.seed, z), Backbone(a, backbone, weights, z)]
        for s in args.shares:
            out.append(BackboneFill("random", "-", a, s, backbone, weights, seed=args.seed, coarse=z))
            for src in MASK_SOURCES:
                out += [BackboneFill(f, src, a, s, backbone, weights, means[src], PAIRS, args.seed, z) for f in ("own", "other")]
    return out


def comparisons(results: dict, domains: tuple[str, ...], args) -> dict:
    lp = {name: np.array([r["logprob"] for r in rows]) for name, rows in results.items()}
    out = {}
    for pair in PAIR_NAMES:
        m = np.array([d in pair for d in domains])
        if not m.any():
            continue
        tag = "-".join(pair)
        for a in args.apertures:
            out[f"{tag}|backbone|{a:.3f}"] = {"backbone_minus_random": paired_bootstrap(lp[f"backbone_{a:.3f}"][m], lp[f"random_{a:.3f}"][m], seed=args.seed)}
            for s in args.shares:
                rnd = lp[f"bb{s:.2f}_random_{a:.3f}"][m]
                for src in MASK_SOURCES:
                    own, other = lp[f"bb{s:.2f}_own_{src}_{a:.3f}"][m], lp[f"bb{s:.2f}_other_{src}_{a:.3f}"][m]
                    out[f"{tag}|{src}|{s:.2f}|{a:.3f}"] = {
                        "own_minus_other": paired_bootstrap(own, other, seed=args.seed),
                        "own_minus_random_fill": paired_bootstrap(own, rnd, seed=args.seed),
                    }
    return out


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    questions = [q for spec in SPECS.values() for q in read_questions(args.prompts_dir, spec, args.limit)]
    domains = tuple(q.domain for q in questions)
    bench = Bench.load(MODELS[args.model])

    with GpuMonitor() as mask_gpu:
        raw = bench.masks([q.prompt for q in questions], args.pooled_batch, args.gradient_batch)
    backbone = raw["gradient"].mean(axis=0)  # generic importance: the same for every question
    means = {src: TopicMeans(m, domains) for src, m in subtract_background(raw).items()}
    policies = policies_for(args, backbone, means, bg.block_weights(bench.ctl), bench.ctl.n_blocks)
    ids = letter_ids(bench.tokenizer)
    with GpuMonitor() as eval_gpu:
        results = evaluate_all(bench.model, bench.tokenizer, bench.ctl, questions, policies, ids, args.eval_batch)

    out_dir = args.out / args.model
    summary = {
        "model": MODELS[args.model],
        "revision": fm.REVISIONS[MODELS[args.model]],
        "questions": {d: domains.count(d) for d in SPECS},
        "apertures": args.apertures,
        "shares": args.shares,
        "coarse": "zero",
        "gpu": {"masks": mask_gpu.summary(), "eval": eval_gpu.summary()},
        "comparisons": comparisons(results, domains, args),
        "configs": summarize(results, questions),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "raw" / "per_question.json", {"questions": [q.__dict__ for q in questions], "results": results})
    for key, c in summary["comparisons"].items():
        first = next(iter(c.values()))
        print(f"{key:36} {next(iter(c)):22} {first['mean']:+.3f} [{first['lo']:+.3f}, {first['hi']:+.3f}]")
    print(f"gpu {summary['gpu']}\n-> {out_dir / 'summary.json'}")
    return out_dir / "summary.json"


if __name__ == "__main__":
    main()
