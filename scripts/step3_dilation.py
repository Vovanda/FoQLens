"""Dilation (prereg/ADDENDUM-06.md): the topic fill widened to the neighbours of its blocks.

Over the generic importance backbone (--share of the precision share) the rest of it is filled
by the own or the other topic's mask as in ADDENDUM-05 - as it is ("none"), or with every fill block
followed at once by its structural neighbours ("struct") or by as many index neighbours ("index").
Outside the precision share: ZERO. Paired bootstrap of the right-letter log-probability.

Writes runs/dilation/<model>/summary.json and raw per-question results.

    uv run python scripts/step3_dilation.py
    uv run python scripts/step3_dilation.py --limit 4 --precision-share 0.95 --out /tmp/dil   # smoke check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from foqlens import budget as bg
from foqlens import model as fm
from foqlens import neighbours as nb
from foqlens.evaluate import LetterChoice
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.io import read_questions, write_json
from foqlens.layouts import Backbone, BackboneFill, TopicMeans, Uniform
from foqlens.pipeline import MASK_SOURCES, Bench, subtract_background
from foqlens.quality import evaluate_all, summarize
from foqlens.quant import Level
from foqlens.stats import paired_bootstrap
from foqlens.topics import PAIR_NAMES, PAIRS, SPECS

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
METRIC = LetterChoice  # the quality metric; layouts are compared on METRIC.primary
# Where the model still answers: at 0.8 and below every policy of ADDENDUM-05 was at chance.
PRECISION_SHARES = [0.97, 0.95, 0.9]
SHARE = 0.8
DILATIONS = ("none", "struct", "index")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--precision-share", nargs="+", type=float, default=PRECISION_SHARES)
    parser.add_argument("--share", type=float, default=SHARE, help="backbone part of the precision share")
    parser.add_argument("--limit", type=int, default=None, help="questions per topic, for a smoke check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-batch", type=int, default=32)
    parser.add_argument("--gradient-batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--gpu-share", type=float, default=default_share(), help="share of the GPU the run takes (foqlens/gpu_share.py)")
    parser.add_argument("--out", type=Path, default=Path("runs/dilation"))
    return parser.parse_args(argv)


def neighbour_tables(ctl) -> dict[str, np.ndarray | None]:
    blocks = {name: m.n_blocks for name, m in ctl.modules.items()}
    table = nb.structural(blocks)
    return {"none": None, "struct": table, "index": nb.index_matched(blocks, table)}


def fill_name(share: float, fill: str, source: str, dilation: str, precision_share: float) -> str:
    wide = "" if dilation == "none" else f"_{dilation}"
    return f"bb{share:.2f}_{fill}_{source}{wide}_{precision_share:.3f}"


def policies_for(args, backbone, means, weights, n_blocks, tables) -> list:
    z = Level.ZERO
    out = [Uniform(Level.BF16, n_blocks), Uniform(z, n_blocks)]
    for a in args.precision_share:
        out += [Backbone(a, backbone, weights, z), BackboneFill("random", "-", a, args.share, backbone, weights, seed=args.seed, coarse=z)]
        for src in MASK_SOURCES:
            for fill in ("own", "other"):
                out += [
                    BackboneFill(fill, src, a, args.share, backbone, weights, means[src], PAIRS, args.seed, z, d, tables[d])
                    for d in DILATIONS
                ]
    return out


def comparisons(results: dict, domains: tuple[str, ...], args) -> dict:
    lp = {name: np.array([r[METRIC.primary] for r in rows]) for name, rows in results.items()}
    out = {}
    for pair in PAIR_NAMES:
        m = np.array([d in pair for d in domains])
        if not m.any():
            continue
        for src in MASK_SOURCES:
            for a in args.precision_share:
                own = {d: lp[fill_name(args.share, "own", src, d, a)][m] for d in DILATIONS}
                other = {d: lp[fill_name(args.share, "other", src, d, a)][m] for d in DILATIONS}
                cell = {f"own_minus_other_{d}": paired_bootstrap(own[d], other[d], seed=args.seed) for d in DILATIONS}
                cell["struct_minus_none"] = paired_bootstrap(own["struct"], own["none"], seed=args.seed)
                cell["struct_minus_index"] = paired_bootstrap(own["struct"], own["index"], seed=args.seed)
                out[f"{'-'.join(pair)}|{src}|{a:.3f}"] = cell
    return out


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    questions = [q for spec in SPECS.values() for q in read_questions(args.prompts_dir, spec, args.limit)]
    domains = tuple(q.domain for q in questions)
    bench = Bench.load(MODELS[args.model], gpu_share=args.gpu_share)

    with GpuMonitor() as mask_gpu:
        raw = bench.masks([q.prompt for q in questions], bench.sources(args.pooled_batch, args.gradient_batch))
    backbone = raw["gradient"].mean(axis=0)  # generic importance, as in ADDENDUM-05
    means = {src: TopicMeans(m, domains) for src, m in subtract_background(raw).items()}
    tables = neighbour_tables(bench.ctl)
    policies = policies_for(args, backbone, means, bg.block_weights(bench.ctl), bench.ctl.n_blocks, tables)
    metric = METRIC.for_tokenizer(bench.tokenizer)
    with GpuMonitor() as eval_gpu:
        results = evaluate_all(bench.model, bench.tokenizer, bench.ctl, questions, policies, metric, args.eval_batch, throttle=bench.throttle)

    out_dir = args.out / args.model
    summary = {
        "model": MODELS[args.model],
        "revision": fm.REVISIONS[MODELS[args.model]],
        "questions": {d: domains.count(d) for d in SPECS},
        "precision_share": args.precision_share,
        "share": args.share,
        "coarse": "zero",
        "neighbours_per_block": {d: float((t != nb.PAD).sum(axis=1).mean()) for d, t in tables.items() if t is not None},
        "gpu_share": args.gpu_share,
        "gpu": {"masks": mask_gpu.summary(), "eval": eval_gpu.summary()},
        "comparisons": comparisons(results, domains, args),
        "configs": summarize(results, questions),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "raw" / "per_question.json", {"questions": [q.__dict__ for q in questions], "results": results})
    for key, c in summary["comparisons"].items():
        line = "  ".join(f"{k} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" for k, v in c.items())
        print(f"{key:30} {line}")
    print(f"gpu {summary['gpu']}\n-> {out_dir / 'summary.json'}")
    return out_dir / "summary.json"


if __name__ == "__main__":
    main()
