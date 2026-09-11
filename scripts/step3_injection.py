"""Mask injection (experiments/E004-injection/ADDENDUM-04.md): the mask of topic A on the questions of topic B.

For every question of the paired topics, at every precision share (outside it: ZERO), the question is
answered under its own mask ("self"), its topic's mask ("own", leave-one-out), the paired topic's
mask ("other") and random blocks ("random"); uniform bf16 and ZERO are the ends. Paired
bootstrap of the right-letter log-probability: own - other, own - random, self - own.

Writes runs/E004-injection/<model>/summary.json and raw per-question results.

    uv run python scripts/step3_injection.py
    uv run python scripts/step3_injection.py --limit 4 --precision-share 0.95 --out /tmp/inj   # smoke check
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
from foqlens.layouts import Directed, OtherTopic, OwnTopic, Random, TopicMask, TopicMeans, Uniform
from foqlens.pipeline import GRADIENT_BATCH, MASK_SOURCES, POOLED_BATCH, Bench, subtract_background
from foqlens.quality import evaluate_all, summarize
from foqlens.quant import Level
from foqlens.stats import paired_bootstrap
from foqlens.topics import PAIR_NAMES, PAIRS, SPECS

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
METRIC = LetterChoice  # the quality metric; layouts are compared on METRIC.primary
PRECISION_SHARES = [0.99, 0.98, 0.97, 0.95, 0.9, 0.8, 0.5]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--precision-share", nargs="+", type=float, default=PRECISION_SHARES)
    parser.add_argument("--limit", type=int, default=None, help="questions per topic, for a smoke check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-batch", type=int, default=POOLED_BATCH)
    parser.add_argument("--gradient-batch", type=int, default=GRADIENT_BATCH)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--gpu-share", type=float, default=default_share(), help="share of the GPU the run takes (foqlens/gpu_share.py)")
    parser.add_argument("--out", type=Path, default=Path("runs/E004-injection"))
    return parser.parse_args(argv)


def policies_for(precision_shares, scores, domains, weights, n_blocks, seed) -> list:
    out = [Uniform(Level.BF16, n_blocks), Uniform(Level.ZERO, n_blocks)]
    means = {s: TopicMeans(scores[s], domains) for s in MASK_SOURCES}
    for a in precision_shares:
        out.append(Random(a, weights, seed, Level.ZERO))
        for s in MASK_SOURCES:
            out.append(Directed(f"self_{s}", a, scores[s], weights, Level.ZERO))
            out += [TopicMask(t, s, a, weights, Level.ZERO) for t in (OwnTopic(means[s]), OtherTopic(means[s], PAIRS))]
    return out


def comparisons(results: dict, domains: tuple[str, ...], precision_shares, seed: int) -> dict:
    """Paired bootstraps per pair of topics, source and precision share."""
    lp = {name: np.array([r[METRIC.primary] for r in rows]) for name, rows in results.items()}
    out = {}
    for pair in PAIR_NAMES:
        mask = np.array([d in pair for d in domains])
        if not mask.any():
            continue
        for s in MASK_SOURCES:
            for a in precision_shares:
                own, other = lp[f"own_topic_{s}_{a:.3f}"][mask], lp[f"other_topic_{s}_{a:.3f}"][mask]
                self_, rnd = lp[f"directed_self_{s}_{a:.3f}"][mask], lp[f"random_{a:.3f}"][mask]
                out[f"{'-'.join(pair)}|{s}|{a:.3f}"] = {
                    "own_minus_other": paired_bootstrap(own, other, seed=seed),
                    "own_minus_random": paired_bootstrap(own, rnd, seed=seed),
                    "self_minus_own": paired_bootstrap(self_, own, seed=seed),
                }
    return out


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    questions = [q for spec in SPECS.values() for q in read_questions(args.prompts_dir, spec, args.limit)]
    domains = tuple(q.domain for q in questions)
    bench = Bench.load(MODELS[args.model], gpu_share=args.gpu_share)

    with GpuMonitor() as mask_gpu:
        scores = subtract_background(bench.masks([q.prompt for q in questions], bench.sources(args.pooled_batch, args.gradient_batch)))
    policies = policies_for(args.precision_share, scores, domains, bg.block_weights(bench.ctl), bench.ctl.n_blocks, args.seed)
    metric = METRIC.for_tokenizer(bench.tokenizer)
    with GpuMonitor() as eval_gpu:
        results = evaluate_all(bench.model, bench.tokenizer, bench.ctl, questions, policies, metric, args.eval_batch, throttle=bench.throttle)

    out_dir = args.out / args.model
    summary = {
        "model": MODELS[args.model],
        "revision": fm.REVISIONS[MODELS[args.model]],
        "questions": {d: domains.count(d) for d in SPECS},
        "precision_share": args.precision_share,
        "coarse": "zero",
        "gpu_share": args.gpu_share,
        "gpu": {"masks": mask_gpu.summary(), "eval": eval_gpu.summary()},
        "comparisons": comparisons(results, domains, args.precision_share, args.seed),
        "configs": summarize(results, questions),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "raw" / "per_question.json", {"questions": [q.__dict__ for q in questions], "results": results})
    for key, c in summary["comparisons"].items():
        oo = c["own_minus_other"]
        print(f"{key:32} own-other {oo['mean']:+.3f} [{oo['lo']:+.3f}, {oo['hi']:+.3f}]  own-random {c['own_minus_random']['mean']:+.3f}")
    print(f"gpu {summary['gpu']}\n-> {out_dir / 'summary.json'}")
    return out_dir / "summary.json"


if __name__ == "__main__":
    main()
