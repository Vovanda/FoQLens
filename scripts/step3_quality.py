"""Step 3: does precision allocated by the question's mask beat uniform and random allocation?

For every MMLU question (question + options):

1. Mask phase, all bf16: the mask of the question from each source - "pooled" (the naive score,
   mode B) and "gradient" (gradient x activation); each source's background (the mean mask over
   all questions of the run) is subtracted before ranking.
2. Evaluation phase, quality under every layout policy: every uniform level; and at every
   aperture a (share of weights read sharp): the question's own top blocks at bf16 and the rest
   at the coarse level ("directed"), and random blocks within the same aperture ("random").

Writes runs/step3/<model>/summary.json (GPU utilization and memory per phase) and raw results.

    uv run python scripts/step3_quality.py --coarse zero
    uv run python scripts/step3_quality.py --limit 3 --out /tmp/step3     # smoke check
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
from foqlens.layouts import Directed, Random, Uniform
from foqlens.pipeline import MASK_SOURCES, Bench, subtract_background
from foqlens.quality import evaluate_all, summarize
from foqlens.quant import Level

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
DOMAINS = ["biology", "math", "chemistry", "physics", "heldout/history", "heldout/geography"]
# 0 and 1 are the uniform coarse and bf16 layouts. With ZERO a random 5% hole already breaks the
# model, so the grid is logarithmic in the hole 1 - a: 1, 2, 3, 5, 10, 20% and one far probe at 50%,
# where random layouts are known dead and only a working mask could survive.
APERTURES = [0.99, 0.98, 0.97, 0.95, 0.9, 0.8, 0.5]
COARSE = {"nf4": Level.NF4, "zero": Level.ZERO}
UNIFORM_LEVELS = (Level.BF16, Level.INT8, Level.NF4, Level.ZERO)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--apertures", nargs="+", type=float, default=APERTURES, help="shares of weights read sharp")
    parser.add_argument("--coarse", choices=sorted(COARSE), default="zero", help="level of the blocks outside the aperture")
    parser.add_argument("--limit", type=int, default=None, help="questions per domain, for a smoke check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-batch", type=int, default=32, help="questions per pooled-mask pass")
    parser.add_argument("--gradient-batch", type=int, default=4, help="questions per gradient pass (backward memory)")
    parser.add_argument("--eval-batch", type=int, default=32, help="questions per evaluation pass")
    parser.add_argument("--out", type=Path, default=Path("runs/step3"))
    return parser.parse_args(argv)


def policies_for(apertures, scores, weights, n_blocks, seed, coarse) -> list:
    out = [Uniform(level, n_blocks) for level in UNIFORM_LEVELS]
    for a in apertures:
        out.append(Random(a, weights, seed, coarse))
        out += [Directed(name, a, scores[name], weights, coarse) for name in MASK_SOURCES]
    return out


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    questions = [q for spec in args.domains for q in read_questions(args.prompts_dir, spec, args.limit)]
    bench = Bench.load(MODELS[args.model])  # step 3 reads no attention weights: fused sdpa

    with GpuMonitor() as mask_gpu:
        raw_masks = bench.masks([q.prompt for q in questions], args.pooled_batch, args.gradient_batch)
    scores = subtract_background(raw_masks)
    policies = policies_for(args.apertures, scores, bg.block_weights(bench.ctl), bench.ctl.n_blocks, args.seed, COARSE[args.coarse])
    ids = letter_ids(bench.tokenizer)
    with GpuMonitor() as eval_gpu:
        results = evaluate_all(bench.model, bench.tokenizer, bench.ctl, questions, policies, ids, args.eval_batch)

    out_dir = args.out / args.model
    summary = {
        "model": MODELS[args.model],
        "revision": fm.REVISIONS[MODELS[args.model]],
        "questions": {d: sum(q.domain == d for q in questions) for d in dict.fromkeys(q.domain for q in questions)},
        "apertures": args.apertures,
        "coarse": args.coarse,
        "batch": {"pooled": args.pooled_batch, "gradient": args.gradient_batch, "eval": args.eval_batch},
        "gpu": {"masks": mask_gpu.summary(), "eval": eval_gpu.summary()},
        "configs": summarize(results, questions),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "raw" / "per_question.json", {"questions": [q.__dict__ for q in questions], "results": results})
    np.save(out_dir / "raw" / "masks.npy", np.stack([raw_masks[s] for s in MASK_SOURCES]))
    for name, c in summary["configs"].items():
        print(f"{name:26} acc {c['accuracy']:.3f}  logprob {c['logprob']:+.3f}  bits {c['mean_bits']:.2f}")
    print(f"gpu {summary['gpu']}\n-> {out_dir / 'summary.json'}")
    return out_dir / "summary.json"


if __name__ == "__main__":
    main()
