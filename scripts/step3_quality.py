"""Step 3: does precision allocated by the question's mask beat uniform and random allocation?

For every MMLU question (question + options):

1. Mask phase, all bf16: the mask of the question from each source - "pooled" (the naive score,
   mode B) and "gradient" (gradient x activation); each source's background (the mean mask over
   all questions of the run) is subtracted before ranking.
2. Evaluation phase, quality under every layout policy: uniform bf16 / int8 / nf4; and at every
   aperture a (share of weights read sharp): the question's own top blocks at bf16 and the rest
   at nf4 ("directed"), and random blocks within the same aperture ("random"). Aperture 1/3 has a
   mean of 8 bits, the same as uniform int8; apertures 0 and 1 are uniform nf4 and bf16.

Writes runs/step3/<model>/summary.json (GPU utilization and memory per phase) and raw results.

    uv run python scripts/step3_quality.py
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
from foqlens.precision import install
from foqlens.quality import compute_masks, evaluate_policy, summarize
from foqlens.quant import Level
from foqlens.scoring import BlockScorer, GradientScorer

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
DOMAINS = ["biology", "math", "chemistry", "physics", "heldout/history", "heldout/geography"]
# 0 and 1 are the uniform coarse and bf16 layouts. With ZERO a random 5% hole already breaks the
# model, so the grid of 0.1 gets a tail where the game is played.
APERTURES = [round(0.1 * i, 1) for i in range(1, 10)] + [0.95, 0.98, 0.99]
COARSE = {"nf4": Level.NF4, "zero": Level.ZERO}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--apertures", nargs="+", type=float, default=APERTURES, help="shares of weights read sharp")
    parser.add_argument("--coarse", choices=sorted(COARSE), default="nf4", help="level of the blocks outside the aperture")
    parser.add_argument("--limit", type=int, default=None, help="questions per domain, for a smoke check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-batch", type=int, default=16, help="questions per pooled-mask pass")
    parser.add_argument("--gradient-batch", type=int, default=4, help="questions per gradient pass (backward memory)")
    parser.add_argument("--eval-batch", type=int, default=32, help="questions per evaluation pass")
    parser.add_argument("--out", type=Path, default=Path("runs/step3"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    questions = [q for spec in args.domains for q in read_questions(args.prompts_dir, spec, args.limit)]
    prompts = [q.prompt for q in questions]

    # step 3 reads no attention weights, so fused sdpa attention: faster backward, less memory
    model, tokenizer = fm.load(MODELS[args.model], attn_implementation="sdpa")
    ctl = install(model)
    pooled, gradient = BlockScorer(ctl.modules), GradientScorer(model, ctl.modules)
    sources = {
        "pooled": (lambda t: [r["pooled"][0] for r in pooled.score_batch(model, tokenizer, t, modes=("pooled",))], args.pooled_batch),
        "gradient": (lambda t: [r["gradient"][0] for r in gradient.score_batch(model, tokenizer, t)], args.gradient_batch),
    }

    with GpuMonitor() as mask_gpu:
        ctl.set_all(Level.BF16)
        masks = {name: compute_masks(score, prompts, batch) for name, (score, batch) in sources.items()}

    weights = bg.block_weights(ctl)
    coarse = COARSE[args.coarse]
    policies = [Uniform(level, ctl.n_blocks) for level in Level]
    for a in args.apertures:
        policies.append(Random(a, weights, args.seed, coarse))
        policies += [Directed(name, a, m - m.mean(axis=0), weights, coarse) for name, m in masks.items()]
    ids = letter_ids(tokenizer)
    with GpuMonitor() as eval_gpu:
        results = {p.name: evaluate_policy(model, tokenizer, ctl, questions, p, ids, args.eval_batch) for p in policies}

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
    np.save(out_dir / "raw" / "masks.npy", np.stack([masks[s] for s in sources]))
    for name, c in summary["configs"].items():
        print(f"{name:26} acc {c['accuracy']:.3f}  logprob {c['logprob']:+.3f}  bits {c['mean_bits']:.2f}")
    print(f"gpu {summary['gpu']}\n-> {out_dir / 'summary.json'}")
    return out_dir / "summary.json"


if __name__ == "__main__":
    main()
