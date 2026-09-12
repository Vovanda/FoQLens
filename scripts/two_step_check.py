"""The two-step metric: the model answers the bare question, then the options are matched to it.

The one-step metric - log-probabilities of " A" .. " D" after "Answer:" - turned out to measure
partly a lean toward a letter, and the address read off a prompt that carries the options moves by
6.3 points of accuracy when those options are merely reordered (runs/reference/address-stability).

This run removes both. The model never sees the options while it answers, so the address is read from
a text in which there is nothing to reorder; and what it writes is then matched to the options by the
model at bf16, identically for every layout, so the matcher's own lean is a constant rather than
something a layout can exploit.

The generation does not depend on the order of the options at all, so it runs once per layout; only
the matching is repeated over the orders, which is exactly the part that can still see them.

First it checks the matcher: the two-step accuracy of bf16 against its one-step accuracy on the same
questions. A matcher that loses the model's right answers cannot measure anything.

Writes runs/reference/two-step/<model>/summary.json and the per-question results.

    uv run python scripts/two_step_check.py
    uv run python scripts/two_step_check.py --limit 4 --orders 2 --out /tmp/two-step   # smoke check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from foqlens import model as fm
from foqlens.evaluate import LETTERS, letter_ids
from foqlens.generation import question_prompt
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.io import read_jsonl, write_json
from foqlens.layouts import OwnZones, TopicMeans, TopicZones, Uniform, graded_zone_layout
from foqlens.matching import LetterMatcher, free_answers
from foqlens.pipeline import GRADIENT_BATCH, POOLED_BATCH, Bench, subtract_background
from foqlens.progress import Progress
from foqlens.quant import Level
from foqlens.stats import paired_bootstrap
from foqlens.topics import PAIRS, PAIR_NAMES, SPECS
from foqlens.weight_map import coactivation_map

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
MASK_SOURCE = "gradient"
BEST_CELL = {"floor": Level.D4, "focus_area": 0.75, "focus_strength": 1.0}
REFERENCES = {"uniform_bf16": Level.BF16, "uniform_d6": Level.D6}
ANSWER_TOKENS = 32  # MMLU answers are short phrases; beyond this the model starts explaining itself


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--orders", type=int, default=6, help="orders the options are matched in; the first is the original")
    parser.add_argument("--focus-area", type=float, default=BEST_CELL["focus_area"])
    parser.add_argument("--answer-tokens", type=int, default=ANSWER_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="questions per topic, for a smoke check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-batch", type=int, default=POOLED_BATCH)
    parser.add_argument("--gradient-batch", type=int, default=GRADIENT_BATCH)
    parser.add_argument("--match-batch", type=int, default=16)
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--out", type=Path, default=Path("runs/reference/two-step"))
    return parser.parse_args(argv)


def lens_name(area: float) -> str:
    return f"lens_own_{BEST_CELL['floor'].name.lower()}_fa{area:.2f}_fs{BEST_CELL['focus_strength']:.2f}"


def read_rows(prompts_dir: Path, limit: int | None) -> tuple[list[dict], list[str]]:
    rows, domains = [], []
    for topic, spec in SPECS.items():
        for row in read_jsonl(prompts_dir / f"{spec}.jsonl", limit):
            rows.append(row)
            domains.append(topic)
    return rows, domains


def orders_of(count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.arange(len(LETTERS))] + [rng.permutation(len(LETTERS)) for _ in range(count - 1)]


def generate_under(bench, prompts: list[str], levels: np.ndarray | None, tokens: int, log) -> list[str]:
    """The model's own answers, with the layout of each question set before that question is asked."""
    if levels is None:
        bench.ctl.set_all(Level.BF16)
        return free_answers(bench.model, bench.tokenizer, prompts, tokens, log)
    out = []
    for i, prompt in enumerate(prompts):
        bench.ctl.set_layout(levels[i])
        out += free_answers(bench.model, bench.tokenizer, [prompt], tokens)
        if (i + 1) % 50 == 0:
            log(f"  generated {i + 1}/{len(prompts)}")
    bench.ctl.set_all(Level.BF16)
    return out


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    rows, domains = read_rows(args.prompts_dir, args.limit)
    domains = np.array(domains)
    questions = [r["text"] for r in rows]
    prompts = [question_prompt(q) for q in questions]
    orders = orders_of(args.orders, args.seed)
    bench = Bench.load(MODELS[args.model], gpu_share=args.gpu_share)
    lens = lens_name(args.focus_area)
    out_dir = args.out / args.model
    log = lambda s: print(s, flush=True)

    # The address is read from the bare questions - the text the model actually answers, and the one
    # in which no option can be reordered.
    with GpuMonitor() as gpu:
        raw = bench.masks(prompts, bench.sources(args.pooled_batch, args.gradient_batch))
        coords = coactivation_map(raw[MASK_SOURCE])
        topics = TopicZones(TopicMeans(subtract_background(raw)[MASK_SOURCE], tuple(domains)), PAIRS, coords)
        policies = {name: Uniform(level, bench.ctl.n_blocks) for name, level in REFERENCES.items()}
        policies[lens] = graded_zone_layout(lens, OwnZones(topics), args.focus_area,
                                            BEST_CELL["focus_strength"], coords, floor=BEST_CELL["floor"])

        answers, bits = {}, {}
        progress = Progress(len(policies), "layout")
        for name, policy in policies.items():
            levels = None if name == "uniform_bf16" else policy.levels(np.arange(len(rows)))
            answers[name] = generate_under(bench, prompts, levels, args.answer_tokens, log)
            bits[name] = float(np.mean([Level(int(c)).bits for c in levels.ravel()])) if levels is not None else 16.0
            log(progress.step(name))

        matcher = LetterMatcher(tuple(letter_ids(bench.tokenizer)), bench.model, bench.tokenizer,
                               bench.ctl, args.match_batch)
        per_order: dict[str, dict] = {}
        progress = Progress(len(orders), "order")
        for k, order in enumerate(orders):
            options = [[r["choices"][i] for i in order] for r in rows]
            right = np.array([int(np.flatnonzero(order == r["answer"])[0]) for r in rows])
            picked = {name: matcher.match(answers[name], questions, options) for name in policies}
            per_order[f"order_{k}"] = {
                "order": order.tolist(),
                "accuracy": {name: (p == right).astype(float).tolist() for name, p in picked.items()},
                "picked": {name: (np.bincount(p, minlength=len(LETTERS)) / len(p)).tolist()
                           for name, p in picked.items()},
            }
            write_json(out_dir / "raw" / "per_order.json", per_order)
            log(progress.step(f"order {order.tolist()}"))

    names = list(policies)
    pooled = {name: np.mean([per_order[o]["accuracy"][name] for o in per_order], axis=0) for name in names}
    compare = lambda keep: {f"lens_minus_{n}": paired_bootstrap(pooled[lens][keep], pooled[n][keep], seed=args.seed)
                            for n in REFERENCES}
    everything = np.ones(len(rows), bool)
    summary = {
        "model": MODELS[args.model],
        "revision": fm.REVISIONS[MODELS[args.model]],
        "questions": {d: int((domains == d).sum()) for d in SPECS},
        "orders": [o.tolist() for o in orders],
        "cell": {"floor": BEST_CELL["floor"].name, "focus_area": args.focus_area,
                 "focus_strength": BEST_CELL["focus_strength"]},
        "answer_tokens": args.answer_tokens,
        "seed": args.seed,
        "gpu_share": args.gpu_share,
        "gpu": gpu.summary(),
        "mean_bits": bits,
        "accuracy_per_order": {name: [float(np.mean(per_order[o]["accuracy"][name])) for o in per_order]
                               for name in names},
        "letters_picked_per_order": {name: [per_order[o]["picked"][name] for o in per_order] for name in names},
        "comparisons": {"pooled_over_orders": compare(everything)}
                       | {f"pooled|{'-'.join(p)}": compare(np.isin(domains, p)) for p in PAIR_NAMES},
        "answers_sample": {name: answers[name][:5] for name in names},
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "raw" / "answers.json", answers)
    log(f"written {out_dir / 'summary.json'}")
    return out_dir / "summary.json"


if __name__ == "__main__":
    main()
