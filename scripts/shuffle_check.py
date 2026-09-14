"""Does the result survive shuffling the answer options? A test of the metric, not of the model.

Every question is asked several times with its four options in a different order, the right answer
moving with them. The model's own bias toward a letter is then spread over all four, so a layout that
only exploits that bias loses its advantage, while a layout that reads the answer keeps it.

The masks are recomputed for every order, because the regulator sees the prompt that is actually
sent: taking the address off a differently ordered prompt would give the zones a head start no
deployment could give them.

Three configurations per order: bf16, uniform D6 - the reference E013 was read against at the same
memory - and the query's own zones at the best cell E013 found (floor D4, focus area 0.75, strength 1).
A mask read from a different text gives zones of a different size, so --focus-area is there to bring
two runs back to the same memory before they are compared.

Writes runs/reference/shuffle-answers/<model>/summary.json and the per-question results.

    uv run python scripts/shuffle_check.py
    uv run python scripts/shuffle_check.py --limit 4 --orders 2 --out /tmp/shuffle   # smoke check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from foqlens import model as fm
from foqlens.evaluate import LETTERS, LetterChoice, Question, mc_prompt
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.io import read_jsonl, write_json
from foqlens.layouts import OwnZones, TopicMeans, TopicZones, Uniform, graded_zone_layout
from foqlens.pipeline import GRADIENT_BATCH, POOLED_BATCH, Bench, subtract_background
from foqlens.progress import Progress
from foqlens.quality import evaluate_all, layout_pool, summarize
from foqlens.quant import Level
from foqlens.stats import paired_bootstrap
from foqlens.topics import PAIRS, PAIR_NAMES, SPECS
from foqlens.weight_map import coactivation_map

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
MASK_SOURCE = "gradient"
# The best cell of E013 (experiments/E013-regulator-map/results.md): what the shuffle has to survive.
BEST_CELL = {"floor": Level.D4, "focus_area": 0.75, "focus_strength": 1.0}
REFERENCES = {"uniform_bf16": Level.BF16, "uniform_d6": Level.D6}


def zone_layout_name(area: float) -> str:
    """The layout is named by the cell it is, so two runs at different sizes never share a key."""
    return f"zones_own_{BEST_CELL['floor'].name.lower()}_fa{area:.2f}_fs{BEST_CELL['focus_strength']:.2f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--orders", type=int, default=6, help="orders of the options per question, the first being the original")
    parser.add_argument("--masks-from-original", action="store_true",
                        help="take the address off the original order and reuse it for every shuffle - it separates "
                             "'the zones lose the address on a reordered prompt' from 'the advantage was in the letter'")
    parser.add_argument("--focus-area", type=float, default=BEST_CELL["focus_area"],
                        help="the size of the zones; the default is the best cell of E013. A different mask gives "
                             "zones of a different size, so matching the memory of another run means moving this")
    parser.add_argument("--address", choices=("prompt", "question", "canonical"), default="prompt",
                        help="what the address is read from. prompt: the prompt as sent, so the order of the options "
                             "moves it. question: the question alone, which is order-free but loses the options. "
                             "canonical: the whole prompt with its options sorted, order-free and still whole")
    parser.add_argument("--answer-in-original-order", action="store_true",
                        help="score every order's address on the original order of the options, so the only thing "
                             "that changes between the orders is the text the address was read from. It measures "
                             "how stable the address itself is, with the questions held fixed")
    parser.add_argument("--limit", type=int, default=None, help="questions per topic, for a smoke check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-batch", type=int, default=POOLED_BATCH)
    parser.add_argument("--gradient-batch", type=int, default=GRADIENT_BATCH)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--out", type=Path, default=Path("runs/reference/shuffle-answers"))
    return parser.parse_args(argv)


def read_rows(prompts_dir: Path, limit: int | None) -> tuple[list[dict], list[str]]:
    """The raw question rows of every topic, with the topic of each - the options are reordered later."""
    rows, domains = [], []
    for topic, spec in SPECS.items():
        for row in read_jsonl(prompts_dir / f"{spec}.jsonl", limit):
            rows.append(row)
            domains.append(topic)
    return rows, domains


def in_order(rows: list[dict], domains: list[str], order: np.ndarray) -> list[Question]:
    """The same questions with their options in this order; the right answer moves with its option."""
    return [
        Question(domain, mc_prompt(row["text"], [row["choices"][i] for i in order]),
                 int(np.flatnonzero(order == row["answer"])[0]))
        for row, domain in zip(rows, domains, strict=True)
    ]


def addressed_text(rows: list[dict], questions: list[Question], address: str) -> list[str]:
    """The text the address is read from - see --address.

    `canonical` sorts the options before building the prompt, so every ordering of the same question
    gives the same text and therefore the same zones, while the mask still sees the whole question.
    """
    if address == "prompt":
        return [q.prompt for q in questions]
    if address == "question":
        return [r["text"] for r in rows]
    return [mc_prompt(r["text"], sorted(r["choices"])) for r in rows]


def orders_of(count: int, seed: int) -> list[np.ndarray]:
    """The original order first, then shuffles - so the run also reproduces the number every other run reports."""
    rng = np.random.default_rng(seed)
    return [np.arange(len(LETTERS))] + [rng.permutation(len(LETTERS)) for _ in range(count - 1)]


def picked_letters(rows: list[dict]) -> list[float]:
    """How often each letter came out on top - the lean the shuffle is there to spread."""
    top = np.array([int(r["picked"]) for r in rows])
    return (np.bincount(top, minlength=len(LETTERS)) / len(top)).tolist()


def compare(values: dict[str, np.ndarray], keep: np.ndarray, seed: int, own: str) -> dict:
    """The zones against each reference on the questions kept, paired over the questions."""
    return {f"zones_minus_{name}": paired_bootstrap(values[own][keep], values[name][keep], seed=seed)
            for name in REFERENCES}


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    rows, domains = read_rows(args.prompts_dir, args.limit)
    domains = np.array(domains)
    orders = orders_of(args.orders, args.seed)
    bench = Bench.load(MODELS[args.model], gpu_share=args.gpu_share)
    metric = LetterChoice.for_tokenizer(bench.tokenizer)
    own = zone_layout_name(args.focus_area)
    out_dir = args.out / args.model

    per_order: dict[str, dict] = {}
    progress = Progress(len(orders), "order")
    own_policy = None
    with layout_pool() as pool, GpuMonitor() as gpu:
        for k, order in enumerate(orders):
            questions = in_order(rows, domains.tolist(), order)
            if own_policy is None or not args.masks_from_original:
                addressed = addressed_text(rows, questions, args.address)
                raw = bench.masks(addressed, bench.sources(args.pooled_batch, args.gradient_batch))
                coords = coactivation_map(raw[MASK_SOURCE])
                topics = TopicZones(TopicMeans(subtract_background(raw)[MASK_SOURCE], tuple(domains)), PAIRS, coords)
                own_policy = graded_zone_layout(own, OwnZones(topics), args.focus_area,
                                                 BEST_CELL["focus_strength"], coords, floor=BEST_CELL["floor"])
            # the address came from this order; the answering can stay on the original one, and then the
            # only thing that differs between the orders is the text the address was read from
            if args.answer_in_original_order:
                questions = in_order(rows, domains.tolist(), orders[0])
            policies = [Uniform(level, bench.ctl.n_blocks) for level in REFERENCES.values()]
            policies.append(own_policy)
            results = evaluate_all(bench.model, bench.tokenizer, bench.ctl, questions, policies, metric,
                                   args.eval_batch, throttle=bench.throttle, pool=pool)
            per_order[f"order_{k}"] = {
                "order": order.tolist(),
                "summary": summarize(results, questions),
                "picked": {name: picked_letters(rows_) for name, rows_ in results.items()},
                "accuracy": {name: [float(r["accuracy"]) for r in rows_] for name, rows_ in results.items()},
                "logprob": {name: [float(r["logprob"]) for r in rows_] for name, rows_ in results.items()},
            }
            write_json(out_dir / "raw" / "per_order.json", per_order)
            print(progress.step(f"order {order.tolist()}"), flush=True)

    # pooled over the orders: a question's value is its mean over the orders, so the letter bias averages out
    def pooled(field: str) -> dict[str, np.ndarray]:
        return {name: np.mean([per_order[o][field][name] for o in per_order], axis=0) for name in
                (own, *REFERENCES)}

    everything = np.ones(len(rows), bool)
    comparisons = {
        "original_order": {field: compare({n: np.array(per_order["order_0"][field][n]) for n in (own, *REFERENCES)},
                                          everything, args.seed, own) for field in ("accuracy", "logprob")},
        "pooled_over_orders": {field: compare(pooled(field), everything, args.seed, own) for field in ("accuracy", "logprob")},
    }
    for pair in PAIR_NAMES:
        keep = np.isin(domains, pair)
        comparisons[f"pooled|{'-'.join(pair)}"] = {field: compare(pooled(field), keep, args.seed, own)
                                                   for field in ("accuracy", "logprob")}

    summary = {
        "model": MODELS[args.model],
        "revision": fm.REVISIONS[MODELS[args.model]],
        "questions": {d: int((domains == d).sum()) for d in SPECS},
        "orders": [o.tolist() for o in orders],
        "cell": {"floor": BEST_CELL["floor"].name, "focus_area": args.focus_area,
                 "focus_strength": BEST_CELL["focus_strength"]},
        "seed": args.seed,
        "masks_from_original": args.masks_from_original,
        "address": args.address,
        "answer_in_original_order": args.answer_in_original_order,
        "gpu_share": args.gpu_share,
        "gpu": gpu.summary(),
        "accuracy_per_order": {name: [per_order[o]["summary"][name]["accuracy"] for o in per_order]
                               for name in (own, *REFERENCES)},
        "letters_picked_per_order": {name: [per_order[o]["picked"][name] for o in per_order]
                                     for name in (own, *REFERENCES)},
        "mean_bits": {name: per_order["order_0"]["summary"][name]["mean_bits"] for name in (own, *REFERENCES)},
        "comparisons": comparisons,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"written {out_dir / 'summary.json'}", flush=True)
    return out_dir / "summary.json"


if __name__ == "__main__":
    main()
