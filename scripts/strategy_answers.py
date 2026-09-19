"""A mechanism of the filter on a small part of the frozen corpus: every question read at its own layout, judged at bf16.

Per corpus, `--share` of the kept questions are laid out and `--calibration-share` other kept questions give the
background and the co-activation graph; both are drawn by the seed, the two sets never meet. `--unknown-share` of the
questions the model does not know are laid out too and listed apart in the summary: they never enter the headline. The masks of all of them come from one
mask source at bf16 (pipeline.Bench.source); the mechanism (strategies.MECHANISMS) turns the laid-out questions'
masks into levels with the knobs of docs/quantization-filter.md; the regulator checks every layout against the
kernel's ladder, applies rule 6 and sets it per question of a batch. The judge reads at bf16 as always.

The summary holds the bytes the kernel reads - per question, and per decoding step of a batch (the union of its
questions' zones) - beside what the uniform ladder reads, so that a mechanism is compared at the same memory; and the
spread of the layouts over layers and module kinds.

    uv run python scripts/strategy_answers.py --mechanism per-block --floor d2 --focus-area 0.2 --focus-strength 1
    uv run python scripts/strategy_answers.py --mechanism static --base bartowski-Q2_K
    uv run python scripts/strategy_answers.py --mechanism signal-path --corpora triviaqa --share 0.002 --min-questions 8 \
        --calibration-share 0.004 --unknown-share 0.01  # smoke
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from foqlens import corpora, refocustensors
from foqlens import model as fm
from foqlens.answering import Asking
from foqlens.attention import PLANS, SPLIT
from foqlens.gguf_weights import PUBLISHED
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.graph_decode import PREFILL_TOKENS, StaticDecoder
from foqlens.io import answers_path, append_answers, read_frozen, write_json
from foqlens.judging import ModelJudge
from foqlens.pipeline import Bench, GRADIENT_BATCH, POOLED_BATCH
from foqlens.prompt_variants import SETUPS, TRAIN_POOL, examples_for, needs_train, setup_named
from foqlens.quant import Level
from foqlens.regulator import Regulator
from foqlens.selection import split_shares
from foqlens.strategies import GRAPHS, MECHANISMS, NEIGHBOURS, REACHES, Inputs, Knobs, mechanism_layout

MODELS = {"e2b-it": fm.E2B_IT, "e4b-it": fm.E4B_IT}
FLOORS = {lv.name.lower(): lv for lv in (Level.ZERO, Level.D2, Level.D4, Level.D6)}
SOURCES = ("pooled", "neuron_activity", "head_energy", "gradient", "gradient_magnitude")
BATCHES = {"gradient": GRADIENT_BATCH, "gradient_magnitude": GRADIENT_BATCH}  # the forward sources take POOLED_BATCH
# The small corpus (Volodya 19.09): 5% of the kept questions, one seeded draw for every configuration; 20% more
# give the background and the co-activation graph (60 a corpus was too few for a graph over thousands of blocks);
# 50 a corpus at least, as ARC-Challenge's 5% is 26; 5% of the unknown share on a line of its own.
LAID_SHARE = 0.05
CALIBRATION_SHARE = 0.2
MIN_QUESTIONS = 50
UNKNOWN_SHARE = 0.05


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--base", choices=sorted(PUBLISHED), default=None, help="the model cut over a published base")
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--share", type=float, default=LAID_SHARE, help="share of a corpus's kept questions laid out")
    parser.add_argument("--calibration-share", type=float, default=CALIBRATION_SHARE,
                        help="share of its other kept questions giving the background and the graph")
    parser.add_argument("--min-questions", type=int, default=MIN_QUESTIONS, help="the fewest of either, per corpus")
    parser.add_argument("--unknown-share", type=float, default=UNKNOWN_SHARE,
                        help="share of the questions the model does not know (unknown_share), laid out on their own line")
    parser.add_argument("--mechanism", choices=list(MECHANISMS), required=True)
    parser.add_argument("--source", choices=SOURCES, default="pooled", help="the mask source of the address (#18)")
    parser.add_argument("--graph", choices=sorted(GRAPHS), default="mutual-nicdm")
    parser.add_argument("--k", type=int, default=NEIGHBOURS, help="neighbours (or strongest edges) of every block")
    parser.add_argument("--floor", choices=list(FLOORS), default="d2", help="the base precision")
    parser.add_argument("--focus-area", type=float, required=True, help="f: the share of the network a zone reaches")
    parser.add_argument("--focus-strength", type=float, default=1.0, help="g: how far a zone rises of the way to D8")
    parser.add_argument("--combine", choices=("sum", "max"), default="sum")
    parser.add_argument("--reach", choices=sorted(REACHES), default="equal",
                        help="f D for every zone (rule 1), or f D split by the zones' own widths")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/strategies"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--attention", choices=list(PLANS), default=SPLIT.name)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    model_id = MODELS[args.model]
    directory = refocustensors.model_directory(model_id, args.base) if args.base else None
    bench = Bench.load(model_id, gpu_share=args.gpu_share, directory=directory)
    tokenizer, ctl = bench.tokenizer, bench.ctl
    fmt = fm.prompt_format(model_id, tokenizer)
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    laid, calibration, unknown, askings = [], [], set(), {}
    for corpus in args.corpora:
        corpus_rows, source = corpora.read(corpus)
        frozen = read_frozen(args.frozen / f"{corpus}.json")
        frozen.check(name, source.revision, frozen.prompt)  # the frozen file names the setup it was frozen in
        known, background = split_shares(list(frozen.kept), args.seed, (args.share, args.calibration_share),
                                         args.min_questions)
        [strangers] = split_shares(list(frozen.unknown_share), args.seed, (args.unknown_share,))
        by_id = {r.id: r for r in corpus_rows}
        laid += [(corpus, by_id[i]) for i in known + strangers]
        unknown |= {(corpus, i) for i in strangers}
        calibration += [(corpus, by_id[i]) for i in background]
        setup = setup_named(corpus, frozen.prompt)
        train = corpora.read_train(corpus, TRAIN_POOL) if needs_train(setup) else []
        askings[corpus] = Asking(corpus, source.revision, name, Level.BF16, setup,
                                 examples_for(corpus, setup, train, args.seed))

    def prompts(pairs: list) -> list[str]:  # the prompt each question is answered with is the one its mask reads
        return [askings[c].prompts(fmt, [r])[0] for c, r in pairs]

    mask_source = bench.source(args.source, BATCHES.get(args.source, POOLED_BATCH))
    masks = bench.masks(prompts(laid) + prompts(calibration), [mask_source])[mask_source.name]
    modules = ctl.modules.values()
    inputs = Inputs(scores=masks[: len(laid)], calibration=masks[len(laid):],
                    block_weights=np.concatenate([m.block_sizes() * m.in_features for m in modules]),
                    domains=tuple(c for c, _ in laid),
                    model_weights={n: m.weight for n, m in ctl.modules.items()},
                    n_heads=bench.model.config.get_text_config(decoder=True).num_attention_heads)
    knobs = Knobs(FLOORS[args.floor], args.focus_area, args.focus_strength, args.combine)
    label = f"{args.mechanism}-{args.source}-{args.reach}-{args.floor}-f{args.focus_area:g}-g{args.focus_strength:g}"
    policy = mechanism_layout(args.mechanism, inputs, knobs, graph=args.graph, k=args.k, reach=args.reach)
    regulator = Regulator(policy, ctl)
    reading = regulator.reading({(c, r.id): i for i, (c, r) in enumerate(laid)}, label)

    decoder = StaticDecoder(attention=PLANS[args.attention], prefill_tokens=PREFILL_TOKENS)
    judge = ModelJudge(bench.model, tokenizer, ctl, fmt, decoder)
    out = args.out / args.model
    with GpuMonitor() as gpu:
        for corpus, asking in askings.items():
            asking = replace(asking, reading=reading)
            rows = [r for c, r in laid if c == corpus]
            for chunk in asking.batches(fmt, tokenizer, rows):
                append_answers(answers_path(out / "answers", label, corpus),
                               asking.answer(bench.model, tokenizer, ctl, fmt, judge, chunk, bench.throttle, decoder))

    codes = np.concatenate(reading.laid)
    uniform = {lv.name.lower(): int(regulator.cost.read_bytes(np.full(ctl.n_blocks, int(lv), dtype=np.uint8))[0])
               for lv in regulator.ladder}
    target = out / f"summary-{label}.json"
    write_json(target, {
        "model": name, "base": args.base, "mechanism": args.mechanism, "source": args.source, "graph": args.graph,
        "k": args.k, "floor": args.floor, "focus_area": args.focus_area, "focus_strength": args.focus_strength,
        "combine": args.combine, "reach": args.reach, "seed": args.seed,
        "questions": {c: [r.id for cc, r in laid if cc == c and (cc, r.id) not in unknown] for c in askings},
        "unknown": {c: [i for cc, i in sorted(unknown) if cc == c] for c in askings},
        "calibration": len(calibration),
        "bytes": {"per_question_mean": float(regulator.cost.read_bytes(codes).mean()),
                  "per_step_mean": float(np.mean([regulator.cost.step_bytes(c) for c in reading.laid])),
                  "uniform": uniform},
        "by_layer": regulator.by_layer(codes), "gpu": gpu.summary(), "pacer": bench.throttle.stats(),
    })
    print(f"written {target}", flush=True)
    return target


if __name__ == "__main__":
    main()
