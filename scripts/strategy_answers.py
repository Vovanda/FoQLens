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
    uv run python scripts/strategy_answers.py --mechanism signal-path --corpora triviaqa \
        --corpus-config configs/smoke-corpus.toml  # smoke
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from foqlens import config, corpora, refocustensors
from foqlens import model as fm
from foqlens.answering import Asking
from foqlens.coverage import question_coverage, run_coverage
from foqlens.attention import PLANS, SPLIT
from foqlens.gguf_weights import PUBLISHED
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.graph_decode import PREFILL_TOKENS, StaticDecoder
from foqlens.io import answers_path, append_answers, read_frozen, write_json
from foqlens.judging import ModelJudge
from foqlens.layouts import WorkingLayers
from foqlens.pipeline import ADDRESS_SOURCES, Bench
from foqlens.prompt_variants import SETUPS, TRAIN_POOL, examples_for, needs_train, setup_named
from foqlens.quant import Level
from foqlens.regulator import Regulator, block_layers
from foqlens.selection import split_shares
from foqlens.strategies import GRAPHS, MECHANISMS, NEIGHBOURS, REACHES, Inputs, Knobs, mechanism_layout

MODELS = {"e2b-it": fm.E2B_IT, "e4b-it": fm.E4B_IT}
FLOORS = {lv.name.lower(): lv for lv in (Level.ZERO, Level.D2, Level.D4, Level.D6)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--base", choices=sorted(PUBLISHED), default=None, help="the model cut over a published base")
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--corpus-config", type=Path, default=Path("configs/small-corpus.toml"),
                        help="the small corpus (config.SmallCorpus): the shares laid out, for calibration, unknown")
    parser.add_argument("--mechanism", choices=list(MECHANISMS), required=True)
    parser.add_argument("--source", choices=sorted(ADDRESS_SOURCES), default="pooled",
                        help="the mask source of the address (#18)")
    parser.add_argument("--graph", choices=sorted(GRAPHS), default="mutual-nicdm")
    parser.add_argument("--k", type=int, default=NEIGHBOURS, help="neighbours (or strongest edges) of every block")
    parser.add_argument("--floor", choices=list(FLOORS), default="d2", help="the base precision")
    parser.add_argument("--focus-area", type=float, required=True, help="f: the share of the network a zone reaches")
    parser.add_argument("--focus-strength", type=float, default=1.0, help="g: how far a zone rises of the way to D8")
    parser.add_argument("--combine", choices=("sum", "max"), default="sum")
    parser.add_argument("--working-layers", type=int, default=0,
                        help="the first layers the address is read from: they read --working-level, the filter acts after")
    parser.add_argument("--working-level", choices=list(FLOORS), default="d2",
                        help="the default level of the working layers: the base the working address is checked at")
    parser.add_argument("--reach", choices=sorted(REACHES), default="equal",
                        help="f D for every zone (rule 1), or f D split by the zones' own widths")
    parser.add_argument("--out", type=Path, default=Path("runs/strategies"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--attention", choices=list(PLANS), default=SPLIT.name)
    parser.add_argument("--coverage-only", action="store_true",
                        help="the layouts' zones, shares of the network and bytes, without answering")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    model_id = MODELS[args.model]
    directory = refocustensors.model_directory(model_id, args.base) if args.base else None
    bench = Bench.load(model_id, gpu_share=args.gpu_share, directory=directory)
    tokenizer, ctl = bench.tokenizer, bench.ctl
    fmt = fm.prompt_format(model_id, tokenizer)
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    small = config.read(args.corpus_config, config.SmallCorpus)
    laid, calibration, unknown, askings = [], [], set(), {}
    for corpus in args.corpora:
        corpus_rows, source = corpora.read(corpus)
        frozen = read_frozen(args.frozen / f"{corpus}.json")
        frozen.check(name, source.revision, frozen.prompt)  # the frozen file names the setup it was frozen in
        known, background = split_shares(list(frozen.kept), small.seed, (small.share, small.calibration_share),
                                         small.floor)
        [strangers] = split_shares(list(frozen.unknown_share), small.seed, (small.unknown_share,))
        by_id = {r.id: r for r in corpus_rows}
        laid += [(corpus, by_id[i]) for i in known + strangers]
        unknown |= {(corpus, i) for i in strangers}
        calibration += [(corpus, by_id[i]) for i in background]
        setup = setup_named(corpus, frozen.prompt)
        train = corpora.read_train(corpus, TRAIN_POOL) if needs_train(setup) else []
        askings[corpus] = Asking(corpus, source.revision, name, Level.BF16, setup,
                                 examples_for(corpus, setup, train, small.seed))

    def prompts(pairs: list) -> list[str]:  # the prompt each question is answered with is the one its mask reads
        return [askings[c].prompts(fmt, [r])[0] for c, r in pairs]

    mask_source = bench.source(args.source)
    masks = bench.masks(prompts(laid) + prompts(calibration), [mask_source])[mask_source.name]
    modules = ctl.modules.values()
    inputs = Inputs(scores=masks[: len(laid)], calibration=masks[len(laid):],
                    block_weights=np.concatenate([m.block_sizes() * m.in_features for m in modules]),
                    domains=tuple(c for c, _ in laid),
                    model_weights={n: m.weight for n, m in ctl.modules.items()},
                    n_heads=bench.model.config.get_text_config(decoder=True).num_attention_heads)
    knobs = Knobs(FLOORS[args.floor], args.focus_area, args.focus_strength, args.combine)
    label = (f"{args.mechanism}-{args.source}-{args.reach}-{args.floor}-f{args.focus_area:g}-g{args.focus_strength:g}"
             f"-w{args.working_layers}{args.working_level}")
    policy = mechanism_layout(args.mechanism, inputs, knobs, graph=args.graph, k=args.k, reach=args.reach)
    working = WorkingLayers(policy, block_layers(ctl) < args.working_layers, FLOORS[args.working_level])
    regulator = Regulator(working, ctl)
    reading = regulator.reading({(c, r.id): i for i, (c, r) in enumerate(laid)}, label)

    codes = regulator.layout(np.arange(len(laid)))
    uniform = {lv.name.lower(): int(regulator.cost.read_bytes(np.full(ctl.n_blocks, int(lv), dtype=np.uint8))[0])
               for lv in regulator.ladder}
    rows = question_coverage(policy, codes, knobs.floor, inputs.block_weights, regulator.cost.read_bytes(codes))
    found = run_coverage(rows, uniform)
    covered = {"run": found, "questions": [{"corpus": c, "id": r.id, "unknown": (c, r.id) in unknown, **row}
                                           for (c, r), row in zip(laid, rows)]}
    print(f"{label}: lifted share median {found['lifted_share']['median']:.3f}, p90 {found['lifted_share']['p90']:.3f}, "
          f"max {found['lifted_share']['max']:.3f}; over half the network {found['over_half']} of "
          f"{found['questions']}; zones {found.get('zones')}; levels (median share) "
          + ", ".join(f"{lv} {s['median']:.3f}" for lv, s in found["levels"].items()), flush=True)

    decoder = StaticDecoder(attention=PLANS[args.attention], prefill_tokens=PREFILL_TOKENS)
    judge = ModelJudge(bench.model, tokenizer, ctl, fmt, decoder)
    out = args.out / args.model
    with GpuMonitor() as gpu:
        for corpus, asking in askings.items() if not args.coverage_only else ():
            asking = replace(asking, reading=reading)
            rows = [r for c, r in laid if c == corpus]
            for chunk in asking.batches(fmt, tokenizer, rows):
                append_answers(answers_path(out / "answers", label, corpus),
                               asking.answer(bench.model, tokenizer, ctl, fmt, judge, chunk, bench.throttle, decoder))

    target = out / f"{'coverage' if args.coverage_only else 'summary'}-{label}.json"
    write_json(target, {
        "model": name, "base": args.base, "mechanism": args.mechanism, "source": args.source, "graph": args.graph,
        "k": args.k, "floor": args.floor, "focus_area": args.focus_area, "focus_strength": args.focus_strength,
        "combine": args.combine, "reach": args.reach, "corpus": small.__dict__,
        "questions": {c: [r.id for cc, r in laid if cc == c and (cc, r.id) not in unknown] for c in askings},
        "unknown": {c: [i for cc, i in sorted(unknown) if cc == c] for c in askings},
        "calibration": len(calibration),
        "bytes": {"per_question_mean": float(regulator.cost.read_bytes(codes).mean()),
                  "per_step_mean": (float(np.mean([regulator.cost.step_bytes(c) for c in reading.laid]))
                                    if reading.laid else None),
                  "uniform": uniform},
        "coverage": covered,
        "by_layer": regulator.by_layer(codes), "gpu": gpu.summary(), "pacer": bench.throttle.stats(),
    })
    print(f"written {target}", flush=True)
    return target


if __name__ == "__main__":
    main()
