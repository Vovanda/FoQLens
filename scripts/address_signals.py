"""Is a mask source an address (foqlens.address): the same questions read in two wrappers, and cheaply at the base.

Everything the run does is its configuration (config.AddressCheck, configs/address.toml): the questions are the small
corpus's laid-out ones - the same seeded draw as strategy_answers - in the corpora with a two-example wrapper beside
the frozen one. Per source and corpus: identification between the frozen wrapper and the two-example one at bf16;
identification between the working reading (its first layers at the base precision) and the full pass at bf16 on the
same blocks; the layer profile of the excess.

    uv run python scripts/address_signals.py --config configs/address.toml
    uv run python scripts/address_signals.py --config configs/address.toml --sources pooled --corpora triviaqa \
        --corpus-config configs/smoke-corpus.toml  # smoke
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from foqlens import config, corpora, refocustensors
from foqlens import model as fm
from foqlens.address import USABLE, assess, working_address
from foqlens.answering import Asking
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.io import read_frozen, write_json
from foqlens.pipeline import ADDRESS_SOURCES, Bench
from foqlens.prompt_variants import TRAIN_POOL, examples_for, needs_train, setup_named
from foqlens.quant import Level
from foqlens.regulator import block_layers
from foqlens.selection import split_shares

MODELS = {"e2b-it": fm.E2B_IT, "e4b-it": fm.E4B_IT}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--sources", nargs="+", help="override the configuration's sources (a smoke)")
    parser.add_argument("--corpora", nargs="+", help="override the configuration's corpora (a smoke)")
    parser.add_argument("--corpus-config", type=Path, help="override the configuration's small corpus (a smoke)")
    parser.add_argument("--out", type=Path, default=Path("runs/address"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    return parser.parse_args(argv)


def prompts_in_wrappers(check: config.AddressCheck, corpus_config: config.SmallCorpus, frozen_dir: Path, name: str,
                        fmt, corpora_names: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    """Per corpus, the laid-out questions' prompts in the frozen wrapper and in the two-example one."""
    out = {}
    for corpus in corpora_names:
        rows, source = corpora.read(corpus)
        frozen = read_frozen(frozen_dir / f"{corpus}.json")
        frozen.check(name, source.revision, frozen.prompt)
        [known] = split_shares(list(frozen.kept), corpus_config.seed, (corpus_config.share,), corpus_config.floor)
        by_id = {r.id: r for r in rows}
        laid = [by_id[i] for i in known]
        pair = []
        for wrapper in (frozen.prompt, check.two_shot[corpus]):
            setup = setup_named(corpus, wrapper)
            train = corpora.read_train(corpus, TRAIN_POOL) if needs_train(setup) else []
            asking = Asking(corpus, source.revision, name, Level.BF16, setup,
                            examples_for(corpus, setup, train, corpus_config.seed))
            pair.append([asking.prompts(fmt, [r])[0] for r in laid])
        out[corpus] = (pair[0], pair[1])
    return out


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    check = config.read(args.config, config.AddressCheck)
    if args.sources:
        check = replace(check, sources=tuple(args.sources))
    sources = [config.choose(s, ADDRESS_SOURCES, "mask source") for s in check.sources]
    corpus_config = config.read(args.corpus_config or Path(check.corpus), config.SmallCorpus)
    model_id = MODELS[args.model]
    directory = refocustensors.model_directory(model_id, check.base) if check.base else None
    bench = Bench.load(model_id, gpu_share=args.gpu_share, directory=directory)
    fmt = fm.prompt_format(model_id, bench.tokenizer)
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    layers = block_layers(bench.ctl)
    base_level = Level[check.base_level.upper()]
    wrappers = prompts_in_wrappers(check, corpus_config, args.frozen, name, fmt, args.corpora or list(check.two_shot))

    summary = {"model": name, "config": str(args.config), "check": check.__dict__ | {"two_shot": dict(check.two_shot)},
               "corpus": corpus_config.__dict__, "usable": USABLE, "sources": {}}
    named = f"{args.config.stem}-{'-'.join(check.sources)}" if args.sources else args.config.stem
    target = args.out / args.model / f"{named}.json"
    with GpuMonitor() as gpu:
        for source_name in sources:
            reads_layers = ADDRESS_SOURCES[source_name].reads_layers
            full = bench.source(source_name)
            # a source that reads every layer has one working reading: the whole pass at the base
            depths = check.working_layers if reads_layers else (int(layers.max()) + 1,)
            summary["sources"][source_name] = {}
            for corpus, (frozen_prompts, two_shot_prompts) in wrappers.items():
                frozen = bench.masks(frozen_prompts, [full])[full.name]
                report = assess(frozen, bench.masks(two_shot_prompts, [full])[full.name], layers)
                report["working"] = {}
                for depth in depths:
                    read = set(range(depth))
                    cheap = bench.source(source_name, layers=read if reads_layers else None)
                    work = bench.masks(frozen_prompts, [cheap], level=base_level)[cheap.name]
                    report["working"][depth] = working_address(frozen, work, np.isin(layers, list(read)))
                summary["sources"][source_name][corpus] = report
                curve = ", ".join(f"{d}: {r['identified']:.3f}/{r['own_cos']:.2f}" for d, r in report["working"].items())
                print(f"{source_name} {corpus}: wrappers {report['wrappers']['identified']:.3f}; working by depth "
                      f"(identified/own cos) {curve}; chance {report['wrappers']['chance']:.4f}", flush=True)
            write_json(target, summary)  # a source at a time: a later failure keeps what is measured
    summary["gpu"] = gpu.summary()
    write_json(target, summary)
    print(f"written {target}", flush=True)
    return target


if __name__ == "__main__":
    main()
