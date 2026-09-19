"""How deep the working address must read (foqlens.address.deep_address): the first N layers against the rest.

Everything the run does is its configuration (config.DepthCheck, configs/address-depth.toml). Per corpus and source,
two passes over the small corpus's calibration and laid-out questions in the frozen wrapper: the full one at bf16 and
the working one at the base precision. A layer's activations do not depend on the layers after it, so one working
pass gives every depth: the first N layers are a slice of it. For every N a projection fitted on the calibration
questions predicts the laid-out questions' address in the layers after N, and the prediction is identified against
the real one; beside it, the share of the weights the zones would act on.

    uv run python scripts/address_depth.py --config configs/address-depth.toml
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from foqlens import config, corpora, refocustensors
from foqlens import model as fm
from foqlens.address import deep_address
from foqlens.answering import Asking
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.io import read_frozen, write_json
from foqlens.pipeline import ADDRESS_SOURCES, Bench
from foqlens.prompt_variants import setup_named
from foqlens.quant import Level
from foqlens.regulator import block_layers
from foqlens.selection import split_shares

MODELS = {"e2b-it": fm.E2B_IT, "e4b-it": fm.E4B_IT}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--sources", nargs="+", help="override the configuration's sources; the summary is named by them")
    parser.add_argument("--corpora", nargs="+", help="override the configuration's corpora; the summary is named by them")
    parser.add_argument("--corpus-config", type=Path, help="override the small corpus; the summary is named by its file")
    parser.add_argument("--out", type=Path, default=Path("runs/address"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    check = config.read(args.config, config.DepthCheck)
    if args.sources:
        check = replace(check, sources=tuple(args.sources))
    if args.corpora:
        check = replace(check, corpora=tuple(args.corpora))
    small = config.read(args.corpus_config or Path(check.corpus), config.SmallCorpus)
    sources = [config.choose(s, ADDRESS_SOURCES, "mask source") for s in check.sources]
    model_id = MODELS[args.model]
    directory = refocustensors.model_directory(model_id, check.base) if check.base else None
    bench = Bench.load(model_id, gpu_share=args.gpu_share, directory=directory)
    fmt = fm.prompt_format(model_id, bench.tokenizer)
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    layers = block_layers(bench.ctl)
    weights = np.concatenate([m.block_sizes() * m.in_features for m in bench.ctl.modules.values()])
    base_level = Level[check.base_level.upper()]

    summary = {"model": name, "config": str(args.config), "check": check.__dict__, "corpus": small.__dict__,
               "sources": {s: {} for s in sources}}
    overrides = [*(args.sources or []), *(args.corpora or []), *([args.corpus_config.stem] if args.corpus_config else [])]
    named = "-".join([args.config.stem, *overrides])
    target = args.out / args.model / f"{named}.json"
    with GpuMonitor() as gpu:
        for corpus in check.corpora:
            rows, source = corpora.read(corpus)
            frozen = read_frozen(args.frozen / f"{corpus}.json")
            frozen.check(name, source.revision, frozen.prompt)
            laid, calibration = split_shares(list(frozen.kept), small.seed, (small.share, small.calibration_share),
                                             small.floor)
            by_id = {r.id: r for r in rows}
            asking = Asking(corpus, source.revision, name, Level.BF16, setup_named(corpus, frozen.prompt), ())
            prompts = [asking.prompts(fmt, [by_id[i]])[0] for i in calibration + laid]
            n_train = len(calibration)
            for source_name in sources:
                made = bench.source(source_name)
                full = bench.masks(prompts, [made])[made.name]
                work = bench.masks(prompts, [made], level=base_level)[made.name]
                curves = {ridge: {depth: deep_address(work[:n_train], full[:n_train], work[n_train:], full[n_train:],
                                                      layers, weights, depth, ridge) for depth in check.depths}
                          for ridge in check.ridges}
                summary["sources"][source_name][corpus] = curves
                for ridge, curve in curves.items():
                    print(f"{source_name} {corpus} ridge {ridge:g} ({len(laid)} laid, {n_train} calibration): " +
                          ", ".join(f"N={d} {r['identified']:.3f} (zoned {r['zoned_weight_share']:.2f})"
                                    for d, r in curve.items()), flush=True)
                write_json(target, summary)  # a source at a time: a later failure keeps what is measured
    summary["gpu"] = gpu.summary()
    write_json(target, summary)
    print(f"written {target}", flush=True)
    return target


if __name__ == "__main__":
    main()
