"""Does a question tell at a shallow depth that it needs a deeper one (foqlens.address.adaptive_depth).

Everything the run does is its configuration (config.AdaptiveCheck, configs/address-adaptive.toml). Per corpus, the
full pass at bf16 and the working pass at the base over the small corpus's calibration and laid-out questions; every
depth predicts the laid-out questions' address from the deepest depth on, so that depths differ only in what they
read; then per question its hit and margin at every depth, and the policy between the configured shallow and deep one.

    uv run python scripts/address_adaptive.py --config configs/address-adaptive.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from foqlens import config, corpora, refocustensors
from foqlens import model as fm
from foqlens.address import adaptive_depth, depth_cap, predict_deep, silhouette_stop, stop_policy
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
    parser.add_argument("--out", type=Path, default=Path("runs/address"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    check = config.read(args.config, config.AdaptiveCheck)
    source_name = config.choose(check.source, ADDRESS_SOURCES, "mask source")
    model_id = MODELS[args.model]
    directory = refocustensors.model_directory(model_id, check.base) if check.base else None
    bench = Bench.load(model_id, gpu_share=args.gpu_share, directory=directory)
    fmt = fm.prompt_format(model_id, bench.tokenizer)
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    layers = block_layers(bench.ctl)
    base_level = Level[check.base_level.upper()]
    target_from = max(check.depths)
    made = bench.source(source_name)

    summary = {"model": name, "config": str(args.config), "check": check.__dict__ | {"corpus_overrides": dict(
        check.corpus_overrides)}, "corpora": {}}
    target = args.out / args.model / f"{args.config.stem}.json"
    with GpuMonitor() as gpu:
        for corpus in check.corpora:
            small = config.read(Path(check.corpus_overrides.get(corpus, check.corpus)), config.SmallCorpus)
            rows, source = corpora.read(corpus)
            frozen = read_frozen(args.frozen / f"{corpus}.json")
            frozen.check(name, source.revision, frozen.prompt)
            laid, calibration = split_shares(list(frozen.kept), small.seed, (small.share, small.calibration_share),
                                             small.floor)
            by_id = {r.id: r for r in rows}
            asking = Asking(corpus, source.revision, name, Level.BF16, setup_named(corpus, frozen.prompt), ())
            prompts = [asking.prompts(fmt, [by_id[i]])[0] for i in calibration + laid]
            n = len(calibration)
            full = bench.masks(prompts, [made])[made.name]
            work = bench.masks(prompts, [made], level=base_level)[made.name]
            predicted = {d: predict_deep(work[:n], full[:n], work[n:], layers, d, target_from, check.ridge)
                         for d in check.depths}
            actual = full[n:, layers >= target_from]
            found = adaptive_depth(predicted, actual, check.low, check.high)
            found["silhouette"] = {}
            for share in check.cap_shares:
                cap = depth_cap(int(layers.max()) + 1, share)
                stops = silhouette_stop(predicted, check.silhouette, check.tolerance, check.patience, cap)
                found["silhouette"][share] = {"cap": cap, **stop_policy(predicted, actual, stops, check.patience)}
            summary["corpora"][corpus] = {"laid": len(laid), "calibration": n, **found}
            print(f"{corpus} ({len(laid)} laid): hits {found['hits']}; rescue AUC margin "
                  f"{found['rescue_auc_margin']:.3f}, moved {found['rescue_auc_moved']:.3f}", flush=True)
            for share, policy in found["silhouette"].items():
                print(f"  cap {share:g} ({policy['cap']} layers): identified {policy['identified']:.3f} at mean stop "
                      f"{policy['mean_stop']:.2f}, read {policy['mean_read']:.2f}, stops {policy['stops']}; fixed "
                      f"{policy['fixed_depth']}: {policy['fixed_identified']:.3f}", flush=True)
            write_json(target, summary)
    summary["gpu"] = gpu.summary()
    write_json(target, summary)
    print(f"written {target}", flush=True)
    return target


if __name__ == "__main__":
    main()
