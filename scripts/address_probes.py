"""How deep sets of questions written for it read under the silhouette rule (foqlens.address.stop_summary).

Everything the run does is its configuration (config.ProbeCheck, configs/address-probes.toml). The calibration
questions of every corpus together, read in full at bf16 and working at the base, fit one projection per depth; each
probe, asked in the frozen wrapper of one of those corpora as its questions are, is read the same two ways; every depth predicts
its address from the deepest depth on, and the silhouette rule picks its stop. Per probe its stop, whether it hit the
cap and whether the address at its stop finds its own real one; per set the stops; between the sets, whether the hard
ones stop deeper (AUC).

    uv run python scripts/address_probes.py --config configs/address-probes.toml
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from foqlens import config, corpora, refocustensors
from foqlens import model as fm
from foqlens.address import (depth_cap, per_question, predict_deep, rank_auc, silhouette_overlap,
                             silhouette_run, stop_summary)
from foqlens.answering import Asking
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.io import read_frozen, read_jsonl, write_json
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


def calibration_prompts(check: config.ProbeCheck, fmt, name: str, frozen_dir: Path) -> tuple[list[str], dict]:
    """Every corpus's calibration questions in its frozen wrapper, the corpora one after another; and every corpus's
    asking, so that a probe is put the way its corpus's questions are."""
    prompts, found = [], {}
    for corpus in check.corpora:
        small = config.read(Path(check.corpus_overrides.get(corpus, check.corpus)), config.SmallCorpus)
        rows, source = corpora.read(corpus)
        frozen = read_frozen(frozen_dir / f"{corpus}.json")
        frozen.check(name, source.revision, frozen.prompt)
        _, calibration = split_shares(list(frozen.kept), small.seed, (small.share, small.calibration_share),
                                      small.floor)
        by_id = {r.id: r for r in rows}
        found[corpus] = Asking(corpus, source.revision, name, Level.BF16, setup_named(corpus, frozen.prompt), ())
        prompts += found[corpus].prompts(fmt, [by_id[i] for i in calibration])
    return prompts, found


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    check = config.read(args.config, config.ProbeCheck)
    source_name = config.choose(check.source, ADDRESS_SOURCES, "mask source")
    model_id = MODELS[args.model]
    directory = refocustensors.model_directory(model_id, check.base) if check.base else None
    bench = Bench.load(model_id, gpu_share=args.gpu_share, directory=directory)
    fmt = fm.prompt_format(model_id, bench.tokenizer)
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    layers = block_layers(bench.ctl)
    base_level = Level[check.base_level.upper()]
    target_from = max(check.depths)
    cap = depth_cap(int(layers.max()) + 1, check.cap_share)
    made = bench.source(source_name)

    sets = {s: read_jsonl(Path(path)) for s, path in check.probes.items()}
    probes = [(s, row) for s, rows in sets.items() for row in rows]
    prompts, asked = calibration_prompts(check, fmt, name, args.frozen)
    # a probe is asked as its corpus's questions are, so that the projection meets the wrapper it was fitted on
    prompts += asked[config.choose(check.probe_corpus, asked, "probe corpus")].prompts(
        fmt, [corpora.Row(r["id"], r["question"], (), None) for _, r in probes])
    n = len(prompts) - len(probes)
    target = args.out / args.model / f"{args.config.stem}.json"
    with GpuMonitor() as gpu:
        full = bench.masks(prompts, [made])[made.name]
        work = bench.masks(prompts, [made], level=base_level)[made.name]
        predicted = {d: predict_deep(work[:n], full[:n], work[n:], layers, d, target_from, check.ridge)
                     for d in check.depths}
        stops, settled = silhouette_run(predicted, check.silhouette, check.tolerance, check.patience, cap)
        overlap = silhouette_overlap(predicted, check.silhouette)
        hits = {d: per_question(predicted[d], full[n:, layers >= target_from])["hit"] for d in check.depths}
    in_set = {s: np.array([p == s for p, _ in probes]) for s in sets}
    summary = {
        "model": name, "config": str(args.config), "calibration": n, "cap": cap,
        "check": check.__dict__ | {"probes": dict(check.probes), "corpus_overrides": dict(check.corpus_overrides)},
        "probes": [{"set": s, "id": r["id"], "stop": int(stops[q]), "capped": bool(not settled[q]),
                    "identified": bool(hits[int(stops[q])][q])} for q, (s, r) in enumerate(probes)],
        "sets": {s: stop_summary(stops[m], settled[m]) for s, m in in_set.items()},
        # the silhouette's mean Jaccard with the depth before, per set: where the rule sees the address settle
        "overlap": {s: {int(d): float(o[m].mean()) for d, o in overlap.items()} for s, m in in_set.items()},
        # every later set against every earlier one: how often its probe stops deeper, ties half; 0.5 is no difference
        "deeper_auc": {f"{b} > {a}": rank_auc(stops[in_set[a] | in_set[b]].astype(float),
                                              in_set[b][in_set[a] | in_set[b]]) for a, b in combinations(sets, 2)},
        "gpu": gpu.summary(),
    }
    for s, found in summary["sets"].items():
        print(f"{s} ({found['questions']}): mean stop {found['mean_stop']:.2f}, stops {found['stops']}, "
              f"capped {found['capped_share']:.2f}", flush=True)
        print("  Jaccard with the depth before: " + ", ".join(f"{d}: {o:.2f}" for d, o in summary["overlap"][s].items()),
              flush=True)
    for pair, auc in summary["deeper_auc"].items():
        print(f"stops deeper, {pair}: AUC {auc:.3f}", flush=True)
    write_json(target, summary)
    print(f"written {target}", flush=True)
    return target


if __name__ == "__main__":
    main()
