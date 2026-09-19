"""The answers at the oracles' minimal masks on the small corpus (config.OracleAnswers): every laid-out question read at
its own minimal mask - its first groups by lift at the oracle's high level, the rest at its low - beside the answer of
every block at the high level in the same batches. Does the ideal answer as the whole network does (the same text, and
where not, read by hand - no judge), and at what bytes against the uniform ladder.

One layout per tolerance: the minimal mask is chosen again from the prefix sweep the oracle kept, the model is never
asked for it twice.

    uv run python scripts/oracle_answers.py --config configs/oracle-answers.toml --base bartowski-Q2_K
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from foqlens import config, refocustensors, runlog
from foqlens import model as fm
from foqlens.attention import PLANS, SPLIT
from foqlens.gguf_weights import PUBLISHED
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.graph_decode import PREFILL_TOKENS, StaticDecoder
from foqlens.group_oracle import RUN_FIELDS, block_groups, minimal_layouts, minimal_prefix
from foqlens.io import answers_path, append_answers, read_npz_parts, write_json, written_ids
from foqlens.judging import NotJudged
from foqlens.layouts import GivenLevels
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS
from foqlens.quant import Level
from foqlens.regulator import Regulator
from foqlens.small_corpus import draw, pick_shard

MODEL = fm.E2B_IT
LOG = logging.getLogger("foqlens.oracle_answers")


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", choices=sorted(PUBLISHED), required=True, help="the base the oracle was run over")
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--corpus-config", type=Path, default=Path("configs/small-corpus.toml"))
    parser.add_argument("--out", type=Path, default=Path("runs/oracles/e2b-it"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--attention", choices=list(PLANS), default=SPLIT.name)
    parser.add_argument("--shard", type=int, default=None, help="part 1..shards of the draw (SmallCorpus.shards)")
    args = parser.parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    runlog.setup(args.out / "logs" / f"oracle_answers-{run}.jsonl", {"run": run, "script": "oracle_answers"})
    check = config.read(args.config, config.OracleAnswers)
    trying = read_npz_parts(check.group_oracle, RUN_FIELDS)
    low, high = Level[str(trying["low"]).upper()], Level[str(trying["high"]).upper()]

    bench = Bench.load(MODEL, gpu_share=args.gpu_share, directory=refocustensors.model_directory(MODEL, args.base))
    tokenizer, ctl = bench.tokenizer, bench.ctl
    fmt = fm.prompt_format(MODEL, tokenizer)
    name = f"{MODEL}@{fm.REVISIONS[MODEL][:8]}"
    small = config.read(args.corpus_config, config.SmallCorpus)
    found, suffix = pick_shard(draw(args.corpora, args.frozen, small, name), small, args.shard)
    # the oracle's questions by key: a whole run, a check run (--limit) or shards all serve any shard of the draw
    row = {k: i for i, k in enumerate(zip(trying["corpus"].tolist(), trying["ids"].tolist()))
           if not np.isnan(trying["prefix"][i]).any()}
    laid = [(c, r) for c, r in found.laid if (c, r.id) in row]
    if len(laid) < len(found.laid):
        LOG.warning("%d questions the oracle did not try are left out", len(found.laid) - len(laid))
    rows = [row[(c, r.id)] for c, r in laid]
    groups, names = block_groups(ctl)
    if names != trying["groups"].tolist():
        raise ValueError("the oracle's groups are not this model's")
    lift, ends, prefix = trying["lift"][rows], trying["ends"][rows], trying["prefix"][rows]
    sweeps = np.concatenate([ends[:, :1], prefix], axis=1)  # k groups lifted -> the answer's NLL, k = 0..g

    decoder = StaticDecoder(attention=PLANS[args.attention], prefill_tokens=PREFILL_TOKENS)
    # no judge (Volodya 20.09): the question is whether the ideal answers as the whole network at `high` does - the
    # same text in the same batches - and where it does not, the answers are read by hand
    judge = NotJudged()
    index = {(c, r.id): i for i, (c, r) in enumerate(laid)}
    # the reference first: every group lifted is every block at `high`, answered in the batches the ideals are - its
    # label names no oracle, so the run over another base reads the answers it already has
    variants = [(f"everything-{high.name.lower()}-{args.base}", None, np.full(len(laid), len(names)))] + [
        (f"minimal-{args.base}-{low.name.lower()}{high.name.lower()}-t{t:g}", t,
         np.array([minimal_prefix(s, e, t) for s, e in zip(sweeps, ends[:, 1])])) for t in check.tolerances]
    targets = []
    progress = Progress(len(variants), "layout")
    for label, tolerance, minimal in variants:
        regulator = Regulator(GivenLevels(label, minimal_layouts(groups, lift, minimal, low, high)), ctl)
        reading = regulator.reading(index, label)
        codes = regulator.layout(np.arange(len(laid)))
        read = regulator.cost.read_bytes(codes)
        uniform = {lv.name.lower(): int(regulator.cost.read_bytes(np.full(ctl.n_blocks, int(lv), dtype=np.uint8))[0])
                   for lv in regulator.ladder}
        LOG.info("%s: minimal mask median %d of %d groups, bytes %.3f of %s's", label, int(np.median(minimal)),
                 len(names), float(read.mean() / uniform[high.name.lower()]), high.name,
                 extra={"layout": label, "minimal_median": float(np.median(minimal))})
        with GpuMonitor() as gpu:
            for corpus, asking in found.askings.items():
                asking = replace(asking, reading=reading)
                path = answers_path(args.out / "answers", label, corpus)
                done = written_ids(path)  # a stopped or crashed run answers only what its file does not hold yet
                chunks = asking.batches(fmt, tokenizer, [r for c, r in laid if c == corpus and r.id not in done])
                answered = Progress(len(chunks), f"{label} {corpus} answer batch")
                for chunk in chunks:
                    judged = asking.answer(bench.model, tokenizer, ctl, fmt, judge, chunk, bench.throttle, decoder)
                    append_answers(path, judged)
                    LOG.info(answered.step(f"{len(chunk)} questions"), extra={"layout": label, "corpus": corpus})
        target = args.out / f"summary-{label}{suffix}.json"  # the answers of the shards gather in one file per corpus
        write_json(target, {
            "model": name, "base": args.base, "group_oracle": check.group_oracle, "tolerance": tolerance,
            "low": low.name, "high": high.name, "corpus": small.__dict__,
            "questions": [{"corpus": c, "id": r.id, "unknown": (c, r.id) in found.unknown, "minimal": int(k),
                           "bytes": int(b)} for (c, r), k, b in zip(laid, minimal, read)],
            "bytes": {"per_question_mean": float(read.mean()), "uniform": uniform},
            "by_layer": regulator.by_layer(codes), "gpu": gpu.summary(), "pacer": bench.throttle.stats(),
        })
        LOG.info(progress.step(f"{label} written {target}"), extra={"layout": label, "file": str(target)})
        targets.append(target)
    return targets


if __name__ == "__main__":
    main()
