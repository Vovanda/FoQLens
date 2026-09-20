"""The answers at the precision fields on the small corpus (config.OracleAnswers): every laid-out question read at every
field's layout and the reference's, and at the uniform rungs, beside the answer of every block at the oracle by trying's
high level in the same batches. Does a layout answer as the whole network does (the same text, and where not, read by
hand - no judge), and at what bytes against the uniform ladder.

    uv run python scripts/oracle_answers.py --config configs/precision-answers.toml --base bartowski-Q2_K --shard 1
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
from foqlens.group_oracle import RUN_FIELDS, block_groups
from foqlens.io import answers_path, append_answers, read_npz_parts, write_json, written_ids
from foqlens.judging import NotJudged
from foqlens.layouts import GivenLevels
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS
from foqlens.quant import Level
from foqlens.regulator import Regulator
from foqlens.small_corpus import draw, named, pick_shard

MODEL = fm.E2B_IT
LOG = logging.getLogger("foqlens.oracle_answers")
NOT_READ = 255  # a question a field was not read for (precision_fields.py): it is read at the top whole


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", choices=sorted(PUBLISHED), required=True, help="the base the oracle was run over")
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--corpus-config", type=Path, default=Path("configs/small-corpus.toml"))
    parser.add_argument("--also", type=Path, default=None,
                        help="a json list of [corpus, id]: lay these questions out on top of the share")
    parser.add_argument("--out", type=Path, default=Path("runs/oracles/e2b-it"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--attention", choices=list(PLANS), default=SPLIT.name)
    parser.add_argument("--shard", type=int, default=None, help="part 1..shards of the draw (SmallCorpus.shards)")
    args = parser.parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    runlog.setup(args.out / "logs" / f"oracle_answers-{run}.jsonl", {"run": run, "script": "oracle_answers"})
    check = config.read(args.config, config.OracleAnswers)
    # only the top rung is read here, and naming it keeps shards of two versions of the oracle joinable
    trying = read_npz_parts(check.group_oracle, RUN_FIELDS, only={"high"})
    high = Level[str(trying["high"]).upper()]

    bench = Bench.load(MODEL, gpu_share=args.gpu_share, directory=refocustensors.model_directory(MODEL, args.base))
    tokenizer, ctl = bench.tokenizer, bench.ctl
    fmt = fm.prompt_format(MODEL, tokenizer)
    name = f"{MODEL}@{fm.REVISIONS[MODEL][:8]}"
    small = config.read(args.corpus_config, config.SmallCorpus)
    found, suffix = pick_shard(draw(args.corpora, args.frozen, small, name, also=named(args.also)), small, args.shard)
    # the rungs' ratios are measured inside a shard on its own questions, so two shards honestly differ in them
    # and joining them is refused; the answers do not read them
    fields = read_npz_parts(check.precision, RUN_FIELDS | {"sources", "ratios", "rungs"},
                            skip={"ratios", "rungs"})
    at = {k: i for i, k in enumerate(zip(fields["corpus"].tolist(), fields["ids"].tolist()))}
    laid = [(c, r) for c, r in found.laid if (c, r.id) in at]
    if len(laid) < len(found.laid):
        LOG.warning("%d questions the fields were not read for are left out", len(found.laid) - len(laid))
    held = [at[(c, r.id)] for c, r in laid]
    groups, names = block_groups(ctl)
    if names != fields["groups"].tolist():
        raise ValueError("the fields' groups are not this model's")

    decoder = StaticDecoder(attention=PLANS[args.attention], prefill_tokens=PREFILL_TOKENS)
    # no judge (Volodya 20.09): the question is whether a layout answers as the whole network at `high` does - the
    # same text in the same batches - and where it does not, the answers are read by hand
    judge = NotJudged()
    index = {(c, r.id): i for i, (c, r) in enumerate(laid)}
    # every variant: a label and its layouts [questions, n_blocks]. The whole network first, then the uniform rungs
    # below it - the ladder the fields are measured against - then every field. The labels name no run, so a rerun
    # reads the answers it already has
    variants = [(f"everything-{lv.name.lower()}-{args.base}", np.full((len(laid), len(groups)), int(lv), np.uint8))
                for lv in (high, Level.D6, Level.D4, Level.D2)]
    for source in fields["sources"].tolist() + (["reference"] if "levels_reference" in fields else []):
        levels = fields[f"levels_{source}"][held]
        has = (levels != NOT_READ).all(axis=1)
        variants.append((f"precision-{source}-{args.base}-t{float(fields['tolerance']):g}",
                         np.where(has[:, None], levels[:, groups], int(high)).astype(np.uint8)))
    targets = []
    progress = Progress(len(variants), "layout")
    for label, layouts in variants:
        regulator = Regulator(GivenLevels(label, layouts), ctl)
        reading = regulator.reading(index, label)
        codes = regulator.layout(np.arange(len(laid)))
        read = regulator.cost.read_bytes(codes)
        uniform = {lv.name.lower(): int(regulator.cost.read_bytes(np.full(ctl.n_blocks, int(lv), dtype=np.uint8))[0])
                   for lv in regulator.ladder}
        LOG.info("%s: bytes %.3f of %s's", label, float(read.mean() / uniform[high.name.lower()]), high.name,
                 extra={"layout": label})
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
            "model": name, "base": args.base, "precision": check.precision, "high": high.name, "corpus": small.__dict__,
            "questions": [{"corpus": c, "id": r.id, "unknown": (c, r.id) in found.unknown, "bytes": int(b)}
                          for (c, r), b in zip(laid, read)],
            "bytes": {"per_question_mean": float(read.mean()), "uniform": uniform},
            "by_layer": regulator.by_layer(codes), "gpu": gpu.summary(), "pacer": bench.throttle.stats(),
        })
        LOG.info(progress.step(f"{label} written {target}"), extra={"layout": label, "file": str(target)})
        targets.append(target)
    return targets


if __name__ == "__main__":
    main()
