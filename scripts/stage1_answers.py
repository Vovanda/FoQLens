"""Stage 1 of the corpus selection, and stage 2 with --level: every corpus answered in its setup, in rounds.

The run goes through all corpora at once in rounds (foqlens.schedule), each round a share of every
corpus, so a run stopped anywhere leaves every corpus equally far. Every answer is appended to
<out>/<model>/answers/<level>/<corpus>.jsonl as soon as its batch is written; a restart reads those
files and asks only what is missing. Within a round the questions go in batches by prompt length.
The questions spent on choosing the prompt (--tuning, the summary of scripts/prompt_tuning.py) are left out.

    uv run python scripts/stage1_answers.py --tuning runs/reference/prompt-tuning/e2b-it/summary.json
    uv run python scripts/stage1_answers.py --level d4 --tuning ...        # stage 2 at one level
    uv run python scripts/stage1_answers.py --rounds 1 --out /tmp/stage1   # smoke check: one round
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foqlens import corpora
from foqlens import model as fm
from foqlens.answering import JUDGE_BATCH, Asking, level_label
from foqlens.generation import DYNAMIC
from foqlens.gpu_monitor import GpuMonitor
from foqlens.graph_decode import STATIC
from foqlens.gpu_share import default_share
from foqlens.io import answers_path, append_answers, write_json, written_ids
from foqlens.judging import ModelJudge
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS, examples_for, needs_train, setup_named
from foqlens.quant import Level
from foqlens.schedule import Schedule

MODELS = {"e2b-it": fm.E2B_IT, "e4b-it": fm.E4B_IT}
LEVELS = {level_label(level): level for level in (Level.BF16, Level.D8, Level.D6, Level.D4, Level.D2)}
TRAIN_POOL = 2000
# The setups frozen by the prompt tuning (runs/reference/prompt-tuning/e2b-it, 2026-09-14): where the
# setups tie within the noise of the tuning share, the one without examples, so that no answer hangs on
# the examples that happened to be drawn. TriviaQA short-0 EM 0.38 (best 0.40); NQ-open short-0 - the
# two draws of 2 examples differ by 6 points; SQuAD passage-0 EM 0.49 like the rest; ARC solve-0 -
# Claude's reading of 16 pairs, 10 right against 8-9 for solve-brief; HotpotQA justify EM 0.49, where
# justify-terse leaves out its answer line.
FROZEN_SETUPS = {"triviaqa": "short-0", "nq_open": "short-0", "squad_v2": "passage-0",
                 "arc_challenge_closed": "solve-0", "hotpotqa": "justify"}
# static: the static cache with every step a CUDA graph (foqlens.graph_decode); dynamic: the reference loop.
DECODERS = {"static": STATIC, "dynamic": DYNAMIC}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--level", choices=list(LEVELS), default="bf16")
    parser.add_argument("--setups", type=json.loads, default=FROZEN_SETUPS, help="JSON {corpus: setup name}")
    parser.add_argument("--tuning", type=Path, default=None, help="summary.json of prompt_tuning.py: its questions are left out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=None, help="stop after this many rounds, for a smoke check")
    parser.add_argument("--out", type=Path, default=Path("runs/reference/stage1"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--decoder", choices=list(DECODERS), default="static")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    model_id, level = MODELS[args.model], LEVELS[args.level]
    bench = Bench.load(model_id, gpu_share=args.gpu_share)
    tokenizer = bench.tokenizer
    fmt = fm.prompt_format(model_id, tokenizer)
    judge = ModelJudge.build(bench.model, tokenizer, bench.ctl, fmt, batch_size=JUDGE_BATCH)
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    out = args.out / args.model
    spent = json.loads(args.tuning.read_text(encoding="utf-8"))["corpora"] if args.tuning else {}

    rows, askings, paths = {}, {}, {}
    for corpus in args.corpora:
        corpus_rows, source = corpora.read(corpus)
        left_out = set(spent.get(corpus, {}).get("tuning_ids", []))
        rows[corpus] = {r.id: r for r in corpus_rows if r.id not in left_out}
        setup = setup_named(corpus, args.setups[corpus])
        train = corpora.read_train(corpus, TRAIN_POOL) if needs_train(setup) else []
        askings[corpus] = Asking(corpus, source.revision, name, level, setup, examples_for(corpus, setup, train, args.seed))
        paths[corpus] = answers_path(out / "answers", args.level, corpus)

    schedule = Schedule.build({c: list(r) for c, r in rows.items()}, args.seed)
    written = {c: written_ids(p) for c, p in paths.items()}
    rounds_done = 0
    with GpuMonitor() as gpu:
        progress = Progress(schedule.rounds, "round")
        for k, todo in schedule.pending(written):
            if args.rounds is not None and rounds_done == args.rounds:
                break
            for corpus, ids in todo.items():
                asking = askings[corpus]
                for chunk in asking.batches(fmt, tokenizer, [rows[corpus][i] for i in ids]):
                    append_answers(paths[corpus], asking.answer(bench.model, tokenizer, bench.ctl, fmt, judge,
                                                                chunk, bench.throttle, DECODERS[args.decoder]))
            rounds_done += 1
            print(progress.step(f"round {k}"), flush=True)

    summary = {
        "model": name, "level": args.level, "seed": args.seed, "setups": args.setups, "decoder": args.decoder,
        "tuning": str(args.tuning) if args.tuning else None, "rounds_this_run": rounds_done,
        "answered": {c: len(written_ids(p)) for c, p in paths.items()},
        "questions": {c: len(r) for c, r in rows.items()},
        "gpu": gpu.summary(), "pacer": bench.throttle.stats(),
    }
    target = out / f"summary-{args.level}.json"
    write_json(target, summary)
    print(f"written {target}", flush=True)
    return target


if __name__ == "__main__":
    main()
