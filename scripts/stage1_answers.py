"""Stage 1 of the corpus selection, and stage 2 with --level: every corpus answered in its setup, in rounds.

The run goes through all corpora at once in rounds (foqlens.schedule), each round a share of every
corpus, so a run stopped anywhere leaves every corpus equally far. Every answer is appended to
<out>/<model>/answers/<level>/<corpus>.jsonl as soon as its batch is written; a restart reads those
files and asks only what is missing. Within a round the questions go in batches by prompt length.
The questions spent on choosing the prompt (--tuning, the summary of scripts/prompt_tuning.py) are left out.
Stage 2 reads only the frozen corpus (--frozen): the kept questions and the unknown share of each file.

    uv run python scripts/stage1_answers.py --tuning runs/reference/prompt-tuning/e2b-it/summary.json
    uv run python scripts/stage1_answers.py --level d4 --frozen corpus/e2b-it --out runs/E002-base-precision-d2/ladder
    uv run python scripts/stage1_answers.py --rounds 1 --out /tmp/stage1   # smoke check: one round
    uv run python scripts/stage1_answers.py --level d2 --gguf <file.gguf> --frozen corpus/e2b-it --parts kept --out <dir>
    uv run python scripts/stage1_answers.py --level d4 --base bartowski-Q2_K --frozen corpus/e2b-it --out <dir>
    uv run python scripts/stage1_answers.py --level d2 --frozen corpus/e2b-it --parts kept \\
        --corpora arc_challenge_closed arc_easy_closed hotpotqa --written-tokens 1024 \\
        --cut-of runs/E002-base-precision-d2/ladder/e2b-it/unjudged/d2 --out runs/E002-base-precision-d2/reask-1024

A cut reply asked again at a raised cap is an observation, not a level's measure: the other levels answered at the
frozen cap.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from foqlens import corpora, refocustensors
from foqlens import model as fm
from foqlens.answering import BATCH_TOKENS, Asking, cut_ids, level_label
from foqlens.attention import PLANS, SPLIT
from foqlens.generation import DYNAMIC
from foqlens.gpu_monitor import GpuMonitor
from foqlens.graph_decode import PREFILL_TOKENS, StaticDecoder
from foqlens.gpu_share import default_share
from foqlens.io import answers_path, append_answers, read_answers, read_frozen, write_json, written_ids
from foqlens.gguf_weights import PUBLISHED, GgufWeights
from foqlens.judging import ModelJudge, NotJudged
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS, WRITTEN_TOKENS, examples_for, needs_train, setup_named
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
# justify-terse leaves out its answer line. ARC-Easy solve-brief (2026-09-15): the four setups judged
# right 0.84-0.90 on 50 questions, a tie; of the two without examples solve-0 ran out of tokens before
# its answer line on 12%, solve-brief never did.
FROZEN_SETUPS = {"triviaqa": "short-0", "nq_open": "short-0", "squad_v2": "passage-0",
                 "arc_challenge_closed": "solve-0", "arc_easy_closed": "solve-brief", "hotpotqa": "justify"}
# static: the static cache with every step a CUDA graph (foqlens.graph_decode), on the kernels of the attention
# plan; dynamic: the reference loop, which stays on the process's default kernel, math, whatever the plan.
DECODERS = {"static": lambda plan, prefill: StaticDecoder(attention=plan, prefill_tokens=prefill),
            "dynamic": lambda plan, prefill: DYNAMIC}
UNJUDGED = "unjudged"  # where a baked level's answers wait for the bf16 judge
BAKED, BLOCKS = "baked", "blocks"  # a quantized level baked into the weights, or read by blocks through the kernel
READS = (BAKED, BLOCKS)
FROZEN_PARTS = ("kept", "unknown_share")  # what stage 2 asks of a frozen file (selection.FrozenCorpus.asked)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--level", choices=list(LEVELS), default="bf16")
    parser.add_argument("--gguf", type=Path, default=None,
                        help="a published .gguf file to read a quantized level from instead of the bench's copy, for comparison")
    parser.add_argument("--base", choices=sorted(PUBLISHED), default=None,
                        help="read the model cut over this published file's base (cut_model.py --base); a quantized "
                        "level is then baked from that copy")
    parser.add_argument("--read", choices=READS, default=BAKED,
                        help="how a quantized level is read: baked into the weights once (the step of bf16), or by "
                        "blocks through the kernel, as a zone layout is - to time the kernel on the corpus")
    parser.add_argument("--setups", type=json.loads, default=FROZEN_SETUPS, help="JSON {corpus: setup name}")
    asked = parser.add_mutually_exclusive_group()
    asked.add_argument("--tuning", type=Path, default=None, help="summary.json of prompt_tuning.py: its questions are left out")
    asked.add_argument("--frozen", type=Path, default=None, help="folder of frozen corpus files: only what they ask")
    parser.add_argument("--parts", nargs="+", choices=FROZEN_PARTS, default=list(FROZEN_PARTS),
                        help="with --frozen: which parts of the frozen files are asked")
    parser.add_argument("--cut-of", type=Path, default=None,
                        help="a folder of answers files: only the questions whose answer there ran into the token cap are asked")
    parser.add_argument("--written-tokens", type=int, default=WRITTEN_TOKENS,
                        help="the cap of a written reply; raised with --cut-of to see whether a cut reply reaches its answer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=None, help="stop after this many rounds, for a smoke check")
    parser.add_argument("--out", type=Path, default=Path("runs/reference/stage1"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--decoder", choices=list(DECODERS), default="static")
    parser.add_argument("--attention", choices=list(PLANS), default=SPLIT.name,
                        help="sdpa kernels per phase (foqlens.attention): the prefill always on math")
    parser.add_argument("--prefill-tokens", type=int, default=PREFILL_TOKENS,
                        help="prompt tokens one prefill pass of the static decoder holds; more are read in chunks of rows")
    parser.add_argument("--batch-tokens", type=int, default=BATCH_TOKENS,
                        help="padded prompt tokens a batch of answers holds: more rows decode in one batch, and a batch "
                        "costs its longest reply in steps")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    model_id, level = MODELS[args.model], LEVELS[args.level]
    plan = PLANS[args.attention]
    directory = refocustensors.model_directory(model_id, args.base) if args.base else None
    bench = Bench.load(model_id, gpu_share=args.gpu_share, directory=directory)
    tokenizer = bench.tokenizer
    fmt = fm.prompt_format(model_id, tokenizer)
    # A quantized level is baked into the weights - one unpacking, the step and the memory of bf16 - and
    # so cannot judge: its answers wait in unjudged/ for the bf16 judge of scripts/rejudge_answers.py.
    if level is Level.BF16:
        judge = ModelJudge(bench.model, tokenizer, bench.ctl, fmt, StaticDecoder(attention=plan, prefill_tokens=args.prefill_tokens))
        root = "answers"
    else:
        if args.gguf is not None:
            bench.ctl.bake_from(GgufWeights(args.gguf), level)
        elif args.read == BAKED:
            bench.ctl.bake(level)
        else:
            bench.ctl.set_all(level)  # read by blocks, as a zone layout is: the k-quant kernel on a decoding step
        judge, root = NotJudged(), UNJUDGED
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    out = args.out / args.model
    spent = json.loads(args.tuning.read_text(encoding="utf-8"))["corpora"] if args.tuning else {}

    rows, askings, paths = {}, {}, {}
    for corpus in args.corpora:
        corpus_rows, source = corpora.read(corpus)
        if args.frozen:
            frozen = read_frozen(args.frozen / f"{corpus}.json")
            frozen.check(name, source.revision, args.setups[corpus])
            asked = {i for part in args.parts for i in getattr(frozen, part)}
            rows[corpus] = {r.id: r for r in corpus_rows if r.id in asked}
            if len(rows[corpus]) != len(asked):
                raise ValueError(f"{corpus}: {len(asked) - len(rows[corpus])} frozen questions are not in the corpus")
        else:
            left_out = set(spent.get(corpus, {}).get("tuning_ids", []))
            rows[corpus] = {r.id: r for r in corpus_rows if r.id not in left_out}
        if args.cut_of:
            cut = cut_ids(read_answers(args.cut_of / f"{corpus}.jsonl"))
            rows[corpus] = {i: r for i, r in rows[corpus].items() if i in cut}
        setup = replace(setup_named(corpus, args.setups[corpus]), written_tokens=args.written_tokens)
        train = corpora.read_train(corpus, TRAIN_POOL) if needs_train(setup) else []
        askings[corpus] = Asking(corpus, source.revision, name, level, setup, examples_for(corpus, setup, train, args.seed))
        paths[corpus] = answers_path(out / root, args.level, corpus)

    schedule = Schedule.build({c: list(r) for c, r in rows.items()}, args.seed)
    written = {c: written_ids(p) for c, p in paths.items()}
    rounds_done = 0
    decoder = DECODERS[args.decoder](plan, args.prefill_tokens)
    with GpuMonitor() as gpu:
        progress = Progress(schedule.rounds, "round")
        for k, todo in schedule.pending(written):
            if args.rounds is not None and rounds_done == args.rounds:
                break
            for corpus, ids in todo.items():
                asking = askings[corpus]
                for chunk in asking.batches(fmt, tokenizer, [rows[corpus][i] for i in ids], args.batch_tokens):
                    append_answers(paths[corpus], asking.answer(bench.model, tokenizer, bench.ctl, fmt, judge,
                                                                chunk, bench.throttle, decoder))
            rounds_done += 1
            print(progress.step(f"round {k}"), flush=True)

    target = out / f"summary-{args.level}.json"
    # A corpus answered later joins the corpora answered before: their counts stay in the summary.
    earlier = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    summary = {
        "model": name, "level": args.level, "weights": "gguf" if args.gguf else "kquant", "gguf": str(args.gguf) if args.gguf else None, "base": args.base, "read": args.read, "seed": args.seed, "decoder": args.decoder, "attention": args.attention, "prefill_tokens": args.prefill_tokens, "batch_tokens": args.batch_tokens,
        "setups": {**earlier.get("setups", {}), **{c: args.setups[c] for c in args.corpora}},
        "tuning": str(args.tuning) if args.tuning else None, "frozen": str(args.frozen) if args.frozen else None,
        "cut_of": str(args.cut_of) if args.cut_of else None, "written_tokens": args.written_tokens,
        "rounds_this_run": rounds_done, "written_to": root,
        "answered": {**earlier.get("answered", {}), **{c: len(written_ids(p)) for c, p in paths.items()}},
        "questions": {**earlier.get("questions", {}), **{c: len(r) for c, r in rows.items()}},
        "gpu": gpu.summary(), "pacer": bench.throttle.stats(),
    }
    write_json(target, summary)
    print(f"written {target}", flush=True)
    return target


if __name__ == "__main__":
    main()
