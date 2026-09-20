"""Choosing each corpus's prompt on its tuning share: every setup of foqlens.prompt_variants, answered and judged.

The tuning share is a seeded random draw of each corpus (selection.tuning_sample); those questions are
spent on the choice and marked in the frozen corpus. Every answer goes to
<out>/<model>/answers/bf16/<corpus>.jsonl as a selection.Answer line; a restart skips what is written.
The summary holds, per corpus and setup, exact match, F1 and the model judge, refusals and length, and how much
the verdict hangs on the setup. Claude reads the answers before a setup is frozen.

    uv run python scripts/prompt_tuning.py
    uv run python scripts/prompt_tuning.py --corpora triviaqa --limit 16 --out /tmp/tuning   # smoke check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foqlens import corpora
from foqlens import model as fm
from foqlens.answering import Asking, level_label
from foqlens.attention import SPLIT
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.graph_decode import StaticDecoder
from foqlens.io import answers_path, append_answers, read_answers, write_json
from foqlens.judging import ModelJudge
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS, TRAIN_POOL, agreement, examples_for, needs_train, setup_summary
from foqlens.quant import Level
from foqlens.selection import tuning_sample

MODELS = {"e2b-it": fm.E2B_IT, "e4b-it": fm.E4B_IT}
LEVEL = Level.BF16  # the setup is chosen on the full model, one for every model afterwards


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="questions per corpus, for a smoke check")
    parser.add_argument("--out", type=Path, default=Path("runs/reference/prompt-tuning"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    model_id = MODELS[args.model]
    bench = Bench.load(model_id, gpu_share=args.gpu_share)
    fmt = fm.prompt_format(model_id, bench.tokenizer)
    judge = ModelJudge(bench.model, bench.tokenizer, bench.ctl, fmt, StaticDecoder(attention=SPLIT))
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    out = args.out / args.model
    # A corpus tuned later joins the corpora tuned before: their tuning ids are what the frozen corpus left out.
    earlier = json.loads((out / "summary.json").read_text(encoding="utf-8"))["corpora"] if (out / "summary.json").exists() else {}
    summary = {"model": name, "seed": args.seed, "level": level_label(LEVEL), "corpora": dict(earlier)}

    with GpuMonitor() as gpu:
        progress = Progress(sum(len(SETUPS[c]) for c in args.corpora), "setup")
        for corpus in args.corpora:
            rows, source = corpora.read(corpus)
            chosen = set(tuning_sample([r.id for r in rows], args.seed))
            rows = [r for r in rows if r.id in chosen][: args.limit]
            path = answers_path(out / "answers", level_label(LEVEL), corpus)
            for setup in SETUPS[corpus]:
                train = corpora.read_train(corpus, TRAIN_POOL) if needs_train(setup) else []
                asking = Asking(corpus, source.revision, name, LEVEL, setup, examples_for(corpus, setup, train, args.seed))
                done = {a.id for a in read_answers(path) if a.prompt == setup.name}
                todo = [r for r in rows if r.id not in done]
                for chunk in asking.batches(fmt, bench.tokenizer, todo):
                    append_answers(path, asking.answer(bench.model, bench.tokenizer, bench.ctl, fmt, judge,
                                                       chunk, bench.throttle))
                print(progress.step(f"{corpus}/{setup.name}"), flush=True)
            answered = [a for a in read_answers(path) if a.id in chosen]
            summary["corpora"][corpus] = {
                "source": str(source), "tuning_ids": sorted(chosen),
                "setups": {s.name: setup_summary([a for a in answered if a.prompt == s.name]) for s in SETUPS[corpus]},
                "agreement": agreement(answered),
            }

    summary["gpu"] = gpu.summary()
    summary["pacer"] = bench.throttle.stats()
    write_json(out / "summary.json", summary)
    print(f"written {out / 'summary.json'}", flush=True)
    return out / "summary.json"


if __name__ == "__main__":
    main()
