"""Answers files judged by the present judge at bf16, loaded once: every file judged is a run of its own.

Two uses in stage 2. The bf16 arm: stage 1's bf16 answers, judged again because the judge changed after
stage 1 (41c12ab) - the kept questions are known at bf16 by construction and the static decoder is nearly
invariant to the composition of a batch, so no reply is generated again. The quantized levels: a level
is baked into the weights and cannot judge itself (scripts/stage1_answers.py), so its answers wait in
unjudged/ and are judged here, after the level has answered: the two never share the card.

A run is one pass of the judge over one answers file, written to <out>/<model>/judge/<level>/<corpus>/<run>.jsonl
with its summary beside it; nothing an earlier run wrote is touched, and the verdict on a question is the
latest run's that read it (foqlens.runs, view `verdicts`). --rows picks lines of the one file given, counted
from 1 - to ask again the answers an earlier run left N/A; without it the file's frozen questions are judged.
A restart under the same --run skips what that run has written.

With --watch the model stays loaded and waits: a request is a JSON file dropped into that folder,
{"level": "bf16", "corpus": "arc_challenge_closed", "rows": "12,305-310"} (rows optional), judged as its own
run and moved to done/ with the run's summary path, or to failed/ with the error.

    uv run python scripts/rejudge_answers.py --answers runs/E016-uniform-quantization/e2b-it/unjudged \\
        --levels d8 d6 d4 d2 --frozen corpus/e2b-it --out runs/E016-uniform-quantization
    uv run python scripts/rejudge_answers.py --answers runs/E016-uniform-quantization/e2b-it/answers --levels bf16 \\
        --corpora arc_challenge_closed --rows 12,305-310 --frozen corpus/e2b-it --out runs/E016-uniform-quantization
    uv run python scripts/rejudge_answers.py --answers runs/E016-uniform-quantization/e2b-it/answers \\
        --watch .claude/judge-requests --frozen corpus/e2b-it --out runs/E016-uniform-quantization
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from foqlens import corpora
from foqlens import model as fm
from foqlens.answering import rejudge
from foqlens.attention import PLANS, SPLIT
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.graph_decode import StaticDecoder
from foqlens.io import (answers_path, append_answers, judge_run_path, judge_run_summary_path, parse_rows, read_answers,
                        read_frozen, write_json, written_ids)
from foqlens.judging import ModelJudge
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS

MODELS = {"e2b-it": fm.E2B_IT, "e4b-it": fm.E4B_IT}
LEVELS = ("bf16", "d8", "d6", "d4", "d2")
CHUNK = 1024  # answers judged between two writes: a stopped run loses at most these
RUN_NAME = "%Y-%m-%dT%H-%M-%S"  # a run is named by when it started; names sort as the runs went
WATCH_POLL_SECONDS = 5  # a request waits at most this long; the loaded judge costs nothing while it looks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--answers", type=Path, required=True, help="folder of <level>/<corpus>.jsonl to judge")
    asked = parser.add_mutually_exclusive_group(required=True)
    asked.add_argument("--levels", nargs="+", choices=LEVELS, help="judged in this order")
    asked.add_argument("--watch", type=Path, help="wait with the model loaded for request files in this folder")
    parser.add_argument("--rows", type=parse_rows, default=None,
                        help="lines of the one file given (one level, one corpus), from 1: 1-500,812")
    parser.add_argument("--run", default=None, help="the run's name, by default its start time; the same name resumes it")
    parser.add_argument("--frozen", type=Path, required=True, help="folder of frozen corpus files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--attention", choices=list(PLANS), default=SPLIT.name,
                        help="sdpa kernels per phase (foqlens.attention): the judge's prefill and its decoding steps")
    args = parser.parse_args(argv)
    if args.rows is not None and (not args.levels or len(args.levels) != 1 or len(args.corpora) != 1):
        parser.error("--rows picks lines of one file: give one level and one corpus")
    return args


@dataclass
class Judging:
    """The loaded judge and what every run of this process shares: the model's name, the code's commit, the corpora read."""

    args: argparse.Namespace
    judge: ModelJudge
    bench: Bench
    name: str
    commit: str
    corpora: dict

    def questions(self, corpus: str) -> tuple[dict, list[str]]:
        """A corpus's rows by id and its frozen questions, read once."""
        if corpus not in self.corpora:
            frozen = read_frozen(self.args.frozen / f"{corpus}.json")
            rows, source = corpora.read(corpus)
            frozen.check(self.name, source.revision, frozen.prompt)
            self.corpora[corpus] = ({r.id: r for r in rows}, frozen.asked())
        return self.corpora[corpus]

    def run(self, level: str, corpus: str, rows: list[int] | None, run: str | None) -> Path:
        """One run: one answers file judged, its verdicts and summary written beside its earlier runs."""
        run = run or datetime.now().strftime(RUN_NAME)
        by_id, asked = self.questions(corpus)
        source = answers_path(self.args.answers, level, corpus)
        lines = read_answers(source)
        if rows is None:
            by_answer = {a.id: a for a in lines}
            missing = [i for i in asked if i not in by_answer]
            if missing:
                raise ValueError(f"{corpus}: {len(missing)} frozen questions have no answer at {level}, e.g. {missing[:3]}")
            picked = [by_answer[i] for i in asked]
        else:
            if max(rows) > len(lines):
                raise ValueError(f"{source} has {len(lines)} lines, row {max(rows)} asked")
            picked = [lines[row - 1] for row in rows]
        root = self.args.out / self.args.model / "judge"
        path = judge_run_path(root, level, corpus, run)
        done = written_ids(path)
        todo = [a for a in picked if a.id not in done]

        with GpuMonitor() as gpu:
            progress = Progress(-(-len(todo) // CHUNK), "chunk")
            for start in range(0, len(todo), CHUNK):
                chunk = todo[start:start + CHUNK]
                append_answers(path, rejudge(self.judge, chunk, [by_id[a.id] for a in chunk], self.bench.throttle))
                print(progress.step(f"{run} {level} {corpus} {start + len(chunk)}/{len(todo)}"), flush=True)

        target = judge_run_summary_path(root, level, corpus, run)
        write_json(target, {"model": self.name, "run": run, "commit": self.commit, "level": level, "corpus": corpus,
                            "answers": str(source), "rows": rows, "ids": [a.id for a in picked],
                            "attention": self.args.attention, "frozen": str(self.args.frozen),
                            "judged": len(written_ids(path)), "gpu": gpu.summary(), "pacer": self.bench.throttle.stats()})
        print(f"written {target}", flush=True)
        return target


def watch(judging: Judging, folder: Path) -> None:
    """Requests in `folder`, oldest first, each a run; done/ and failed/ keep what became of them."""
    for sub in ("done", "failed"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    print(f"waiting for requests in {folder}", flush=True)
    while True:
        requests = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not requests:
            time.sleep(WATCH_POLL_SECONDS)
            continue
        request = requests[0]
        asked = json.loads(request.read_text(encoding="utf-8"))
        try:
            rows = parse_rows(asked["rows"]) if asked.get("rows") else None
            summary = judging.run(asked["level"], asked["corpus"], rows, asked.get("run"))
            write_json(folder / "done" / request.name, {**asked, "summary": str(summary)})
        except Exception:  # a bad request must not stop the judge waiting for the next
            write_json(folder / "failed" / request.name, {**asked, "error": traceback.format_exc()})
        request.unlink()


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    model_id = MODELS[args.model]
    bench = Bench.load(model_id, gpu_share=args.gpu_share)
    judge = ModelJudge(bench.model, bench.tokenizer, bench.ctl, fm.prompt_format(model_id, bench.tokenizer),
                       StaticDecoder(attention=PLANS[args.attention]))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    judging = Judging(args, judge, bench, f"{model_id}@{fm.REVISIONS[model_id][:8]}", commit, {})
    if args.watch:
        watch(judging, args.watch)
        return []
    return [judging.run(level, corpus, args.rows, args.run) for corpus in args.corpora for level in args.levels]


if __name__ == "__main__":
    main()
