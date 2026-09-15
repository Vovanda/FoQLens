"""The bf16 arm of stage 2: the stage 1 answers at bf16 on the frozen corpus, read again by the present judge.

No reply is generated again: the kept questions are known at bf16 by construction, and the static
decoder is nearly invariant to the composition of a batch. The judge changed after stage 1 (41c12ab),
so its verdicts and the kinds of answer are taken anew and written to
<out>/<model>/answers/bf16/<corpus>.jsonl beside the levels of stage 2; a restart skips what is written.

    uv run python scripts/rejudge_answers.py --frozen corpus/e2b-it --out runs/E016-uniform-quantization
"""

from __future__ import annotations

import argparse
from pathlib import Path

from foqlens import corpora
from foqlens import model as fm
from foqlens.answering import JUDGE_BATCH, rejudge
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.io import answers_path, append_answers, read_answers, read_frozen, write_json, written_ids
from foqlens.judging import ModelJudge
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS

MODELS = {"e2b-it": fm.E2B_IT, "e4b-it": fm.E4B_IT}
LEVEL = "bf16"
CHUNK = 1024  # answers judged between two writes: a stopped run loses at most these


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--frozen", type=Path, required=True, help="folder of frozen corpus files")
    parser.add_argument("--stage1", type=Path, default=Path("runs/reference/stage1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu-share", type=float, default=default_share())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    model_id = MODELS[args.model]
    bench = Bench.load(model_id, gpu_share=args.gpu_share)
    judge = ModelJudge.build(bench.model, bench.tokenizer, bench.ctl, fm.prompt_format(model_id, bench.tokenizer),
                             batch_size=JUDGE_BATCH)
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    out = args.out / args.model

    todo = {}
    for corpus in args.corpora:
        frozen = read_frozen(args.frozen / f"{corpus}.json")
        corpus_rows, source = corpora.read(corpus)
        frozen.check(name, source.revision, frozen.prompt)
        rows = {r.id: r for r in corpus_rows}
        earlier = {a.id: a for a in read_answers(answers_path(args.stage1 / args.model / "answers", LEVEL, corpus))}
        path = answers_path(out / "answers", LEVEL, corpus)
        done = written_ids(path)
        ids = [i for i in frozen.asked() if i not in done]
        missing = [i for i in ids if i not in earlier]
        if missing:
            raise ValueError(f"{corpus}: {len(missing)} frozen questions have no stage 1 answer, e.g. {missing[:3]}")
        todo[corpus] = (path, [earlier[i] for i in ids], [rows[i] for i in ids])

    with GpuMonitor() as gpu:
        progress = Progress(sum(-(-len(a) // CHUNK) for _, a, _ in todo.values()), "chunk")
        for corpus, (path, answers, rows) in todo.items():
            for start in range(0, len(answers), CHUNK):
                chunk = slice(start, start + CHUNK)
                append_answers(path, rejudge(judge, answers[chunk], rows[chunk], bench.throttle))
                print(progress.step(f"{corpus} {start + len(answers[chunk])}/{len(answers)}"), flush=True)

    target = out / f"summary-rejudge-{LEVEL}.json"
    write_json(target, {"model": name, "level": LEVEL, "frozen": str(args.frozen), "stage1": str(args.stage1),
                        "answered": {c: len(written_ids(p)) for c, (p, _, _) in todo.items()},
                        "gpu": gpu.summary(), "pacer": bench.throttle.stats()})
    print(f"written {target}", flush=True)
    return target


if __name__ == "__main__":
    main()
