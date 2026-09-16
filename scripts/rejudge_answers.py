"""Answers of one level on the frozen corpus, judged by the present judge at bf16.

Two uses in stage 2. The bf16 arm: stage 1's bf16 answers, judged again because the judge changed after
stage 1 (41c12ab) - the kept questions are known at bf16 by construction and the static decoder is nearly
invariant to the composition of a batch, so no reply is generated again. The quantized levels: a level
is baked into the weights and cannot judge itself (scripts/stage1_answers.py), so its answers wait in
unjudged/ and are judged here. Either way the verdict - the kind of answer and whether it is accepted - is taken anew and written
to <out>/<model>/answers/<level>/<corpus>.jsonl; a restart skips what is written.

    uv run python scripts/rejudge_answers.py --answers runs/reference/stage1/e2b-it/answers --level bf16 \\
        --frozen corpus/e2b-it --out runs/E016-uniform-quantization
    uv run python scripts/rejudge_answers.py --answers runs/E016-uniform-quantization/e2b-it/unjudged --level d4 \\
        --frozen corpus/e2b-it --out runs/E016-uniform-quantization
"""

from __future__ import annotations

import argparse
from pathlib import Path

from foqlens import corpora
from foqlens import model as fm
from foqlens.answering import rejudge
from foqlens.attention import PLANS, SPLIT
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.graph_decode import StaticDecoder
from foqlens.io import answers_path, append_answers, read_answers, read_frozen, write_json, written_ids
from foqlens.judging import ModelJudge
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS

MODELS = {"e2b-it": fm.E2B_IT, "e4b-it": fm.E4B_IT}
LEVELS = ("bf16", "d8", "d6", "d4", "d2")
CHUNK = 1024  # answers judged between two writes: a stopped run loses at most these


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--answers", type=Path, required=True, help="folder of <level>/<corpus>.jsonl to judge")
    parser.add_argument("--level", choices=LEVELS, required=True)
    parser.add_argument("--frozen", type=Path, required=True, help="folder of frozen corpus files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--attention", choices=list(PLANS), default=SPLIT.name,
                        help="sdpa kernels per phase (foqlens.attention): the judge's prefill and its decoding steps")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    model_id = MODELS[args.model]
    bench = Bench.load(model_id, gpu_share=args.gpu_share)
    judge = ModelJudge(bench.model, bench.tokenizer, bench.ctl, fm.prompt_format(model_id, bench.tokenizer),
                       StaticDecoder(attention=PLANS[args.attention]))
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"
    out = args.out / args.model

    todo = {}
    for corpus in args.corpora:
        frozen = read_frozen(args.frozen / f"{corpus}.json")
        corpus_rows, source = corpora.read(corpus)
        frozen.check(name, source.revision, frozen.prompt)
        rows = {r.id: r for r in corpus_rows}
        earlier = {a.id: a for a in read_answers(answers_path(args.answers, args.level, corpus))}
        path = answers_path(out / "answers", args.level, corpus)
        done = written_ids(path)
        ids = [i for i in frozen.asked() if i not in done]
        missing = [i for i in ids if i not in earlier]
        if missing:
            raise ValueError(f"{corpus}: {len(missing)} frozen questions have no answer at {args.level}, e.g. {missing[:3]}")
        todo[corpus] = (path, [earlier[i] for i in ids], [rows[i] for i in ids])

    with GpuMonitor() as gpu:
        progress = Progress(sum(-(-len(a) // CHUNK) for _, a, _ in todo.values()), "chunk")
        for corpus, (path, answers, rows) in todo.items():
            for start in range(0, len(answers), CHUNK):
                chunk = slice(start, start + CHUNK)
                append_answers(path, rejudge(judge, answers[chunk], rows[chunk], bench.throttle))
                print(progress.step(f"{corpus} {start + len(answers[chunk])}/{len(answers)}"), flush=True)

    target = out / f"summary-judged-{args.level}.json"
    write_json(target, {"model": name, "level": args.level, "attention": args.attention,
                        "frozen": str(args.frozen), "answers": str(args.answers),
                        "judged": {c: len(written_ids(p)) for c, (p, _, _) in todo.items()},
                        "gpu": gpu.summary(), "pacer": bench.throttle.stats()})
    print(f"written {target}", flush=True)
    return target


if __name__ == "__main__":
    main()
