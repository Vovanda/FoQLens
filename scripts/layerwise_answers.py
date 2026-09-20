"""The regulator's answers over the small corpus (config.RegulatorAnswers): the approach is chosen in the file -
`layerwise`, where every layer is decided while the pass runs (foqlens.layerwise, variant B), or `upfront`, where the
map is decided before it (scripts/oracle_answers.py, variant A). A run per price of memory, and the uniform rungs
beside them as the ladder. What is compared afterwards is the reply against the whole network's and the bytes each
layout read (scripts/precision_answers.py).

    uv run python scripts/layerwise_answers.py --config configs/layerwise-answers.toml
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
from foqlens.answering import UniformReading
from foqlens.attention import PLANS, SPLIT
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.generation import DYNAMIC
from foqlens.graph_decode import PREFILL_TOKENS, StaticDecoder
from foqlens.io import answers_path, append_answers, write_json, written_ids
from foqlens.judging import NotJudged
from foqlens.layerwise import SOURCES, LayerwiseRegulator
from foqlens.pipeline import Bench
from foqlens.progress import Progress
from foqlens.prompt_variants import SETUPS
from foqlens.quant import Level
from foqlens.small_corpus import draw, pick_shard

MODEL = fm.E2B_IT
LOG = logging.getLogger("foqlens.layerwise_answers")
LAYERWISE, UPFRONT = "layerwise", "upfront"


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--corpora", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--shard", type=int, default=None, help="a part of the draw; without it the whole 5%%")
    parser.add_argument("--limit", type=int, default=None, help="the first N questions only: a check")
    parser.add_argument("--out", type=Path, default=Path("runs/regulator/e2b-it"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    parser.add_argument("--attention", choices=list(PLANS), default=SPLIT.name)
    args = parser.parse_args(argv)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    runlog.setup(args.out / "logs" / f"layerwise_answers-{run}.jsonl", {"run": run, "script": "layerwise_answers"})
    check = config.read(args.config, config.RegulatorAnswers)
    if check.approach != LAYERWISE:
        raise ValueError(f"this script runs the {LAYERWISE} approach; {check.approach} is scripts/oracle_answers.py")
    floor, ceiling = Level[check.floor.upper()], Level[check.ceiling.upper()]
    source = config.choose(check.source, SOURCES, "a score of a layer")

    bench = Bench.load(MODEL, gpu_share=args.gpu_share, directory=refocustensors.model_directory(MODEL, check.base))
    tokenizer, ctl = bench.tokenizer, bench.ctl
    fmt = fm.prompt_format(MODEL, tokenizer)
    name = f"{MODEL}@{fm.REVISIONS[MODEL][:8]}"
    small = config.read(Path(check.corpus), config.SmallCorpus)
    found, suffix = pick_shard(draw(args.corpora, args.frozen, small, name), small, args.shard)
    laid = found.laid[:args.limit] if args.limit else found.laid
    decoder = StaticDecoder(attention=PLANS[args.attention], prefill_tokens=PREFILL_TOKENS)
    judge = NotJudged()  # the question is whether the reply is the whole network's, read by hand where it is not

    targets = []
    runs = [(f"uniform-{rung}-{check.base}", UniformReading(Level[rung.upper()]), None) for rung in check.uniform]
    for price in check.prices:
        label = f"layerwise-{check.source}-{check.base}-{floor.name.lower()}{ceiling.name.lower()}-p{price:g}"
        regulator = LayerwiseRegulator(ctl, SOURCES[source](), price=price, base=floor, ceiling=ceiling, rim=check.rim)
        runs.append((label, regulator.reading(label), regulator))
    progress = Progress(len(runs), "layout")
    for label, reading, regulator in runs:
        spent = []
        with GpuMonitor() as gpu:
            if regulator is not None:
                regulator.attach(bench.model)
            try:
                for corpus, asking in found.askings.items():
                    asking = replace(asking, reading=reading)
                    path = answers_path(args.out / "answers", label, corpus)
                    done = written_ids(path)  # a stopped run answers only what its file does not hold yet
                    chunks = asking.batches(fmt, tokenizer, [r for c, r in laid if c == corpus and r.id not in done])
                    answered = Progress(len(chunks), f"{label} {corpus} answer batch")
                    for chunk in chunks:
                        if regulator is not None:
                            regulator.start_counting(len(chunk))
                        # the layer-wise regulator decides inside the pass, so its pass is not captured in a graph:
                        # a captured graph would freeze the layout of the pass it was captured on
                        judged = asking.answer(bench.model, tokenizer, ctl, fmt, judge, chunk, bench.throttle,
                                               DYNAMIC if regulator is not None else decoder)
                        if regulator is not None:
                            spent += regulator.bits_a_weight().tolist()
                        append_answers(path, judged)
                        LOG.info(answered.step(f"{len(chunk)} questions"), extra={"layout": label, "corpus": corpus})
            finally:
                if regulator is not None:
                    regulator.detach()
        target = args.out / f"summary-{label}{suffix}.json"
        write_json(target, {
            "model": name, "base": check.base, "approach": check.approach, "source": check.source,
            "floor": floor.name, "ceiling": ceiling.name, "rim": check.rim, "questions": len(laid),
            "bits_a_weight": {"mean": float(np.mean(spent)) if spent else None,
                              "median": float(np.median(spent)) if spent else None,
                              "p90": float(np.quantile(spent, 0.9)) if spent else None},
            "gpu": gpu.summary(), "pacer": bench.throttle.stats(),
        })
        LOG.info(progress.step(f"{label} written {target}"), extra={"layout": label, "file": str(target)})
        targets.append(target)
    return targets


if __name__ == "__main__":
    main()
