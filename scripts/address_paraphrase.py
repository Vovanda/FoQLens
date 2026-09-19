"""Does a mask source find a question's meaning or its words (foqlens.address): the question against its paraphrase.

Everything the run does is its configuration (config.ParaphraseCheck, configs/address-paraphrase.toml). Per source:
identification between the questions and their paraphrases, both in the corpus's frozen wrapper at bf16, beside the
same identification of their bags of tokens - the lexical control; and the agreement of every two sources, which
score different blocks, on how they place the questions.

    uv run python scripts/address_paraphrase.py --config configs/address-paraphrase.toml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from foqlens import config, corpora, refocustensors
from foqlens import model as fm
from foqlens.address import agreement, bag_of_tokens, identification
from foqlens.answering import Asking
from foqlens.gpu_monitor import GpuMonitor
from foqlens.gpu_share import default_share
from foqlens.io import read_frozen, write_json
from foqlens.pipeline import ADDRESS_SOURCES, Bench
from foqlens.prompt_variants import setup_named
from foqlens.quant import Level

MODELS = {"e2b-it": fm.E2B_IT, "e4b-it": fm.E4B_IT}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b-it")
    parser.add_argument("--frozen", type=Path, default=Path("corpus/e2b-it"))
    parser.add_argument("--out", type=Path, default=Path("runs/address"))
    parser.add_argument("--gpu-share", type=float, default=default_share())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    check = config.read(args.config, config.ParaphraseCheck)
    sources = [config.choose(s, ADDRESS_SOURCES, "mask source") for s in check.sources]
    model_id = MODELS[args.model]
    directory = refocustensors.model_directory(model_id, check.base) if check.base else None
    bench = Bench.load(model_id, gpu_share=args.gpu_share, directory=directory)
    fmt = fm.prompt_format(model_id, bench.tokenizer)
    name = f"{model_id}@{fm.REVISIONS[model_id][:8]}"

    rows, source = corpora.read(check.corpus)
    frozen = read_frozen(args.frozen / f"{check.corpus}.json")
    frozen.check(name, source.revision, frozen.prompt)
    by_id = {r.id: r for r in rows}
    pairs = [json.loads(line) for line in Path(check.paraphrases).read_text(encoding="utf-8").splitlines() if line]
    originals = [by_id[p["id"]] for p in pairs]
    paraphrased = [replace(by_id[p["id"]], question=p["paraphrase"]) for p in pairs]
    asking = Asking(check.corpus, source.revision, name, Level.BF16, setup_named(check.corpus, frozen.prompt), ())
    prompts = [[asking.prompts(fmt, [r])[0] for r in group] for group in (originals, paraphrased)]

    words = [bench.tokenizer([r.question for r in group], add_special_tokens=False)["input_ids"]
             for group in (originals, paraphrased)]
    bag = bag_of_tokens(words[0] + words[1])
    summary = {"model": name, "config": str(args.config), "questions": len(pairs),
               "bag_of_tokens": identification(bag[: len(pairs)], bag[len(pairs):]), "sources": {}, "agreement": {}}
    frozen_masks = {}
    with GpuMonitor() as gpu:
        for source_name in sources:
            made = bench.source(source_name)
            original, paraphrase = (bench.masks(p, [made])[made.name] for p in prompts)
            frozen_masks[source_name] = original
            summary["sources"][source_name] = identification(original, paraphrase)
            print(f"{source_name}: paraphrase {summary['sources'][source_name]['identified']:.3f} "
                  f"(bag of tokens {summary['bag_of_tokens']['identified']:.3f}, chance {1 / len(pairs):.3f})", flush=True)
    for a, b in combinations(sources, 2):
        summary["agreement"][f"{a}~{b}"] = agreement(frozen_masks[a], frozen_masks[b])
    summary["gpu"] = gpu.summary()
    target = args.out / args.model / f"{args.config.stem}.json"
    write_json(target, summary)
    print(f"agreement {summary['agreement']}\nwritten {target}", flush=True)
    return target


if __name__ == "__main__":
    main()
