"""Read depths of the sliced copy (docs/plan.md, step 5): quality at every level, and the memory once bf16 is dropped.

Mean perplexity over the same question texts at bf16, int8, nf4 and the read depths D2 ... D8;
then the bf16 weights are dropped and D8 is read again from the resident sliced copy. GPU memory
is taken right after install (bf16 only, no quantized copy) and after the drop.

Writes runs/E006-read-depths/<model>/summary.json.

    uv run python scripts/depth_perplexity.py
    uv run python scripts/depth_perplexity.py --per-domain 1 --out /tmp/depths   # smoke check
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from foqlens import model as fm
from foqlens.io import read_jsonl, write_json
from foqlens.precision import install
from foqlens.quant import Level

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
DOMAINS = ["biology", "math", "chemistry", "physics"]
LEVELS = (Level.BF16, Level.INT8, Level.NF4, Level.D2, Level.D4, Level.D6, Level.D8)
MIB = 2**20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--per-domain", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("runs/E006-read-depths"))
    return parser.parse_args(argv)


def mean_perplexity(model, tokenizer, texts: list[str]) -> float:
    return statistics.mean(fm.perplexity(model, tokenizer, t) for t in texts)


def allocated_mib() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / MIB


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    texts = [r["text"] for spec in args.domains for r in read_jsonl(args.prompts_dir / f"{spec}.jsonl", args.per_domain)]
    model, tokenizer = fm.load(MODELS[args.model])
    ctl = install(model)
    bf16_mib = allocated_mib()
    perplexity = {}
    for level in LEVELS:
        ctl.set_all(level)
        perplexity[level.name.lower()] = mean_perplexity(model, tokenizer, texts)
    ctl.set_all(Level.D8)
    ctl.drop_bf16()
    summary = {
        "model": MODELS[args.model],
        "revision": fm.REVISIONS[MODELS[args.model]],
        "texts": len(texts),
        "perplexity": perplexity,
        "resident": {
            "d8": mean_perplexity(model, tokenizer, texts),
            "memory_allocated_mib": {"bf16": bf16_mib, "resident": allocated_mib()},
        },
    }
    out = args.out / args.model / "summary.json"
    write_json(out, summary)
    for name, p in perplexity.items():
        print(f"{name:6} {p:14.3f}")
    print(f"resident d8 {summary['resident']['d8']:.3f}  memory {summary['resident']['memory_allocated_mib']}\n-> {out}")
    return out


if __name__ == "__main__":
    main()
