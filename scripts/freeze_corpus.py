"""Freeze the selected corpus: one file of question numbers per corpus, which every later run reads.

Reads the stage 1 answers and Claude's verdicts of one model and level, the questions spent on choosing
the prompt, and writes <out>/<corpus>.json (foqlens.selection.FrozenCorpus).

    uv run python scripts/freeze_corpus.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from foqlens import corpora
from foqlens.io import answers_path, read_answers, read_verdicts, write_json
from foqlens.prompt_variants import SETUPS
from foqlens.selection import freeze


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("runs/reference/stage1/e2b-it"))
    parser.add_argument("--level", default="bf16")
    parser.add_argument("--tuning", type=Path, default=Path("runs/reference/prompt-tuning/e2b-it/summary.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("corpus/e2b-it"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    spent = json.loads(args.tuning.read_text(encoding="utf-8"))["corpora"]
    for corpus in SETUPS:
        rows = corpora.read(corpus)[0]
        answers = read_answers(answers_path(args.root / "answers", args.level, corpus))
        claude = {i: v.verdict for i, v in read_verdicts(answers_path(args.root / "verdicts", args.level, corpus)).items()}
        frozen = freeze(answers, claude, {r.id: (r.question, tuple(r.answers)) for r in rows},
                        tuple(spent[corpus]["tuning_ids"]), args.seed)
        write_json(args.out / f"{corpus}.json", frozen.to_json())
        print(f"{corpus:22} kept {len(frozen.kept):5} (two-way {len(frozen.two_way):4})  excluded {len(frozen.excluded):5}"
              f"  unknown share {len(frozen.unknown_share):4}  tuning {len(frozen.tuning):3}")


if __name__ == "__main__":
    main()
