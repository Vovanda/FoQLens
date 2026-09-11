"""Reference answers: what the model says, in its own words, at every uniform precision level.

A few questions per domain, asked without their options, answered greedily at bf16, int8, nf4 and
with every block removed (ZERO). Writes runs/reference/<model>/answers.json and answers.md.

    uv run python scripts/reference_answers.py
    uv run python scripts/reference_answers.py --per-domain 1 --max-new-tokens 8 --out /tmp/ref   # smoke check
"""

from __future__ import annotations

import argparse
from pathlib import Path

from foqlens import model as fm
from foqlens.generation import generate_answer, question_prompt
from foqlens.io import read_jsonl, write_json
from foqlens.precision import install
from foqlens.quant import Level

MODELS = {"e2b": fm.E2B, "e4b": fm.E4B}
DOMAINS = ["biology", "math", "chemistry", "physics", "heldout/history", "heldout/geography"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="e2b")
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--per-domain", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--out", type=Path, default=Path("runs/reference"))
    return parser.parse_args(argv)


def to_markdown(rows: list[dict]) -> str:
    lines = ["# Reference answers", ""]
    for r in rows:
        lines += [f"## {r['domain']}: {r['question']}", "", f"Right option: {r['right']}", ""]
        lines += [f"- **{level}**: {text or '(empty)'}" for level, text in r["answers"].items()]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    model, tokenizer = fm.load(MODELS[args.model], attn_implementation="sdpa")
    ctl = install(model)
    rows = []
    for spec in args.domains:
        for q in read_jsonl(args.prompts_dir / f"{spec}.jsonl", args.per_domain):
            answers = {}
            for level in Level:
                ctl.set_all(level)
                answers[level.name.lower()] = generate_answer(model, tokenizer, question_prompt(q["text"]), args.max_new_tokens)
            rows.append({"domain": Path(spec).name, "question": q["text"], "right": q["choices"][q["answer"]], "answers": answers})
    out_dir = args.out / args.model
    write_json(out_dir / "answers.json", rows)
    (out_dir / "answers.md").write_text(to_markdown(rows), encoding="utf-8")
    print(f"{len(rows)} questions x {len(Level)} levels -> {out_dir / 'answers.md'}")
    return out_dir / "answers.json"


if __name__ == "__main__":
    main()
