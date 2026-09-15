"""Claude reads the stage 1 answers sheet by sheet, and the questions the model knows are listed.

`sheet` lays out the next answers of one corpus waiting in one turn (foqlens.selection.Turn) and
keeps their question numbers in the sheet file; `mark` turns Claude's marks on that sheet into
verdicts/<level>/<corpus>.jsonl (foqlens.reading_sheet); `passed` writes passed-<level>.json - every
corpus's known questions with the judge that decided, and what still waits in each turn.

    uv run python scripts/claude_reading.py sheet --corpus triviaqa --turn 1 --size 100 --sheet /tmp/sheet.json
    uv run python scripts/claude_reading.py mark --sheet /tmp/sheet.json --marks "o 3 7" "w 5: too general"
    uv run python scripts/claude_reading.py passed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from foqlens import corpora
from foqlens.io import answers_path, append_verdicts, read_answers, read_verdicts, write_json
from foqlens.prompt_variants import SETUPS
from foqlens.reading_sheet import DEFAULTS, Sheet, parse_marks, render
from foqlens.selection import JUDGE_YES, ClaudeVerdict, Turn, Verdict, turn, two_way_choice, verdict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("runs/reference/stage1/e2b-it"))
    parser.add_argument("--level", default="bf16")
    commands = parser.add_subparsers(dest="command", required=True)
    sheet = commands.add_parser("sheet")
    sheet.add_argument("--corpus", required=True, choices=list(SETUPS))
    sheet.add_argument("--turn", type=int, required=True, choices=[int(t) for t in Turn])
    sheet.add_argument("--size", type=int, default=100)
    sheet.add_argument("--passage", action="store_true", help="show the passage under each answer")
    sheet.add_argument("--sheet", type=Path, required=True)
    mark = commands.add_parser("mark")
    mark.add_argument("--sheet", type=Path, required=True)
    mark.add_argument("--marks", nargs="*", default=[], help='one reading per argument: "w 3 7 12", "o 5: note"')
    commands.add_parser("passed")
    return parser.parse_args(argv)


def verdicts_file(args: argparse.Namespace, corpus: str) -> Path:
    return answers_path(args.root / "verdicts", args.level, corpus)


def load(args: argparse.Namespace, corpus: str):
    rows = {r.id: r for r in corpora.read(corpus)[0]}
    return rows, read_answers(answers_path(args.root / "answers", args.level, corpus)), read_verdicts(verdicts_file(args, corpus))


def make_sheet(args: argparse.Namespace) -> None:
    rows, answers, read = load(args, args.corpus)
    waiting = [a for a in answers if a.id not in read and turn(a, bool(rows[a.id].answers)) == args.turn]
    on_sheet = waiting[:args.size]
    sheet = Sheet(args.corpus, args.level, Turn(args.turn), tuple(a.id for a in on_sheet))
    write_json(args.sheet, sheet.to_json())
    default = DEFAULTS[sheet.turn]
    print(f"{args.corpus}, turn {sheet.turn.name}: {len(on_sheet)} of {len(waiting)} waiting; "
          f"unmarked = {default or 'none, mark every answer'}")
    print(render(on_sheet, rows, args.passage))


def mark(args: argparse.Namespace) -> None:
    sheet = Sheet.from_json(json.loads(args.sheet.read_text(encoding="utf-8")))
    path = verdicts_file(args, sheet.corpus)
    already = read_verdicts(path).keys() & set(sheet.ids)
    if already:
        raise SystemExit(f"{len(already)} answers of this sheet are read already: it was marked once")
    readings = parse_marks("\n".join(args.marks), len(sheet.ids), DEFAULTS[sheet.turn])
    today = date.today().isoformat()
    append_verdicts(path, [ClaudeVerdict(sheet.corpus, i, sheet.level, reading, today, note)
                           for i, (reading, note) in zip(sheet.ids, readings, strict=True)])
    print(f"{sheet.corpus}: {len(sheet.ids)} read -", dict(Counter(str(r) for r, _ in readings)))


def passed(args: argparse.Namespace) -> None:
    summary = {}
    for corpus in SETUPS:
        rows, answers, read = load(args, corpus)
        known, abstained, reasons, waiting = [], [], Counter(), Counter()
        agree = Counter()
        for a in answers:
            answerable = bool(rows[a.id].answers)
            claude = read.get(a.id)
            v, reason = verdict(a, claude.verdict if claude else None, answerable)
            reasons[f"{v}:{reason}"] += 1
            if v == Verdict.KNOWN:
                (known if answerable else abstained).append(a.id)
            if claude:
                agree["exact_match"] += (a.exact_match == 1.0) == (v == Verdict.KNOWN)
                agree["judge"] += (a.judge_with_reference > JUDGE_YES) == (v == Verdict.KNOWN)
            elif (t := turn(a, answerable)) is not None:
                waiting[t.name] += 1
        two_way = [i for i in known if two_way_choice(rows[i].question, rows[i].answers)]
        summary[corpus] = {"answers": len(answers), "passed": len(known), "read": len(read),
                           "agreement_with_claude": {k: n / len(read) for k, n in agree.items()} if read else {},
                           "waiting": dict(waiting), "reasons": dict(reasons),
                           "passed_ids": known, "two_way_ids": two_way, "abstained_ids": abstained}
        print(f"{corpus:22} passed {len(known):5} (two-way {len(two_way):4})  read {len(read):5}  waiting {dict(waiting)}")
    write_json(args.root / f"passed-{args.level}.json", summary)


def main(argv: list[str] | None = None) -> None:
    # The corpora carry every script there is; a Windows console prints cp1251 unless told otherwise.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    {"sheet": make_sheet, "mark": mark, "passed": passed}[args.command](args)


if __name__ == "__main__":
    main()
