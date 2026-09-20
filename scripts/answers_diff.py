"""Two runs of the same layouts, reply against reply (foqlens.runs.RunFiles.replies_of_two_runs).

What it answers: did a rewrite of the reading path leave the model's replies where they were. The layouts are matched
by name, the questions by corpus, id and prompt; a layout only one of the runs holds is left out.

    uv run python scripts/answers_diff.py --left runs/regulator/e2b-it/answers --right runs/regulator-card/answers
"""

from __future__ import annotations

import argparse
from pathlib import Path

from foqlens.runs import RunFiles


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", type=Path, required=True, help="a run's answers folder")
    parser.add_argument("--right", type=Path, required=True, help="the other run's answers folder")
    args = parser.parse_args(argv)
    rows = RunFiles.replies_of_two_runs(args.left, args.right)
    for row in rows:
        print(f"{row['level']:52} {row['answered']:5} answered, {row['differ']:5} differ")
    return rows


if __name__ == "__main__":
    main()
