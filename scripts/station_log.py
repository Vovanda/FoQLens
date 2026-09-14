"""Rebuilds the log and the peaks of docs/station.md from every run summary under runs/.

A summary without a date of its own - written before GpuMonitor kept a clock - is dated by the
commit that added it, or by the file's time when it was never committed. The runs of the first
corpus were deleted with it (DELETED_WITH_THE_CORPUS); their results carry no verdict, but the hours
the card spent on them are the station's all the same, so their summaries are read from the history.

    uv run python scripts/station_log.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path, PurePosixPath

from foqlens.station import Entry, entry, render, splice

# 1e89a81, 2026-09-13: "delete the runs and results of the corpus that could not carry a verdict"
DELETED_WITH_THE_CORPUS = "1e89a81"
SUMMARY = "summary*.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--doc", type=Path, default=Path("docs/station.md"))
    return parser.parse_args(argv)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def added_on(path: str, at: str = "HEAD") -> str | None:
    """The author's date: the history was squashed on 2026-09-13, which moved every commit's own date to it."""
    dates = git("log", "--diff-filter=A", "--format=%as", at, "--", path).split()
    return dates[-1] if dates else None


def run_name(path: PurePosixPath, runs: PurePosixPath) -> str:
    """The run's folder under runs/, and the summary's own name when it is not the plain one."""
    name = path.parent.relative_to(runs).as_posix()
    return name if path.name == "summary.json" else f"{name}/{path.stem}"


def on_disk(runs: Path) -> list[Entry]:
    out = []
    for path in sorted(runs.glob(f"**/{SUMMARY}")) + sorted((runs / "station").glob("**/*.json")):
        posix = PurePosixPath(path.as_posix())
        run = run_name(posix, PurePosixPath(runs.as_posix())) if path.match(SUMMARY) else path.stem
        dated = added_on(path.as_posix()) or date.fromtimestamp(path.stat().st_mtime).isoformat()
        out.append(entry(json.loads(path.read_text(encoding="utf-8")), run, dated))
    return out


def deleted(runs: Path) -> list[Entry]:
    before = f"{DELETED_WITH_THE_CORPUS}^"
    out = []
    for name in git("ls-tree", "-r", "--name-only", before, "--", runs.as_posix()).split():
        path = PurePosixPath(name)
        if not path.match(SUMMARY) or Path(name).exists():  # a run kept on disk is read there
            continue
        record = json.loads(git("show", f"{before}:{name}"))
        record.setdefault("note", f"deleted with the first corpus in {DELETED_WITH_THE_CORPUS}")
        out.append(entry(record, run_name(path, PurePosixPath(runs.as_posix())), added_on(name, before)))
    return out


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    entries = [e for e in deleted(args.runs) + on_disk(args.runs) if e is not None]
    args.doc.write_text(splice(args.doc.read_text(encoding="utf-8"), render(entries)), encoding="utf-8")
    print(f"{len(entries)} entries -> {args.doc}")
    return args.doc


if __name__ == "__main__":
    main()
