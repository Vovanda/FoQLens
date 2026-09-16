"""The judge's runs: one file per pass over one answers file, rows picked by line, the latest run's verdict counts."""

import json
from pathlib import Path

import pytest

from foqlens.io import judge_run_path, judge_run_summary_path, parse_rows
from foqlens.runs import RunFiles


@pytest.mark.parametrize("spec, rows", [
    ("12", [12]),
    ("1-3,7", [1, 2, 3, 7]),
    (" 5-6 , 2 ", [5, 6, 2]),
    ("3,1-4", [3, 1, 2, 4]),  # each once, in the order asked
])
def test_rows_are_lines_counted_from_one(spec, rows):
    assert parse_rows(spec) == rows


@pytest.mark.parametrize("spec", ["0", "5-3", "a", "1-"])
def test_rows_below_one_or_backwards_are_refused(spec):
    with pytest.raises(ValueError):
        parse_rows(spec)


def test_a_run_lies_under_its_level_and_corpus_with_its_summary_beside_it():
    root = Path("judge")
    assert judge_run_path(root, "d2", "triviaqa", "2026-09-16T17-40-00") == root / "d2/triviaqa/2026-09-16T17-40-00.jsonl"
    assert judge_run_summary_path(root, "d2", "triviaqa", "r").name == "r.summary.json"


def write_run(root: Path, run: str, lines: list[tuple[str, str, bool]]) -> None:
    path = judge_run_path(root, "bf16", "c", run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"corpus": "c", "id": i, "level": "bf16", "judge_kind": kind,
                                        "judge_accepted": accepted, "judge_with_reference": None}) + "\n"
                            for i, kind, accepted in lines), encoding="utf-8")


def test_the_verdict_on_a_question_is_the_latest_run_that_read_it_and_earlier_runs_stay(tmp_path):
    (tmp_path / "answers/bf16").mkdir(parents=True)
    (tmp_path / "answers/bf16/c.jsonl").write_text(json.dumps({"corpus": "c", "id": "1", "level": "bf16"}) + "\n",
                                                   encoding="utf-8")
    judge = tmp_path / "judge"
    write_run(judge, "2026-09-16T15-27-00", [("1", "N/A", False), ("2", "Correct", True)])
    write_run(judge, "2026-09-16T17-40-00", [("1", "Nearly", True)])  # the N/A asked again, alone
    files = RunFiles(tmp_path / "answers", judged=judge)
    got = files.query("select id, run, judge_kind, accepted from verdicts order by id")
    assert got == [{"id": "1", "run": "2026-09-16T17-40-00", "judge_kind": "Nearly", "accepted": True},
                   {"id": "2", "run": "2026-09-16T15-27-00", "judge_kind": "Correct", "accepted": True}]
    assert files.query("select count(*) n from judged") == [{"n": 3}]
