"""Slices over the run files: small files written here, and stage 1 against what its selection wrote."""

import json
from pathlib import Path

import pytest

from foqlens.runs import RunFiles

STAGE1 = Path("runs/reference/stage1/e2b-it")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def answer(i: str, em: float, judge: float) -> dict:
    return {"corpus": "c", "id": i, "level": "bf16", "exact_match": em, "judge_with_reference": judge}


def test_agreement_counts_what_the_exact_match_and_the_judge_say_against_the_reading(tmp_path):
    write_jsonl(tmp_path / "answers/bf16/c.jsonl",
                [answer("1", 1.0, 0.9), answer("2", 0.0, 0.9), answer("3", 0.0, 0.1), answer("4", 1.0, 0.1)])
    write_jsonl(tmp_path / "verdicts/bf16/c.jsonl",
                [{"corpus": "c", "id": i, "level": "bf16", "reading": r}
                 for i, r in (("1", "right"), ("2", "other_words"), ("3", "wrong"))])
    got = RunFiles(tmp_path / "answers", tmp_path / "verdicts").agreement_with_readings("bf16")
    # read: 1 right (EM yes, judge yes), 2 other words (EM no, judge yes), 3 wrong (EM no, judge no); 4 unread
    assert got == {"c": {"read": 3, "exact_match": pytest.approx(2 / 3), "judge": 1.0}}


def test_a_slice_reads_the_files_as_they_are_when_asked(tmp_path):
    path = tmp_path / "answers/bf16/c.jsonl"
    write_jsonl(path, [answer("1", 1.0, 0.9)])
    runs = RunFiles(tmp_path / "answers")
    write_jsonl(path, [answer("1", 1.0, 0.9), answer("2", 0.0, 0.1)])
    assert runs.query("select count(*) n from answers") == [{"n": 2}]


def test_stage1_agreement_equals_what_its_selection_wrote():
    passed = json.loads((STAGE1 / "passed-bf16.json").read_text(encoding="utf-8"))
    got = RunFiles(STAGE1 / "answers", STAGE1 / "verdicts").agreement_with_readings("bf16")
    assert set(got) == set(passed)
    for corpus, row in passed.items():
        assert got[corpus]["read"] == row["read"]
        assert got[corpus]["exact_match"] == pytest.approx(row["agreement_with_claude"]["exact_match"], abs=1e-12)
        assert got[corpus]["judge"] == pytest.approx(row["agreement_with_claude"]["judge"], abs=1e-12)


def test_the_frozen_corpus_reads_as_one_row_per_question():
    frozen = RunFiles(STAGE1 / "answers", frozen=Path("corpus/e2b-it"))
    counts = {r["part"]: r["n"] for r in frozen.query(
        "select part, count(*) n from frozen where corpus = 'triviaqa' group by part")}
    source = json.loads(Path("corpus/e2b-it/triviaqa.json").read_text(encoding="utf-8"))
    assert counts == {"kept": len(source["kept"]), "unknown_share": len(source["unknown_share"]),
                      "tuning": len(source["tuning"])}
