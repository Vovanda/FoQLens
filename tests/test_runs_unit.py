"""Slices over the run files: small files written here, and stage 1 against what its selection wrote."""

import json
from pathlib import Path

import pytest

from foqlens.corpora import CORPORA, PASSAGE, TWO_PASSAGES, WEIGHTS
from foqlens.judging import GRADE_GROUPS, GRADES, NOT_READ
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


def test_two_runs_are_compared_reply_against_reply_on_the_layouts_both_hold(tmp_path):
    def reply(i: str, text: str, level: str = "d2") -> dict:
        return {"corpus": "c", "id": i, "level": level, "prompt": "short-0", "reply": text}

    write_jsonl(tmp_path / "left/d2/c.jsonl", [reply("1", "Rome"), reply("2", "Paris"), reply("3", "Bonn")])
    write_jsonl(tmp_path / "left/d4/c.jsonl", [reply("1", "Rome", "d4")])  # only the left run holds d4
    write_jsonl(tmp_path / "right/d2/c.jsonl", [reply("1", "Rome"), reply("2", "Lyon")])  # 3 is unanswered here
    assert RunFiles.replies_of_two_runs(tmp_path / "left", tmp_path / "right") == [
        {"level": "d2", "answered": 2, "differ": 1}]


def test_accepted_is_the_reasoning_judges_verdict_where_it_gave_one_and_the_one_token_judges_elsewhere(tmp_path):
    write_jsonl(tmp_path / "answers/bf16/c.jsonl", [answer("1", 1.0, 0.9), answer("2", 0.0, 0.1)])
    write_jsonl(tmp_path / "answers/d2/c.jsonl",
                [{**answer("3", 0.0, float("nan")), "level": "d2", "judge_kind": "Garbage", "judge_accepted": False},
                 {**answer("4", 1.0, float("nan")), "level": "d2", "judge_kind": "Correct", "judge_accepted": True}])
    rows = RunFiles(tmp_path / "answers").query("select id, accepted from answers order by id")
    assert [(r["id"], r["accepted"]) for r in rows] == [("1", True), ("2", False), ("3", False), ("4", True)]


def test_a_reply_is_the_same_answer_through_a_capital_a_full_stop_and_a_double_space(tmp_path):
    write_jsonl(tmp_path / "answers/top/c.jsonl",
                [{"corpus": "c", "id": i, "level": "top", "reply": r}
                 for i, r in (("1", "water tower"), ("2", "water tower"), ("3", "water tower"),
                              ("4", "commutative ring R"))])
    write_jsonl(tmp_path / "answers/map/c.jsonl",
                [{"corpus": "c", "id": i, "level": "map", "reply": r}
                 for i, r in (("1", "Water tower"), ("2", "water tower."), ("3", "water  tower"),
                              ("4", "commutative ring"))])
    got = RunFiles(tmp_path / "answers").same_reply("top")[0]
    assert got["same"] == pytest.approx(3 / 4)  # only the answer that lost the R differs
    assert got["same_word_for_word"] == pytest.approx(0.0)  # word for word, the layout is charged for all four
    assert got["one_inside_the_other"] == pytest.approx(1.0)  # the truncation is the reference's own opening
    assert 0.9 < got["likeness"] <= 1.0  # and by likeness all four stand next to the reference


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


def judged_run(root: Path, corpus: str, kinds: list[str]) -> None:
    """Answers left unread, an earlier judge run calling everything Garbage, a later one giving `kinds`; one more
    question sits in the unknown share and is never judged."""
    def verdict(i: str, kind: str) -> dict:
        return {"corpus": corpus, "id": i, "level": "d2", "judge_kind": kind, "judge_accepted": kind in ("Correct", "Nearly")}
    write_jsonl(root / f"answers/d2/{corpus}.jsonl", [verdict(str(i), "N/A") for i in range(len(kinds) + 1)])
    write_jsonl(root / f"judge/d2/{corpus}/2026-01-01T00-00-00.jsonl", [verdict(str(i), "Garbage") for i in range(len(kinds))])
    write_jsonl(root / f"judge/d2/{corpus}/2026-01-02T00-00-00.jsonl", [verdict(str(i), k) for i, k in enumerate(kinds)])
    (root / "frozen").mkdir(exist_ok=True)
    (root / f"frozen/{corpus}.json").write_text(json.dumps(
        {"corpus": corpus, "kept": [str(i) for i in range(len(kinds))], "unknown_share": [str(len(kinds))], "tuning": []}))


def test_grades_are_shares_of_the_grade_groups_on_one_part_from_the_latest_judge_run(tmp_path):
    judged_run(tmp_path, "triviaqa", ["Correct", "Nearly", "Partial", "Wrong", "Related", "Noise", "Garbage", "N/A"])
    runs = RunFiles(tmp_path / "answers", frozen=tmp_path / "frozen", judged=tmp_path / "judge")
    assert runs.grades("kept") == {"d2": {"n": 8, "excellent": 2 / 8, "good": 1 / 8, "bad": 2 / 8,
                                          "incoherent": 2 / 8, "not_read": 1 / 8}}
    judged_kinds = {k for kinds in GRADE_GROUPS.values() for k in kinds} | {NOT_READ}
    assert judged_kinds == {kind for kind, _ in GRADES} | {NOT_READ}


def test_grades_by_regime_split_the_level_by_where_the_answer_comes_from(tmp_path):
    judged_run(tmp_path, "triviaqa", ["Correct", "Wrong"])
    judged_run(tmp_path, "squad_v2", ["Correct", "Correct", "Partial", "Garbage"])
    judged_run(tmp_path, "hotpotqa", ["Nearly"])
    runs = RunFiles(tmp_path / "answers", frozen=tmp_path / "frozen", judged=tmp_path / "judge")
    got = runs.grades_by_regime("kept")
    assert {key: (row["n"], row["excellent"]) for key, row in got.items()} == {
        ("d2", WEIGHTS): (2, 0.5), ("d2", PASSAGE): (4, 0.5), ("d2", TWO_PASSAGES): (1, 1.0)}
    assert {name for name, corpus in CORPORA.items() if corpus.passage} == {"squad_v2", "hotpotqa"}
