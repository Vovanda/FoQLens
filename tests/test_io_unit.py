"""Reading and writing the files a run leaves beside it."""

from foqlens import io


def test_wordings_take_the_corpus_from_the_file_name(tmp_path):
    (tmp_path / "e006-triviaqa.jsonl").write_text('{"id": "a", "paraphrase": "Who made it?"}\n', encoding="utf-8")
    (tmp_path / "e006-nq_open.jsonl").write_text('{"id": "7", "paraphrase": "where is it"}\n', encoding="utf-8")
    (tmp_path / "other.jsonl").write_text('{"id": "z", "paraphrase": "no"}\n', encoding="utf-8")
    assert io.read_wordings(tmp_path, "e006-*.jsonl") == {("triviaqa", "a"): "Who made it?",
                                                         ("nq_open", "7"): "where is it"}


def test_no_wording_files_means_no_wordings(tmp_path):
    assert io.read_wordings(tmp_path, "e006-*.jsonl") == {}
