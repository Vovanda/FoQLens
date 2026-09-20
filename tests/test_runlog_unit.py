"""The log of a run: stages with their durations, progress throttled to its interval."""

import logging
from pathlib import Path

import pytest

from foqlens.runlog import stage


def test_a_stage_that_raises_logs_its_failure_as_an_error_with_the_traceback(caplog):
    log = logging.getLogger("foqlens.test")
    with caplog.at_level(logging.INFO, logger="foqlens"):
        with pytest.raises(RuntimeError):
            with stage(log, "masks"):
                raise RuntimeError("boom")
    lines = [r.getMessage() for r in caplog.records]
    assert lines[0] == "start masks" and lines[1].startswith("failed masks after ") and len(lines) == 2
    assert caplog.records[1].levelno == logging.ERROR and caplog.records[1].exc_info is not None


def test_a_stage_that_ends_logs_done_with_its_duration(caplog):
    with caplog.at_level(logging.INFO, logger="foqlens"):
        with stage(logging.getLogger("foqlens.test"), "graph"):
            pass
    assert [r.getMessage().split(" in ")[0] for r in caplog.records] == ["start graph", "done graph"]


def test_the_file_sink_writes_one_json_event_a_line_with_the_run_context_and_the_event_fields(tmp_path):
    import json

    from foqlens.runlog import setup

    path = tmp_path / "run.log.jsonl"
    setup(path, {"run": "r1", "script": "test"}, config=Path(__file__).resolve().parents[1] / "configs" / "logging.toml")
    with stage(logging.getLogger("foqlens.test"), "masks"):
        pass
    for handler in logging.getLogger("foqlens").handlers:
        handler.flush()
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    done = events[-1]
    assert done["level"] == "INFO" and done["logger"] == "foqlens.test" and done["message"].startswith("done masks")
    assert done["run"] == "r1" and done["script"] == "test" and done["stage"] == "masks" and "seconds" in done
    assert "timestamp" in done
    logging.getLogger("foqlens").handlers.clear()
