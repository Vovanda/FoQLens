"""The run scripts end to end on a few questions: they run, and write what the analysis reads."""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_answers_smoke(tmp_path):
    ref = load_script("reference_answers")
    out = ref.main(["--domains", "biology", "--per-domain", "1", "--max-new-tokens", "6",
                    "--prompts-dir", str(ROOT / "prompts"), "--out", str(tmp_path)])
    [row] = json.loads(Path(out).read_text())
    assert set(row["answers"]) == {"bf16", "int8", "nf4", "zero"} and row["domain"] == "biology"
    assert (tmp_path / "e2b" / "answers.md").read_text().startswith("# Reference answers")
