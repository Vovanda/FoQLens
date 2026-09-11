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


def test_step3_quality_smoke(tmp_path):
    step3 = load_script("step3_quality")
    out = step3.main(
        ["--limit", "2", "--domains", "biology", "heldout/geography", "--apertures", "0.1",
         "--pooled-batch", "2", "--gradient-batch", "2", "--eval-batch", "4",
         "--prompts-dir", str(ROOT / "prompts"), "--out", str(tmp_path)]
    )
    summary = json.loads(Path(out).read_text())
    assert summary["questions"] == {"biology": 2, "geography": 2}
    assert set(summary["configs"]) == {
        "uniform_bf16", "uniform_int8", "uniform_nf4", "uniform_zero",
        "random_0.100", "directed_pooled_0.100", "directed_gradient_0.100",
    }
    assert summary["configs"]["uniform_zero"]["mean_bits"] == 0.0
    assert summary["configs"]["uniform_int8"]["mean_bits"] == 8.0
    assert summary["configs"]["random_0.100"]["mean_bits"] == pytest.approx(
        summary["configs"]["directed_pooled_0.100"]["mean_bits"], abs=0.05
    )
    assert summary["gpu"]["eval"]["samples"] > 0
    assert summary["gpu"]["masks"]["memory_allocated_peak_mib"] > 0
    assert (tmp_path / "e2b" / "raw" / "masks.npy").exists()
