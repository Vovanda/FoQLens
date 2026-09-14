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


def test_depth_perplexity_smoke(tmp_path):
    dp = load_script("depth_perplexity")
    out = dp.main(["--per-domain", "1", "--prompts-dir", str(ROOT / "prompts"), "--out", str(tmp_path)])
    s = json.loads(Path(out).read_text())
    assert set(s["perplexity"]) == {"bf16", "int8", "nf4", "d2", "d4", "d6", "d8"} and s["texts"] == 4
    assert s["resident"]["d8"] == s["perplexity"]["d8"]  # the resident copy reads exactly the same weights
    memory = s["resident"]["memory_allocated_mib"]
    assert memory["resident"] < memory["bf16"]


def test_step3_injection_smoke(tmp_path):
    inj = load_script("step3_injection")
    out = inj.main(["--limit", "3", "--precision-share", "0.95", "--pooled-batch", "4", "--gradient-batch", "2",
                    "--eval-batch", "6", "--prompts-dir", str(ROOT / "prompts"), "--out", str(tmp_path)])
    summary = json.loads(Path(out).read_text())
    assert summary["questions"] == {"biology": 3, "math": 3, "history": 3, "geography": 3}
    assert set(summary["comparisons"]) == {
        f"{pair}|{s}|0.950" for pair in ("biology-math", "history-geography") for s in ("pooled", "gradient")
    }
    c = summary["comparisons"]["biology-math|pooled|0.950"]["own_minus_other"]
    assert c["lo"] <= c["mean"] <= c["hi"] and c["n"] == 6
    assert {"own_topic_gradient_0.950", "other_topic_pooled_0.950", "random_0.950", "uniform_zero"} <= set(summary["configs"])
    assert {"share", "cooling_breaks", "cooling_break_s", "temperature_peak_c", "thermal_pause_s"} <= set(summary["pacer"])


def test_step3_filter_smoke(tmp_path, capsys):
    sl = load_script("step3_filter")
    out = sl.main(["--limit", "3", "--floor", "d4", "zero", "--focus-area", "0.5", "--focus-strength", "1.0",
                   "--pooled-batch", "4", "--gradient-batch", "2", "--eval-batch", "6",
                   "--prompts-dir", str(ROOT / "prompts"), "--out", str(tmp_path)])
    log = capsys.readouterr().out
    assert "cell [1/2] d4 fa0.50 fs1.00 (50%), ETA " in log and "cell [2/2] zero fa0.50 fs1.00 (100%), ETA " in log
    s = json.loads(Path(out).read_text())
    assert {"zones_own_d4_fa0.50_fs1.00", "zones_other_d4_fa0.50_fs1.00", "zones_backbone_d4_fa0.50_fs1.00",
            "zones_nomask_d4_fa0.50_fs1.00", "floor_d4", "floor_zero", "uniform_bf16"} <= set(s["configs"])
    assert {"own_minus_other", "own_minus_nomask", "own_minus_backbone", "own_minus_floor", "own_minus_bf16"}         <= set(s["comparisons"]["biology-math|d4|fa0.50|fs1.00"])
    assert s["cells_done"] == [["d4", 0.5, 1.0], ["zero", 0.5, 1.0]]  # the D4 floor first
    assert s["pacer"]["cooling_breaks"] == 0 and s["pacer"]["temperature_peak_c"] < 80  # a smoke is short and cool
    cfg = s["configs"]
    # no budget: the layout costs what its zones ask for, and the shuffled control costs exactly the same
    assert cfg["zones_own_d4_fa0.50_fs1.00"]["mean_bits"] == pytest.approx(cfg["zones_nomask_d4_fa0.50_fs1.00"]["mean_bits"])
    assert cfg["floor_d4"]["mean_bits"] == pytest.approx(4.0)
    assert cfg["floor_zero"]["mean_bits"] == pytest.approx(0.0)
    assert 4.0 < cfg["zones_own_d4_fa0.50_fs1.00"]["mean_bits"] < 8.0
