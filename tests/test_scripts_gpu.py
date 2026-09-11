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


def test_step3_backbone_smoke(tmp_path):
    bb = load_script("step3_backbone")
    out = bb.main(["--limit", "3", "--precision-share", "0.95", "--shares", "0.8", "--pooled-batch", "4", "--gradient-batch", "2",
                   "--eval-batch", "6", "--prompts-dir", str(ROOT / "prompts"), "--out", str(tmp_path)])
    summary = json.loads(Path(out).read_text())
    assert {"backbone_0.950", "bb0.80_random_0.950", "bb0.80_own_gradient_0.950", "bb0.80_other_pooled_0.950"} <= set(summary["configs"])
    assert "own_minus_other" in summary["comparisons"]["biology-math|gradient|0.80|0.950"]
    assert "backbone_minus_random" in summary["comparisons"]["history-geography|backbone|0.950"]
    bits = {name: c["mean_bits"] for name, c in summary["configs"].items() if name.endswith("0.950")}
    assert max(bits.values()) - min(bits.values()) < 0.05  # every policy at one precision share spends the same budget


def test_step3_dilation_smoke(tmp_path):
    dl = load_script("step3_dilation")
    out = dl.main(["--limit", "3", "--precision-share", "0.95", "--pooled-batch", "4", "--gradient-batch", "2",
                   "--eval-batch", "6", "--prompts-dir", str(ROOT / "prompts"), "--out", str(tmp_path)])
    s = json.loads(Path(out).read_text())
    assert {"backbone_0.950", "bb0.80_random_0.950", "bb0.80_own_gradient_0.950", "bb0.80_own_gradient_struct_0.950",
            "bb0.80_other_pooled_index_0.950"} <= set(s["configs"])
    assert {"own_minus_other_struct", "struct_minus_none", "struct_minus_index"} <= set(s["comparisons"]["history-geography|gradient|0.950"])
    bits = {name: c["mean_bits"] for name, c in s["configs"].items() if name.endswith("0.950")}
    assert max(bits.values()) - min(bits.values()) < 0.05  # dilation spends the same budget


def test_step3_zones_smoke(tmp_path, capsys):
    sz = load_script("step3_zones")
    out = sz.main(["--limit", "3", "--precision-share", "0.25", "--focus-area", "0.5", "0.0", "--pooled-batch", "4",
                   "--gradient-batch", "2", "--eval-batch", "6", "--prompts-dir", str(ROOT / "prompts"), "--out", str(tmp_path)])
    log = capsys.readouterr().out
    assert "cell [1/2] ps0.250 fa0.50 (50%), ETA " in log and "cell [2/2] ps0.250 fa0.00 (100%), ETA " in log
    s = json.loads(Path(out).read_text())
    assert {"zone_uniform_ps0.250", "zone_own_gradient_fa0.50_ps0.250", "zone_random_pooled_fa0.00_ps0.250",
            "zone_other_pooled_fa0.50_ps0.250", "zone_fixed_backbone_fa0.50_ps0.250", "uniform_d4"} <= set(s["configs"])
    assert {"own_minus_random", "own_minus_other", "own_minus_uniform"} <= set(s["comparisons"]["biology-math|gradient|ps0.250|fa0.50"])
    assert s["cells_done"] == [[0.25, 0.5], [0.25, 0.0]]  # the promising cell first
    bits = [c["mean_bits"] for name, c in s["configs"].items() if name.startswith("zone_")]
    assert max(abs(b - 5.0) for b in bits) < 0.05  # precision share 0.25 spends 4 + 4 x 0.25 bits


def test_step3_quality_smoke(tmp_path):
    step3 = load_script("step3_quality")
    out = step3.main(
        ["--limit", "2", "--domains", "biology", "heldout/geography", "--precision-share", "0.1", "--coarse", "nf4",
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
