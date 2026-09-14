from datetime import datetime
from pathlib import Path

import pytest
import torch

# The card is watched through the whole test session: a run of the GPU tests that heats it is reported
# at the end, not left to be noticed by ear (2026-09-14, 83 °C), and goes into the station's log.
GPU_WATCH = pytest.StashKey()
STATION_TESTS = Path(__file__).resolve().parents[1] / "runs" / "station" / "tests"
OUTCOMES = ("passed", "failed", "error")


def ran_gpu_tests(terminalreporter) -> bool:
    return any("gpu" in r.keywords for k in OUTCOMES for r in terminalreporter.stats.get(k, []))


def pytest_sessionstart(session):
    if torch.cuda.is_available():
        from foqlens.gpu_monitor import GpuMonitor

        session.config.stash[GPU_WATCH] = GpuMonitor().__enter__()


def pytest_terminal_summary(terminalreporter, config):
    monitor = config.stash.get(GPU_WATCH, None)
    if monitor is None:
        return
    monitor.__exit__(None, None, None)
    s = monitor.summary()
    if s["samples"]:
        terminalreporter.write_line(
            f"GPU during the tests: peak {s['temperature_peak_c']:.0f} C (mean {s['temperature_mean_c']:.0f}), "
            f"peak {s['power_peak_w']:.0f} W, utilization {s['utilization_mean']:.0f}% mean, {s['samples']} samples")
    if s["samples"] and ran_gpu_tests(terminalreporter):
        from foqlens.io import write_json
        from foqlens.station import session_record

        outcomes = {k: len(terminalreporter.stats.get(k, [])) for k in OUTCOMES}
        write_json(STATION_TESTS / f"{datetime.now():%Y-%m-%dT%H-%M-%S}.json", session_record(s, outcomes))


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="module")
def e2b_eager():
    """Gemma 4 E2B with eager attention and the precision controller installed.

    Module scope: released at the end of each test file, so two 12 GB models never share the GPU.
    """
    from foqlens import model as fm
    from foqlens.precision import install

    model, tokenizer = fm.load(fm.E2B, attn_implementation="eager")
    ctl = install(model)
    yield model, tokenizer, ctl
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def e2b_it_sdpa():
    """Gemma 4 E2B-it - the model the corpus is selected on - with sdpa attention and the controller installed."""
    from foqlens import model as fm
    from foqlens.precision import install

    model, tokenizer = fm.load(fm.E2B_IT, attn_implementation="sdpa")
    ctl = install(model)
    yield model, tokenizer, ctl
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def e2b_sdpa():
    """Gemma 4 E2B with sdpa attention - the attention of the runs (Bench.load) - and the controller installed."""
    from foqlens import model as fm
    from foqlens.precision import install

    model, tokenizer = fm.load(fm.E2B, attn_implementation="sdpa")
    ctl = install(model)
    yield model, tokenizer, ctl
    del model
    torch.cuda.empty_cache()
