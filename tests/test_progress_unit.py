"""The progress line of long runs (foqlens/progress.py)."""

from datetime import datetime

import pytest

from foqlens.progress import Progress


def clock(times):
    ticks = iter(times)
    return lambda: next(ticks)


def hhmm(t: float) -> str:
    return f"{datetime.fromtimestamp(t):%H:%M}"


def test_the_expected_end_is_the_mean_step_time_times_the_steps_left():
    p = Progress(4, "cell", clock=clock([1000.0, 1060.0, 1300.0]))
    assert p.step("a") == f"cell [1/4] a (25%), ETA {hhmm(1060.0 + 3 * 60.0)}"
    assert p.step("b") == f"cell [2/4] b (50%), ETA {hhmm(1300.0 + 2 * 150.0)}"


def test_the_last_step_ends_now_and_there_is_no_step_after_it():
    p = Progress(2, clock=clock([0.0, 50.0, 80.0]))
    p.step("a")
    assert p.step("b") == f"[2/2] b (100%), ETA {hhmm(80.0)}"
    with pytest.raises(ValueError):
        p.step("c")


def test_the_percent_rounds_down_so_that_100_means_done():
    p = Progress(3, clock=clock([0.0, 1.0, 2.0]))
    assert "(33%)" in p.step("a") and "(66%)" in p.step("b")


def test_a_run_of_negative_length_is_refused():
    with pytest.raises(ValueError):
        Progress(-1)
