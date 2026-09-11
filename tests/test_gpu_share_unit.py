"""The GPU share throttle with a made-up clock: no GPU."""

import pytest

from foqlens.gpu_share import DEFAULT_SHARE, ENV, FULL, Throttle, default_share


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_a_batch_is_followed_by_rest_that_keeps_the_gpu_busy_its_share_of_the_time():
    clock = FakeClock()
    throttle = Throttle(0.8, clock=clock, sleep=clock.sleep)
    with throttle.batch():
        clock.now += 2.0  # the batch kept the GPU busy for 2 s
    assert clock.slept == [pytest.approx(0.5)]
    assert 2.0 / (2.0 + clock.slept[0]) == pytest.approx(0.8)


def test_the_full_share_never_rests():
    clock = FakeClock()
    with Throttle(1.0, clock=clock, sleep=clock.sleep).batch():
        clock.now += 3.0
    assert clock.slept == []
    assert FULL.share == 1.0


def test_a_share_outside_zero_to_one_is_refused():
    for share in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            Throttle(share)


def test_the_default_share_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert default_share() == DEFAULT_SHARE == 0.8
    monkeypatch.setenv(ENV, "0.6")
    assert default_share() == 0.6
