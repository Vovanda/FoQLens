"""The GPU share throttle with a made-up clock: no GPU."""

import pytest

from foqlens.gpu_share import DEFAULT_SHARE, ENV, FULL, Cooldown, ThermalGuard, Throttle, default_share


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


class FakeSensor:
    """Temperatures read in turn; the last one repeats."""

    def __init__(self, *readings: float):
        self.readings = list(readings)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.readings[min(self.calls, len(self.readings)) - 1]


def guard(sensor, clock, share=1.0):
    return ThermalGuard(Throttle(share, clock=clock, sleep=clock.sleep), sensor, ceiling=80, resume=72, soft=75,
                        poll=1.0, clock=clock, sleep=clock.sleep)


def test_a_cool_card_adds_no_rest_to_the_share():
    clock = FakeClock()
    with guard(FakeSensor(60), clock, share=0.8).batch():
        clock.now += 2.0
    assert clock.slept == [pytest.approx(0.5)]  # the share's own rest, nothing more


def test_the_rest_grows_as_the_card_nears_its_ceiling():
    clock = FakeClock()
    with guard(FakeSensor(70, 77.5), clock).batch():   # cool at the start, 77.5 after: half the soft band
        clock.now += 2.0
    assert clock.slept == [pytest.approx(1.0)]          # half the batch's busy time


def test_at_the_ceiling_the_run_waits_until_the_card_is_back_under_resume():
    clock = FakeClock()
    sensor = FakeSensor(81, 79, 75, 72, 60)
    g = guard(sensor, clock)
    with g.batch():
        assert sensor.calls == 4                        # 81 at the start, then 79, 75, 72 while paused
        clock.now += 1.0
    assert clock.slept[:3] == [1.0, 1.0, 1.0]
    assert g.paused_s == pytest.approx(3.0) and g.peak == 81


def test_the_sensor_is_read_at_most_once_per_poll():
    clock = FakeClock()
    sensor = FakeSensor(60)
    g = guard(sensor, clock)
    for _ in range(10):
        with g.batch():
            clock.now += 0.05                           # ten short batches inside one second
    assert sensor.calls == 1


def test_a_resume_or_soft_point_at_or_above_the_ceiling_is_refused():
    with pytest.raises(ValueError):
        ThermalGuard(FULL, FakeSensor(60), ceiling=80, resume=80)
    with pytest.raises(ValueError):
        ThermalGuard(FULL, FakeSensor(60), ceiling=80, soft=85)


def test_a_long_run_takes_one_cooling_break_per_period():
    clock = FakeClock()
    cooldown = Cooldown(Throttle(1.0, clock=clock, sleep=clock.sleep), every=3600, pause=300,
                        clock=clock, sleep=clock.sleep)
    for _ in range(26):                                 # 26 batches of 10 min: 4 h 20 min of work
        with cooldown.batch():
            clock.now += 600
    assert clock.slept == [300, 300, 300, 300]         # after each full hour, before the next batch
    assert cooldown.breaks == 4 and cooldown.paused_s == 1200


def test_a_run_shorter_than_the_period_never_breaks():
    clock = FakeClock()
    cooldown = Cooldown(FULL, every=3600, pause=300, clock=clock, sleep=clock.sleep)
    for _ in range(5):
        with cooldown.batch():
            clock.now += 600
    assert clock.slept == [] and cooldown.breaks == 0


def test_the_default_share_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert default_share() == DEFAULT_SHARE == 0.8
    monkeypatch.setenv(ENV, "0.6")
    assert default_share() == 0.6
