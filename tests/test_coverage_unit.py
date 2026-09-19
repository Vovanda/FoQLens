"""How much of the network a layout sharpens: zone shares, the share lifted, and a run's spread against the ladder."""

from dataclasses import dataclass

import numpy as np

from foqlens.coverage import level_shares, lifted_shares, question_coverage, run_coverage, spread, zone_shares
from foqlens.quant import Level

WEIGHTS = np.array([1.0, 1.0, 2.0, 4.0])  # 8 weights in all
LIFTS = np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])  # the second zone lies inside the first


@dataclass(frozen=True)
class TwoZones:
    """A zoned policy: every question has the zones of LIFTS, the first up to D8, the second up to D4."""

    name: str = "two"

    def levels(self, indices: np.ndarray) -> np.ndarray:
        row = np.where(LIFTS.max(axis=0) > 0, int(Level.D8), int(Level.D2)).astype(np.uint8)
        return np.stack([row for _ in indices])

    def zone_cover(self, index: int) -> tuple[np.ndarray, list[Level]]:
        return LIFTS, [Level.D8, Level.D4]


@dataclass(frozen=True)
class NoZones:
    name: str = "flat"

    def levels(self, indices: np.ndarray) -> np.ndarray:
        return np.full((len(indices), len(WEIGHTS)), int(Level.D2), dtype=np.uint8)


def test_a_zones_share_is_the_weight_its_lift_reaches():
    lifts = np.array([[1.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0]])
    assert zone_shares(lifts, WEIGHTS).tolist() == [0.25, 0.25]


def test_the_share_lifted_counts_overlapping_zones_once_and_lies_between_the_largest_zone_and_the_sum():
    codes = TwoZones().levels(np.arange(1))
    lifted = lifted_shares(codes, Level.D2, WEIGHTS)
    shares = zone_shares(LIFTS, WEIGHTS)
    assert lifted.tolist() == [0.5] and shares.max() <= lifted[0] <= shares.sum()


def test_every_question_gets_its_zones_shares_and_ceilings_and_a_flat_policy_its_share_alone():
    policy = TwoZones()
    rows = question_coverage(policy, policy.levels(np.arange(2)), Level.D2, WEIGHTS, np.array([10, 20]))
    assert rows[1] == {"lifted_share": 0.5, "levels": {"D2": 0.5, "D8": 0.5}, "bytes": 20,
                       "zone_shares": [0.5, 0.25], "zone_ceilings": ["D8", "D4"]}
    flat = question_coverage(NoZones(), NoZones().levels(np.arange(1)), Level.D2, WEIGHTS, np.array([5]))
    assert flat == [{"lifted_share": 0.0, "levels": {"D2": 1.0}, "bytes": 5}]


def test_the_level_shares_of_a_question_sum_to_one_by_weight():
    codes = np.array([[int(Level.ZERO), int(Level.D2), int(Level.D4), int(Level.D8)]], dtype=np.uint8)
    shares = level_shares(codes, WEIGHTS)[0]
    assert shares == {"ZERO": 0.125, "D2": 0.125, "D4": 0.25, "D8": 0.5} and sum(shares.values()) == 1.0
    run = run_coverage([{"lifted_share": 0.875, "levels": shares, "bytes": 1}], {})
    assert run["levels"]["ZERO"]["median"] == 0.125 and "D6" not in run["levels"]


def test_a_run_counts_the_questions_over_half_and_reads_bytes_against_the_ladder():
    rows = [{"lifted_share": s, "bytes": b, "zone_shares": [0.1] * n}
            for s, b, n in ((0.1, 100, 1), (0.4, 200, 2), (0.6, 300, 3))]
    found = run_coverage(rows, {"d2": 100, "d8": 400})
    assert found["over_half"] == 1 and found["zones"]["max"] == 3.0
    assert found["bytes_to_uniform"]["d2"]["median"] == 2.0 and found["bytes_to_uniform"]["d8"]["max"] == 0.75


def test_a_run_without_zones_reports_no_zone_count_and_an_empty_run_no_spread():
    assert "zones" not in run_coverage([{"lifted_share": 0.2, "bytes": 1}], {})
    assert spread(np.array([])) == {"median": None, "p90": None, "max": None}
