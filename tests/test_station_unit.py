"""The station's log from made-up run summaries: no GPU, no files."""

import pytest

from foqlens.station import BEGIN, END, SAMPLE_S, entry, render, session_record, splice

PHASE = {"samples": 100, "started": "2026-09-14T21:05", "wall_s": 120.0, "utilization_mean": 80.0,
         "temperature_mean_c": 60.0, "temperature_peak_c": 70.0, "power_peak_w": 300.0}


def test_a_flat_summary_is_one_phase_dated_by_its_start():
    e = entry({"gpu": PHASE, "pacer": {"cooling_breaks": 2}}, "reference/x/e2b", "2026-01-01")
    assert (e.date, e.run, e.seconds, e.estimated) == ("2026-09-14", "reference/x/e2b", 120.0, False)
    assert (e.utilization, e.temperature_peak, e.power_peak, e.cooling_breaks) == (80.0, 70.0, 300.0, 2)


def test_phases_add_their_time_weight_their_means_and_keep_the_highest_peak():
    other = PHASE | {"wall_s": 360.0, "utilization_mean": 40.0, "temperature_peak_c": 76.0, "power_peak_w": 250.0}
    e = entry({"gpu": {"masks": PHASE, "eval": other}}, "run", "2026-01-01")
    assert e.seconds == 480.0
    assert e.utilization == pytest.approx((80 * 120 + 40 * 360) / 480)
    assert (e.temperature_peak, e.power_peak) == (76.0, 300.0)


def test_a_summary_older_than_the_clock_counts_its_samples_and_is_marked_estimated():
    old = {"samples": 359, "utilization_mean": 58.0}
    e = entry({"gpu": old}, "reference/address-stability/e2b", "2026-09-13")
    assert (e.date, e.seconds, e.estimated) == ("2026-09-13", 359 * SAMPLE_S, True)
    assert e.temperature_peak is None and e.cooling_breaks is None


def test_a_summary_that_never_watched_the_card_is_no_entry():
    assert entry({"accuracy": 0.5}, "run", "2026-09-14") is None
    assert entry({"gpu": {"samples": 0}}, "run", "2026-09-14") is None


def test_a_record_entered_by_hand_keeps_its_own_date_name_and_estimate():
    record = {"run": "calibration", "date": "2026-09-14", "estimated": True, "note": "entered by hand",
              "gpu": {"wall_s": 1500, "temperature_peak_c": 83, "power_peak_w": 403}}
    e = entry(record, "file-stem", "2026-09-20")
    assert (e.date, e.run, e.estimated, e.note, e.utilization) == ("2026-09-14", "calibration", True, "entered by hand", None)


def test_the_log_is_in_date_order_and_the_peaks_name_their_run():
    hot = entry({"run": "hot", "date": "2026-09-14", "gpu": {"wall_s": 60, "temperature_peak_c": 83, "power_peak_w": 403}}, "", "")
    long = entry({"gpu": PHASE | {"started": "2026-09-13T10:00", "wall_s": 7200.0}}, "long", "")
    text = render([hot, long])
    log = text.split("## Peaks")[0]
    assert log.index("| 2026-09-13 | long | 2 h 00 min |") < log.index("| 2026-09-14 | hot | 1 min |")
    assert "| Temperature | 83 °C | 2026-09-14, hot |" in text and "| Longest run | 2 h 00 min | 2026-09-13, long |" in text
    assert "| GPU time in total | 2 h 01 min | 2 entries |" in text
    assert "| 2026-09-14 | hot | 1 min | - | - / 83 °C | 403 W | - |  |" in text


def test_splice_replaces_only_what_stands_between_the_markers():
    doc = f"# Title\n\nprose\n\n{BEGIN}\nold table\n{END}\n\ntail\n"
    out = splice(doc, "new table")
    assert out == f"# Title\n\nprose\n\n{BEGIN}\nnew table\n{END}\n\ntail\n"
    assert splice(out, "new table") == out
    with pytest.raises(ValueError):
        splice("# no markers", "table")


def test_a_test_session_leaves_its_files_its_monitor_and_its_outcome():
    record = session_record(PHASE, {"passed": 61, "failed": 0, "error": 1}, ["test_stand_gpu", "test_bake_gpu"])
    run = "GPU tests: test_bake_gpu, test_stand_gpu"
    assert record == {"run": run, "gpu": PHASE, "note": "61 passed, 1 error"}
    assert entry(record, "2026-09-14T21-05-00", "").run == run
