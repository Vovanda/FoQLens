"""The grid of bands: names that are safe as paths, and reading a field at three bounds."""

import importlib.util
from pathlib import Path

import numpy as np

from foqlens.quant import Level

spec = importlib.util.spec_from_file_location("band_grid", Path("scripts/band_grid.py"))
band_grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(band_grid)


def test_a_band_name_holds_nothing_a_path_cannot():
    name = band_grid.band_name("lift_per_weight", (0.5, 0.667, 0.95))
    assert name == "lift_per_weight_b050067095"
    assert not set(name) & set(':/\|<>*?"')


def test_every_bound_sends_a_group_to_its_rung():
    field = np.array([[0.1, 0.5, 0.7, 0.96]])
    got = band_grid.read_band(field, (0.5, 0.7, 0.95))
    assert got.ravel().tolist() == [int(Level.D2), int(Level.D4), int(Level.D6), int(Level.D8)]


def test_a_field_below_the_first_bound_stays_at_the_base():
    assert (band_grid.read_band(np.zeros((2, 4)), (0.5, 0.7, 0.95)) == int(Level.D2)).all()


def test_named_scales_are_three_rising_bounds_and_may_switch_the_top_rung_off():
    import pytest

    assert band_grid.named_bands(["0.50,0.95,1.00", "0.35,0.60,0.88"]) == [(0.5, 0.95, 1.0), (0.35, 0.6, 0.88)]
    with pytest.raises(ValueError):
        band_grid.named_bands(["0.95,0.50,1.00"])
    with pytest.raises(ValueError):
        band_grid.named_bands(["0.50,0.95"])
