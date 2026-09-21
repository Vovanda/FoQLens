"""Building the block of maps: the network's part, the question's part, and a block that rebuilds itself."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("build_maps", Path("scripts/build_maps.py"))
build_maps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_maps)


def test_the_network_part_is_what_every_question_shares():
    field = np.array([[0.2, 0.8], [0.4, 0.6]])
    assert np.allclose(build_maps.network_part(field), [0.3, 0.7])


def test_the_question_part_is_what_is_left_and_stays_in_the_unit_range():
    field = np.array([[0.2, 0.9], [0.4, 0.5]])
    got = build_maps.question_part(field)
    assert got.min() >= 0.0 and got.max() <= 1.0
    assert got[1, 0] > got[0, 0]  # the question standing above the shared part keeps the lead


def test_a_field_every_question_agrees_on_leaves_no_question_part():
    field = np.tile([0.3, 0.7], (5, 1))
    assert (build_maps.question_part(field) == 0.0).all()
