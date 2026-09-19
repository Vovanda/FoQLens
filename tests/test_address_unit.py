"""Is a mask source an address: identification across two readings and where the excess lives."""

import numpy as np
import pytest

from foqlens.address import excess, identification, layer_profile


def test_masks_identify_themselves_and_noise_identifies_nothing_beyond_chance():
    rng = np.random.default_rng(0)
    masks = rng.random((200, 500))
    assert identification(masks, masks)["identified"] == 1.0
    noise = identification(masks, rng.random((200, 500)))
    assert noise["chance"] == 1 / 200
    assert noise["identified"] < 0.05


def test_a_shared_wrapper_drops_out_and_the_question_is_found_through_it():
    rng = np.random.default_rng(1)
    questions = rng.normal(size=(100, 300))
    wrapper_a, wrapper_b = rng.normal(size=300) * 50, rng.normal(size=300) * 50  # far louder than any question
    found = identification(questions + wrapper_a, questions + wrapper_b + rng.normal(size=(100, 300)) * 0.3)
    assert found["identified"] > 0.95
    assert found["own_cos"] > found["other_cos"]


def test_the_layer_profile_sums_to_one_and_follows_the_excess():
    layers = np.repeat(np.arange(3), 4)
    masks = np.zeros((2, 12))
    masks[0, :4] = 1.0  # question 0 stands out in layer 0, question 1 in layer 2
    masks[1, 8:] = 1.0
    profile = layer_profile(masks, layers)
    assert sum(profile.values()) == pytest.approx(1.0)
    assert profile[0] == pytest.approx(0.5) and profile[2] == pytest.approx(0.5) and profile[1] == 0.0


def test_the_excess_is_against_the_same_reading_and_the_shapes_must_match():
    masks = np.arange(12.0).reshape(3, 4)
    assert np.allclose(excess(masks).mean(axis=0), 0)
    with pytest.raises(ValueError):
        identification(masks, masks[:2])
