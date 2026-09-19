"""Is a mask source an address: identification across two readings and where the excess lives."""

import numpy as np
import pytest

from foqlens.address import (adaptive_depth, agreement, bag_of_tokens, deep_address, depth_cap, excess,
                             identification, layer_profile, per_question, rank_auc, silhouette_stop, stop_policy)


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


def test_two_sources_over_different_blocks_agree_when_they_place_the_questions_alike():
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(80, 10))  # what the questions are
    first = latent @ rng.normal(size=(10, 200))  # two readings of it over different blocks
    second = latent @ rng.normal(size=(10, 300))
    assert agreement(first, second) > 0.8
    assert abs(agreement(first, rng.normal(size=(80, 300)))) < 0.1


def test_early_layers_that_carry_the_question_predict_its_deep_address_and_noise_does_not():
    rng = np.random.default_rng(3)
    layers = np.repeat(np.arange(4), 50)  # four layers of 50 blocks
    weights = np.ones(len(layers))
    latent = rng.normal(size=(500, 8))
    masks = latent @ rng.normal(size=(8, len(layers))) + rng.normal(size=(500, len(layers))) * 0.1
    train, test = masks[:400], masks[400:]
    found = deep_address(train, train, test, test, layers, weights, depth=1, ridge=1e-3)
    assert found["identified"] > 0.9 and found["zoned_weight_share"] == 0.75
    noise = rng.normal(size=masks.shape)
    lost = deep_address(noise[:400], train, noise[400:], test, layers, weights, depth=1, ridge=1e-3)
    assert lost["identified"] < 0.1


def test_a_window_reads_only_its_layers_and_leaves_the_same_deep_part():
    rng = np.random.default_rng(4)
    layers = np.repeat(np.arange(4), 50)
    weights = np.ones(len(layers))
    latent = rng.normal(size=(500, 8))
    masks = latent @ rng.normal(size=(8, len(layers))) + rng.normal(size=(500, len(layers))) * 0.1
    blind = masks.copy()
    blind[:, layers == 0] = rng.normal(size=(500, 50))  # layer 0 carries nothing: a window that skips it loses nothing
    window = deep_address(blind[:400], masks[:400], blind[400:], masks[400:], layers, weights, depth=2, ridge=1e-3,
                          start=1)
    assert window["identified"] > 0.9 and window["zoned_weight_share"] == 0.5


def test_a_question_is_hit_when_its_own_is_nearest_and_its_margin_says_by_how_much():
    actual = np.eye(4) * 10.0
    predicted = actual.copy()
    predicted[3] = actual[2]  # question 3 predicted as question 2
    verdict = per_question(predicted, actual)
    assert verdict["hit"].tolist() == [True, True, True, False]
    assert verdict["margin"][3] < 0 < verdict["margin"][0]


def test_the_rank_auc_is_one_when_every_positive_is_above_and_half_without_signal():
    assert rank_auc(np.array([3.0, 4.0, 1.0, 2.0]), np.array([True, True, False, False])) == 1.0
    assert rank_auc(np.array([1.0, 1.0]), np.array([True, False])) == 0.5
    assert np.isnan(rank_auc(np.array([1.0]), np.array([True])))


def test_the_policy_spans_from_the_shallow_depth_to_the_deep_one():
    rng = np.random.default_rng(5)
    actual = rng.normal(size=(60, 40))
    shallow = actual + rng.normal(size=actual.shape) * 3.0  # noisy: many misses
    deep = actual + rng.normal(size=actual.shape) * 0.1  # almost exact
    found = adaptive_depth({4: shallow, 8: deep}, actual, low=4, high=8)
    assert found["hits"][8] > found["hits"][4]
    depths = [p["mean_depth"] for p in found["policy"]]
    assert depths[0] == 4 and depths[-1] == pytest.approx(4 + 4 * (59 / 60))  # the top quantile keeps one question
    assert found["only_high"] + found["hit_low_and_high"] == round(found["hits"][8] * 60)


def test_the_drift_between_depths_is_read_on_the_excess_not_drowned_by_a_shared_mean():
    rng = np.random.default_rng(6)
    actual = rng.normal(size=(50, 30))
    shared = np.full(30, 1000.0)  # a mean address far louder than any question
    before, after = shared + rng.normal(size=actual.shape), shared + rng.normal(size=actual.shape)
    found = adaptive_depth({4: before, 5: after}, actual, low=4, high=5)
    assert found["moved"][5] > 0.5  # two unrelated predictions move a lot, whatever mean they share


def silhouettes(changes_until: list[int], depths: range, blocks: int = 40, k: int = 4) -> dict[int, np.ndarray]:
    """Predictions whose top k blocks move at every depth up to changes_until[q] and hold after it."""
    predicted = {}
    for d in depths:
        rows = np.zeros((len(changes_until), blocks))
        for q, last in enumerate(changes_until):
            start = (min(d, last) * k + q * 7) % (blocks - k)  # a new silhouette every depth until `last`
            rows[q, start : start + k] = 10.0
        predicted[d] = rows
    return predicted


def test_a_settled_silhouette_stops_early_a_moving_one_later_and_a_restless_one_at_the_cap():
    depths = range(3, 13)
    predicted = silhouettes([6, 8, 20], depths)  # settles at 6, at 8, never
    stops = silhouette_stop(predicted, k=4, tolerance=0.1, patience=2, cap=12)
    assert stops.tolist() == [6, 8, 10]  # the restless one: 10 + patience 2 = the cap of 12
    assert (stops + 2 <= 12).all()


def test_the_cap_is_a_share_of_the_network_and_refuses_a_whole_or_empty_one():
    assert depth_cap(35, 0.3) == 10 and depth_cap(42, 0.3) == 12
    with pytest.raises(ValueError):
        depth_cap(35, 1.0)


def test_the_policy_reports_identification_at_each_questions_own_stop():
    actual = np.eye(3) * 5.0
    predicted = {4: np.eye(3)[[1, 0, 2]] * 5.0, 5: actual.copy()}  # at 4 two questions are swapped, at 5 all right
    found = stop_policy(predicted, actual, np.array([5, 5, 4]), patience=1)
    assert found["identified"] == 1.0 and found["mean_read"] == pytest.approx(14 / 3 + 1)
    assert found["stops"] == {4: 1, 5: 2}


def test_the_bag_counts_every_token_once_per_occurrence_over_a_shared_vocabulary():
    bag = bag_of_tokens([[5, 7, 7], [7, 9]])
    assert bag.shape == (2, 3)  # tokens 5, 7, 9
    assert bag.tolist() == [[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]]


def test_the_excess_is_against_the_same_reading_and_the_shapes_must_match():
    masks = np.arange(12.0).reshape(3, 4)
    assert np.allclose(excess(masks).mean(axis=0), 0)
    with pytest.raises(ValueError):
        identification(masks, masks[:2])
