"""The oracles' block scores in groups, brought to one scale and laid over one another, and the bootstrap."""

import numpy as np
import pytest

from foqlens.oracle_overlay import block_group_ids, bootstrap, field, overlay, to_groups, unit_field, unit_scale


def test_a_field_on_its_own_scale_lands_in_the_unit_range_and_keeps_its_order():
    field = unit_field(np.array([-1.0, 0.0, 0.5, 2.0, np.nan]), scale=2.0)
    assert field.tolist() == [0.0, 0.0, 0.25, 1.0, 0.0]
    rising = unit_field(np.array([1.0, 3.0, 2.0]), scale=10.0)
    assert np.argsort(rising).tolist() == [0, 2, 1]
    with pytest.raises(ValueError):
        unit_field(np.array([1.0]), scale=0.0)


def test_the_scale_is_a_quantile_so_one_outlying_group_does_not_set_it_for_everybody():
    values = np.array([1.0, 1.0, 1.0, 1.0, 1000.0])
    assert unit_scale(values, share=0.5) == 1.0
    assert unit_field(values, unit_scale(values, share=0.5)).tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert unit_scale(np.array([-1.0, 0.0, np.nan])) == 1.0  # nothing positive to read a scale from


def test_agreement_puts_out_what_one_oracle_holds_low_and_the_soft_overlay_does_not():
    high, low = np.array([1.0, 1.0]), np.array([1.0, 0.01])
    assert overlay([high, low], "product").tolist() == pytest.approx([1.0, 0.1])  # the root of 1 * 0.01
    assert overlay([high, low], "mean").tolist() == pytest.approx([1.0, 0.505])
    assert overlay([high, low], "least").tolist() == pytest.approx([1.0, 0.01])
    alone = np.array([0.3, 0.7])
    for how in ("product", "mean", "least"):
        assert overlay([alone], how).tolist() == pytest.approx(alone.tolist())
    with pytest.raises(ValueError):
        overlay([alone, alone], "median")


def test_one_method_makes_a_field_of_every_oracle_whatever_it_measured():
    nats = np.array([[0.0, 0.4, 2.0], [0.1, 0.2, 0.3]])  # an oracle that swept the answer
    energy = np.array([[0.0, 4e5, 9e7], [1e3, 2e3, 3e3]])  # another, in the energy of the error
    made = [field(values) for values in (nats, energy)]
    for one in made:  # the same kind of thing, in the same layout, whatever was inside
        assert one.shape == (2, 3) and (0.0 <= one).all() and (one <= 1.0).all()
    assert np.array_equal(np.argsort(made[0], axis=1), np.argsort(made[1], axis=1))  # both keep their own order
    assert overlay(made, "product").shape == (2, 3)  # and they lay over one another as they are


def test_the_overlays_keep_their_order_and_the_unit_range():
    fields = [np.array([0.9, 0.2, 0.5]), np.array([0.4, 0.8, 0.5]), np.array([0.1, 0.6, 0.5])]
    least, product, mean = (overlay(fields, how) for how in ("least", "product", "mean"))
    assert (least <= product + 1e-12).all() and (product <= mean + 1e-12).all()
    assert all((0.0 <= f).all() and (f <= 1.0).all() for f in (least, product, mean))


def test_blocks_sum_into_their_groups_and_a_missing_score_counts_nothing():
    scores = np.array([[1.0, 2.0, np.nan, 4.0]])
    assert to_groups(scores, np.array([0, 0, 1, 1]), 2).tolist() == [[3.0, 4.0]]


def test_a_block_joins_its_layers_attention_or_the_rest_of_its_layer():
    ids = block_group_ids(np.array([0, 0, 1]), np.array(["self_attn.q_proj", "mlp.up_proj", "per_layer_projection"]),
                          ["0.attention", "0.mlp", "1.mlp"])
    assert ids.tolist() == [0, 1, 2]


def test_the_bootstrap_interval_holds_the_statistic_of_a_constant_sample():
    assert bootstrap(np.full(20, 3.0), np.mean, draws=50, seed=0, level=0.95) == (3.0, 3.0)
