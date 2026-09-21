"""The oracles' block scores in groups, brought to one scale and laid over one another, and the bootstrap."""

import numpy as np
import pytest

from foqlens.oracle_overlay import (
    Kept,
    Oracle,
    block_group_ids,
    bootstrap,
    overlay,
    to_groups,
)


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


def test_every_oracle_answers_the_same_contract_whatever_it_measured():
    # the same two questions, measured by two oracles in units that share nothing, each against its own threshold
    nats = Kept("swept", np.array([[0.0, 0.4, 2.0], [0.1, 0.2, 0.3]]), np.array([0.5, 0.2]))
    energy = Kept("energy", np.array([[0.0, 4e5, 9e7], [1e3, 2e3, 3e3]]), np.array([1e6, 2e3]))
    assert isinstance(nats, Oracle) and isinstance(energy, Oracle)
    made = [oracle.field() for oracle in (nats, energy)]
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


def test_every_oracle_s_map_comes_back_out_of_its_own_field():
    """The field is the one every oracle is compared on, so every oracle's map must read back out of it."""
    import numpy as np
    from pathlib import Path

    from foqlens.maps import read_map
    from foqlens.oracle_overlay import demand
    from foqlens.quant import Level

    kept = np.load(Path("runs/E006-oracle-masks-that-hold/precision-fields-bartowski-Q2_K-small-corpus.npz"),
                   allow_pickle=True)
    codes = [int(Level[str(r)]) for r in kept["rungs"]]
    for source, least in (("lift_per_weight", 0.99), ("pooled", 0.98), ("drop", 0.94)):
        got = read_map(demand(kept[f"field_{source}"], kept[f"eps_{source}"]), kept["ratios"], 1.0, codes,
                       int(Level.D8))
        assert (got == kept[f"levels_{source}"]).mean() >= least, source
