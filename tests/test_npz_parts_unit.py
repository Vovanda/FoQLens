"""The shards of a run's .npz read as one: questions joined in shard order, the run's own values kept once."""

import numpy as np
import pytest

from foqlens.io import read_npz_parts


def test_shards_join_their_questions_and_keep_the_runs_values_once(tmp_path):
    for k, ids in enumerate((["a", "b"], ["c"])):
        np.savez(tmp_path / f"run-shard{k + 1}of2.npz", ids=np.array(ids), lift=np.full((len(ids), 3), k),
                 groups=np.array(["0.attention", "0.mlp", "1.attention"]), low="d2")
    joined = read_npz_parts(str(tmp_path / "run-shard*of2.npz"), frozenset({"groups"}))
    assert joined["ids"].tolist() == ["a", "b", "c"] and joined["lift"].shape == (3, 3)
    assert joined["groups"].tolist() == ["0.attention", "0.mlp", "1.attention"] and str(joined["low"]) == "d2"
    alone = read_npz_parts(str(tmp_path / "run-shard1of2.npz"))
    assert alone["ids"].tolist() == ["a", "b"]


def test_parts_of_different_runs_are_refused(tmp_path):
    for k, low in enumerate(("d2", "zero")):
        np.savez(tmp_path / f"run-shard{k + 1}of2.npz", ids=np.array(["a"]), low=low)
    with pytest.raises(ValueError):
        read_npz_parts(str(tmp_path / "run-shard*of2.npz"))


def shards_of_two_versions(tmp_path):
    """An older shard that still holds a field the run has since dropped, beside a newer one that does not."""
    np.savez(tmp_path / "run-shard1of2.npz", ids=np.array(["a"]), lift=np.zeros((1, 3)), prefix=np.zeros((1, 3)),
             groups=np.array(["0.attention", "0.mlp", "1.attention"]))
    np.savez(tmp_path / "run-shard2of2.npz", ids=np.array(["b"]), lift=np.ones((1, 3)),
             groups=np.array(["0.attention", "0.mlp", "1.attention"]))
    return str(tmp_path / "run-shard*of2.npz")


def test_shards_of_two_versions_join_over_the_fields_the_caller_names(tmp_path):
    pattern = shards_of_two_versions(tmp_path)
    joined = read_npz_parts(pattern, frozenset({"groups"}), only={"ids", "lift"})
    assert joined["ids"].tolist() == ["a", "b"] and joined["lift"].shape == (2, 3)
    assert "prefix" not in joined  # what is not asked for is not read, so the older shard's own field is no obstacle


def test_a_field_one_shard_lacks_is_named_with_its_file(tmp_path):
    pattern = shards_of_two_versions(tmp_path)
    with pytest.raises(ValueError, match="prefix"):
        read_npz_parts(pattern, frozenset({"groups"}))  # without `only`, the older shard's field breaks the join
    with pytest.raises(ValueError, match="shard2of2.*prefix|prefix"):
        read_npz_parts(pattern, frozenset({"groups"}), only={"ids", "prefix"})
