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
