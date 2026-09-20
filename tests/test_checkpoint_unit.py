"""The partial state of a long pass: read back bit for bit for the same plan, refused for another, gone when cleared."""

import numpy as np
import pytest

from foqlens.io import Checkpoint, plan_of, save_npz_atomic


def test_a_checkpoint_reads_back_for_its_plan_only(tmp_path):
    path = tmp_path / "pass.partial.npz"
    masks = np.random.default_rng(0).standard_normal((4, 3)).astype(np.float32)
    Checkpoint(path, plan_of({"a": 1}, [[0, 1], [2]])).save(masks=masks, done=2)
    kept = Checkpoint(path, plan_of({"a": 1}, [[0, 1], [2]])).load()
    assert np.array_equal(kept["masks"], masks) and int(kept["done"]) == 2
    with pytest.raises(ValueError):
        Checkpoint(path, plan_of({"a": 2}, [[0, 1], [2]])).load()
    Checkpoint(path, "any").clear()
    assert Checkpoint(path, "any").load() is None and not path.exists()


def test_an_atomic_save_leaves_no_temporary_file_and_replaces_the_old(tmp_path):
    path = tmp_path / "out.npz"
    save_npz_atomic(path, x=np.arange(3))
    save_npz_atomic(path, x=np.arange(5))
    with np.load(path) as z:
        assert z["x"].tolist() == [0, 1, 2, 3, 4]
    assert [p.name for p in tmp_path.iterdir()] == ["out.npz"]
