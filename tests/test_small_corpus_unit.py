"""The small corpus's stored masks read back with their questions and the blocks' structure."""

import numpy as np

from foqlens.corpora import Row
from foqlens.small_corpus import Draw, StoredMasks, store


def test_stored_masks_read_back_bit_for_bit_with_their_questions_and_blocks(tmp_path):
    found = Draw(laid=[("triviaqa", Row("t1", "q1", ("a",), None)), ("nq_open", Row("n9", "q2", ("b",), None))],
                 calibration=[("triviaqa", Row("t7", "q3", ("c",), None))], unknown={("nq_open", "n9")})
    masks = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    kept = store(found, masks, np.array([0, 0, 1, 1]), np.array(["mlp.up_proj"] * 2 + ["self_attn.q_proj"] * 2),
                 np.array([8.0, 8.0, 4.0, 4.0]), {"source": "hybrid", "level": "BF16"})
    kept.save(tmp_path / "m.npz")
    back = StoredMasks.load(tmp_path / "m.npz")
    assert np.array_equal(back.masks, masks) and back.ids.tolist() == ["t1", "n9", "t7"]
    assert back.laid.tolist() == [True, True, False] and back.unknown.tolist() == [False, True, False]
    assert back.block_kind.tolist()[2] == "self_attn.q_proj" and back.meta == {"source": "hybrid", "level": "BF16"}
