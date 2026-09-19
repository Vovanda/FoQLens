"""The small corpus's stored masks read back with their questions and the blocks' structure."""

import numpy as np

from foqlens.config import SmallCorpus
from foqlens.corpora import Row
from foqlens.small_corpus import Draw, StoredMasks, pick_shards, store, stratified_shards

ROWS = [("t", Row(f"t{i}", "q", ("a",), None)) for i in range(6)] + [("n", Row(f"n{i}", "q", ("a",), None))
                                                                     for i in range(4)]
BLOCKS = (np.array([0]), np.array(["mlp.up_proj"]), np.array([1.0]))


def test_shards_together_are_their_union_in_the_draws_order_and_name_their_files():
    small = SmallCorpus(0.05, 0.2, 50, 0.05, seed=1, shards=3)
    found = Draw(laid=ROWS, calibration=ROWS[:3])
    both, suffix = pick_shards(found, small, [3, 1])
    ones = [pick_shards(found, small, [k])[0] for k in (1, 3)]
    assert suffix == "-shards1-3of3" and pick_shards(found, small, [2])[1] == "-shard2of3"
    assert sorted(r.id for _, r in both.laid) == sorted(r.id for s in ones for _, r in s.laid)
    order = [ROWS.index(p) for p in both.laid]
    assert order == sorted(order)
    assert pick_shards(found, small, None) == (found, "")


def test_masks_are_read_by_question_and_a_glob_joins_the_shards(tmp_path):
    shards = [Draw(laid=ROWS[:4]), Draw(laid=ROWS[4:])]
    for k, s in enumerate(shards):
        masks = np.arange(len(s.laid), dtype=np.float32)[:, None] + 10 * k
        store(s, masks, *BLOCKS, {}).save(tmp_path / f"m-shard{k + 1}of2.npz")
    whole = StoredMasks.read(str(tmp_path / "m-shard*of2.npz"))
    assert len(whole.ids) == len(ROWS)
    got = whole.rows_of([("n", "n1"), ("t", "t0"), ("x", "gone")])
    assert got[:2, 0].tolist() == [13.0, 0.0] and np.isnan(got[2, 0])


def test_shards_part_the_questions_and_hold_every_corpus_share_to_within_one():
    labels = ["a"] * 23 + ["b"] * 7 + ["c"] * 50
    parts = stratified_shards(labels, 10, seed=0)
    assert sorted(np.concatenate(parts).tolist()) == list(range(80))
    for label, total in (("a", 23), ("b", 7), ("c", 50)):
        counts = [sum(labels[i] == label for i in p) for p in parts]
        assert max(counts) - min(counts) <= 1 and sum(counts) == total
    assert [p.tolist() for p in parts] == [p.tolist() for p in stratified_shards(labels, 10, seed=0)]


def test_a_draws_shard_keeps_the_draws_order_and_its_shards_concat_back():
    rows = [("t", Row(f"t{i}", "q", ("a",), None)) for i in range(6)] + [("n", Row(f"n{i}", "q", ("a",), None))
                                                                         for i in range(4)]
    found = Draw(laid=rows, calibration=rows[:3])
    shards = [found.shard(k, 3, seed=1) for k in range(3)]
    assert sorted(r.id for s in shards for _, r in s.laid) == sorted(r.id for _, r in rows)
    for s in shards:
        order = [rows.index(p) for p in s.laid]
        assert order == sorted(order)
    blocks = (np.array([0]), np.array(["mlp.up_proj"]), np.array([1.0]))
    kept = [store(s, np.full((len(s.laid) + len(s.calibration), 1), k, dtype=np.float32), *blocks, {})
            for k, s in enumerate(shards)]
    whole = StoredMasks.concat(kept)
    assert len(whole.ids) == len(rows) + 3 and whole.laid.sum() == len(rows)


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
