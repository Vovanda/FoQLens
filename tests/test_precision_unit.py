"""The precision controller on synthetic linear layers: no model, seconds."""

import bitsandbytes.functional as bnbf
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from functools import partial

from foqlens.refinements import KQuantLadder
from foqlens.precision import Controller, MixedPrecisionLinear, samples
from foqlens.quant import MAX_DEPTH, NF4_BLOCKSIZE, SCALE_GROUP, Int8Weight, Level, Nf4Weight, RefinedWeight

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
# the fused nf4 kernel may round differently from linear() on the dequantized weight
BF16_TOL = {"atol": 2e-2, "rtol": 2e-2}


B, N, Z = Level.BF16, Level.NF4, Level.ZERO


def codes(*rows) -> np.ndarray:
    """A layout by level names: one row of levels, or several rows for per-sample layouts."""
    return np.array(rows if isinstance(rows[0], list) else list(rows), dtype=np.uint8)


def make_linear(out_features: int = 200, in_features: int = 256, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=False, device=DEVICE, dtype=torch.bfloat16)
    nn.init.normal_(linear.weight, std=0.02)
    return linear


def make_input(batch: int = 3, in_features: int = 256) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, 5, in_features, device=DEVICE, dtype=torch.bfloat16)


def test_bf16_is_bit_exact_with_original_linear():
    linear = make_linear()
    x = make_input()
    expected = linear(x)
    assert torch.equal(MixedPrecisionLinear(linear, block_rows=64)(x), expected)


def test_nf4_output_matches_the_bnb_dequantized_weight():
    linear = make_linear()
    x = make_input()
    packed, state = bnbf.quantize_4bit(linear.weight.data.contiguous(), blocksize=NF4_BLOCKSIZE, quant_type="nf4")
    expected = F.linear(x, bnbf.dequantize_4bit(packed, state).to(torch.bfloat16))
    mixed = MixedPrecisionLinear(linear, block_rows=64)
    mixed.set_levels(Level.NF4)
    torch.testing.assert_close(mixed(x), expected, **BF16_TOL)
    assert not torch.equal(mixed(x), linear(x))


def test_zero_level_removes_the_block_and_costs_no_bits():
    mixed = MixedPrecisionLinear(make_linear(out_features=200), block_rows=64)
    x = make_input()
    before = mixed(x)
    levels = mixed.levels
    levels[1] = Level.ZERO
    mixed.set_levels(levels)
    after = mixed(x)
    assert torch.equal(after[..., 64:128], torch.zeros_like(after[..., 64:128]))
    assert torch.equal(after[..., :64], before[..., :64]) and torch.equal(after[..., 128:], before[..., 128:])
    ctl = Controller({"layers.0.a": mixed})
    ctl.set_all(Level.ZERO)
    assert ctl.mean_bits() == 0.0 and torch.equal(mixed(x), torch.zeros_like(before))


def test_packed_copies_exist_only_for_levels_in_use():
    mixed = MixedPrecisionLinear(make_linear(), block_rows=64)
    assert mixed.storages == ()
    mixed.set_levels(codes([B, Z, B, Z], [Z, B, Z, B]))
    assert mixed.storages == ()  # a bf16 / ZERO bench holds no quantized copy
    mixed.set_levels(Level.NF4)
    assert mixed.storages == (Nf4Weight,)
    mixed.set_levels(codes(B, Level.INT8, Level.NF4, Z))
    assert mixed.storages == (Nf4Weight, Int8Weight)
    mixed.set_levels(np.array([Level.D2, Level.D8, Level.D4, Level.ZERO], dtype=np.uint8))
    assert mixed.storages == (Nf4Weight, Int8Weight, RefinedWeight)  # one refined copy serves every depth


def test_the_error_shrinks_by_the_step_of_each_depth():
    weight = make_linear().weight.data
    full = RefinedWeight.quantize(weight)
    groups = weight.float().view(weight.shape[0], -1, SCALE_GROUP)
    for depth in range(1, MAX_DEPTH + 1):
        err = (full.dequantize(torch.float32, depth).view_as(groups) - groups).abs()
        bound = full.scale / 2 / 4 ** (depth - 1)
        assert torch.all(err <= bound * (1 + 1e-4)), depth


def test_a_depth_reads_only_its_own_codes_and_costs_one_byte_per_weight():
    weight = make_linear().weight.data
    full = RefinedWeight.quantize(weight)
    shallow = RefinedWeight(packed=full.packed[:2].clone(), scale=full.scale)
    assert torch.equal(shallow.dequantize(torch.float32, 2), full.dequantize(torch.float32, 2))
    assert full.packed.dtype == torch.uint8 and full.packed.numel() == weight.numel()
    assert full.scale.numel() == weight.numel() // SCALE_GROUP


def test_depth_levels_switch_only_their_block_and_cost_their_bits():
    mixed = MixedPrecisionLinear(make_linear(out_features=200), block_rows=64)
    x = make_input()
    before = mixed(x)
    mixed.set_levels(codes(B, Level.D4, B, B))
    after = mixed(x)
    assert torch.equal(after[..., :64], before[..., :64]) and torch.equal(after[..., 128:], before[..., 128:])
    assert not torch.equal(after[..., 64:128], before[..., 64:128])
    ctl = Controller({"layers.0.a": mixed})
    for level, bits in ((Level.D2, 2), (Level.D4, 4), (Level.D6, 6), (Level.D8, 8)):
        ctl.set_all(level)
        assert ctl.mean_bits() == bits


def test_drop_bf16_keeps_the_depth_outputs_and_refuses_the_dropped_levels():
    mixed = MixedPrecisionLinear(make_linear(out_features=200), block_rows=64)
    x = make_input()
    layout = np.array([Level.D8, Level.D4, Level.ZERO, Level.D2], dtype=np.uint8)
    mixed.set_levels(Level.NF4)  # a copy the resident module must let go
    mixed.set_levels(layout)
    before = mixed(x)
    mixed.drop_bf16()
    assert torch.equal(mixed(x), before)
    assert mixed.weight is None and mixed.storages == (RefinedWeight,)
    for level in (Level.BF16, Level.INT8, Level.NF4):
        with pytest.raises(ValueError):
            mixed.set_levels(level)
    mixed.set_levels(Level.D6)
    assert mixed(x).shape == before.shape


def test_a_resident_module_on_a_meta_weight_reads_as_one_built_from_bf16():
    linear = make_linear(out_features=200)
    built = MixedPrecisionLinear(linear, block_rows=64)
    layout = np.array([Level.D8, Level.D4, Level.ZERO, Level.D2], dtype=np.uint8)
    built.set_levels(layout)
    copy = built._packed[RefinedWeight]
    x = make_input()
    expected = built(x)
    with torch.device("meta"):
        empty = nn.Linear(linear.in_features, linear.out_features, bias=False, dtype=torch.bfloat16)
    resident = MixedPrecisionLinear.resident(empty, copy, DEVICE, block_rows=64)
    assert resident.weight is None and resident.storages == (RefinedWeight,)
    with pytest.raises(ValueError):
        resident.set_levels(Level.BF16)
    resident.set_levels(layout)
    assert torch.equal(resident(x), expected)


@pytest.mark.parametrize("level", [Level.D2, Level.D4, Level.D6, Level.D8])
def test_a_baked_level_reads_as_the_refined_depth_and_refuses_bf16(level):
    mixed = MixedPrecisionLinear(make_linear(out_features=200), block_rows=64)
    x = make_input()
    mixed.set_levels(level)
    unpacked = mixed(x)
    mixed.bake(level)
    assert torch.equal(mixed(x), unpacked) and mixed.storages == ()
    mixed.set_levels(level)  # the judge of a run sets the level before every batch
    assert torch.equal(mixed(x), unpacked) and mixed.storages == ()
    for other in (Level.BF16, Level.D2 if level is not Level.D2 else Level.D4):
        with pytest.raises(ValueError):
            mixed.set_levels(other)
    with pytest.raises(ValueError):
        mixed.bake(level)


def test_a_weight_baked_from_another_source_is_read_as_given_and_never_quantized_again():
    linear = make_linear(out_features=200)
    given = linear.weight.data.flip(0).clone()  # any weight other than the module's own
    x = make_input()
    ctl = Controller({"layers.0.a": MixedPrecisionLinear(linear, block_rows=64)})

    class Given:
        def read(self, name, weight, level):
            return given

    ctl.bake_from(Given(), Level.D2)
    mixed = ctl.modules["layers.0.a"]
    expected = F.linear(x, given)
    assert torch.equal(mixed(x), expected) and mixed.storages == ()
    ctl.set_all(Level.D2)  # what Asking.answer does before every batch
    assert torch.equal(mixed(x), expected) and mixed.storages == ()
    with pytest.raises(ValueError):
        mixed.bake_weight(given[:10], Level.D2)


def test_only_a_read_depth_is_baked():
    with pytest.raises(ValueError):
        MixedPrecisionLinear(make_linear(), block_rows=64).bake(Level.NF4)


def test_depth_caps_keep_the_reads_within_them_and_free_the_deeper_depths():
    mixed = MixedPrecisionLinear(make_linear(out_features=200, in_features=256), block_rows=64)
    x = make_input()
    layout = np.array([Level.D8, Level.D4, Level.D2, Level.ZERO], dtype=np.uint8)
    mixed.set_levels(layout)
    before = mixed(x)
    mixed.drop_bf16()
    mixed.set_caps(np.array([4, 2, 1, 0]))
    assert torch.equal(mixed(x), before)
    # depth 0 for blocks 0-2 (192 rows), depth 1 for blocks 0-1, depths 2 and 3 for block 0; 64 bytes a row
    assert mixed.stored_bytes() == (192 + 128 + 64 + 64) * 256 // 4
    with pytest.raises(ValueError):
        mixed.set_levels(np.array([Level.D8, Level.D6, Level.D2, Level.ZERO], dtype=np.uint8))  # block 1 is capped at D4
    ctl = Controller({"layers.0.a": mixed})
    expected = (64 * 8 + 64 * 4 + 64 * 2 + 8 * 0) / 200
    assert ctl.stored_bits() == pytest.approx(expected)


def test_reading_several_depths_at_once_is_exact_against_separate_reads():
    import torch.nn.functional as F

    from foqlens.quant import CappedRefinedWeight

    linear = make_linear(out_features=192)
    x = make_input()
    full = RefinedWeight.quantize(linear.weight.data)
    capped = CappedRefinedWeight.from_full(full, torch.tensor([4, 4, 4], device=DEVICE), 64)
    for store in (full, capped):
        together = store.linear_at_depths(x, None, [2, 3, 4])
        separate = [F.linear(x, store.dequantize(x.dtype, d)) for d in (2, 3, 4)]
        assert all(torch.equal(a, b) for a, b in zip(together, separate)), type(store).__name__


def test_unpacking_all_depths_at_once_sums_exactly_as_depth_by_depth():
    from foqlens.quant import _DepthReader

    full = RefinedWeight.quantize(make_linear(out_features=192).weight.data)
    for depths in ([1, 2, 3, 4], [2, 4], [3], [4], [1]):
        fast = [w.clone() for w in full._sums(depths)]
        reference = [w.clone() for w in _DepthReader._sums(full, depths)]  # the depth-by-depth path
        assert all(torch.equal(a, b) for a, b in zip(fast, reference, strict=True)), depths


def test_caps_need_a_resident_module():
    mixed = MixedPrecisionLinear(make_linear(), block_rows=64)
    with pytest.raises(ValueError):
        mixed.set_caps(np.array([4, 4, 4, 4]))


def test_int8_error_is_within_half_a_step():
    weight = make_linear().weight.data
    q = Int8Weight.quantize(weight)
    err = (q.dequantize(torch.float32) - weight.float()).abs()
    assert torch.all(err <= q.scale / 2 + 1e-6)


@pytest.mark.parametrize("name", ["layers.0.self_attn.q_proj", "layers.0.mlp.down_proj"])
def test_a_k_quant_copy_is_read_by_blocks_each_at_its_own_depth(name):
    linear = make_linear(out_features=200)
    ladder = KQuantLadder()
    copy = ladder.quantize(name, linear.weight.data)
    mixed = MixedPrecisionLinear(linear, block_rows=64, copy=partial(ladder.quantize, name))
    depths = [Level.D2, Level.D4, Level.D6, Level.D8]
    mixed.set_levels(np.array([int(lv) for lv in depths], dtype=np.uint8))
    x = make_input()
    out = mixed(x)
    for block, level in enumerate(depths):
        rows = slice(64 * block, min(64 * (block + 1), 200))
        expected = F.linear(x, copy.dequantize(x.dtype, level.depth))
        assert torch.equal(out[..., rows], expected[..., rows]), level


def test_switching_one_block_changes_only_its_rows():
    mixed = MixedPrecisionLinear(make_linear(out_features=200), block_rows=64)
    x = make_input()
    before = mixed(x)
    levels = mixed.levels
    levels[1] = Level.NF4  # rows 64..127
    mixed.set_levels(levels)
    after = mixed(x)
    assert torch.equal(after[..., :64], before[..., :64])
    assert torch.equal(after[..., 128:], before[..., 128:])
    assert not torch.equal(after[..., 64:128], before[..., 64:128])


def test_per_sample_layouts_match_running_each_sample_alone():
    mixed = MixedPrecisionLinear(make_linear(out_features=200), block_rows=64)
    x = make_input(batch=3)
    layouts = codes([B, B, B, B], [N, B, N, B], [Level.INT8, N, B, N])
    mixed.set_levels(layouts)
    batched = mixed(x)
    for b in range(3):
        mixed.set_levels(layouts[b])
        torch.testing.assert_close(batched[b : b + 1], mixed(x[b : b + 1]), **BF16_TOL)


def test_per_sample_layouts_reject_a_wrong_batch():
    mixed = MixedPrecisionLinear(make_linear(), block_rows=64)
    mixed.set_levels(codes([B, N, B, N], [N, B, N, B]))
    with pytest.raises(ValueError):
        mixed(make_input(batch=3))


def test_a_part_of_the_batch_reads_its_own_samples_layouts():
    linear = make_linear(out_features=200)
    model = nn.Sequential(MixedPrecisionLinear(linear, block_rows=64))
    x = make_input(batch=4)
    layouts = np.array([[0, 0, 0, 0], [2, 0, 2, 0], [1, 2, 0, 2], [0, 1, 1, 0]], dtype=np.uint8)
    part = slice(1, 3)
    model[0].set_levels(layouts[part])
    alone = model(x[part])
    model[0].set_levels(layouts)
    with samples(model, part):
        assert torch.equal(model(x[part]), alone)
        with pytest.raises(ValueError):
            model(x)
    assert model(x).shape[0] == 4


def test_last_block_may_be_partial():
    mixed = MixedPrecisionLinear(make_linear(out_features=200), block_rows=64)
    assert mixed.n_blocks == 4
    assert mixed.block_sizes().tolist() == [64, 64, 64, 8]
    mixed.set_levels(codes(B, B, B, Level.INT8))
    assert mixed(make_input()).shape[-1] == 200


def test_mean_bits_is_weighted_by_weight_count():
    small = MixedPrecisionLinear(make_linear(out_features=64, in_features=256), block_rows=64)
    big = MixedPrecisionLinear(make_linear(out_features=192, in_features=256), block_rows=64)
    ctl = Controller({"layers.0.small": small, "layers.1.big": big})
    assert ctl.mean_bits() == 16.0
    ctl.set_all(Level.NF4)
    assert ctl.mean_bits() == 4.0
    ctl.set_all(Level.BF16)
    ctl.set_module("layers.1.big", Level.NF4)
    # 64 rows at 16 bits and 192 rows at 4 bits
    assert ctl.mean_bits() == pytest.approx((64 * 16 + 192 * 4) / 256)
    ctl.set_blocks("layers.1.big", [0], Level.INT8)
    assert ctl.mean_bits() == pytest.approx((64 * 16 + 64 * 8 + 128 * 4) / 256)


def test_layout_across_modules_and_per_sample_mean_bits():
    small = MixedPrecisionLinear(make_linear(out_features=64), block_rows=64)
    big = MixedPrecisionLinear(make_linear(out_features=192), block_rows=64)
    ctl = Controller({"layers.0.small": small, "layers.1.big": big})
    assert ctl.n_blocks == 4
    ctl.set_layout(codes([B, B, B, B], [N, N, N, N]))
    assert ctl.mean_bits().tolist() == [16.0, 4.0]
    ctl.set_layout(codes(N, B, B, B))
    assert ctl.layout() == {"layers.0.small": [int(N)], "layers.1.big": [int(B)] * 3}
    with pytest.raises(ValueError):
        ctl.set_layout(np.zeros(5, dtype=np.uint8))


def test_layout_reports_actual_levels():
    ctl = Controller({"layers.0.a": MixedPrecisionLinear(make_linear(), block_rows=64)})
    ctl.set_blocks("layers.0.a", [0, 2], Level.NF4)
    assert ctl.layout() == {"layers.0.a": [int(N), int(B), int(N), int(B)]}
    with pytest.raises(KeyError):
        ctl.set_layer(5, Level.NF4)


def test_a_resident_module_on_a_copy_cut_short_refuses_the_depths_it_does_not_hold():
    linear = make_linear(out_features=200)
    copy = RefinedWeight.quantize(linear.weight.data)
    short = RefinedWeight(packed=copy.packed[:2].clone(), scale=copy.scale)
    resident = MixedPrecisionLinear.resident(linear, short, DEVICE, block_rows=64)
    assert resident.levels.tolist() == [Level.D4] * resident.n_blocks
    for level in (Level.D6, Level.D8):
        with pytest.raises(ValueError):
            resident.set_levels(level)
    resident.set_levels(Level.D2)
