"""The precision controller on synthetic linear layers: no model, seconds."""

import bitsandbytes.functional as bnbf
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from foqlens.precision import Controller, MixedPrecisionLinear
from foqlens.quant import NF4_BLOCKSIZE, Int8Weight, Level

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
# the fused nf4 kernel may round differently from linear() on the dequantized weight
BF16_TOL = {"atol": 2e-2, "rtol": 2e-2}


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
    assert mixed.packed_levels == ()
    mixed.set_levels(np.array([[0, 3, 0, 3], [3, 0, 3, 0]], dtype=np.uint8))
    assert mixed.packed_levels == ()  # a bf16 / ZERO bench holds no quantized copy
    mixed.set_levels(Level.NF4)
    assert mixed.packed_levels == (Level.NF4,)
    mixed.set_levels(np.array([0, 1, 2, 3], dtype=np.uint8))
    assert mixed.packed_levels == (Level.NF4, Level.INT8)


def test_int8_error_is_within_half_a_step():
    weight = make_linear().weight.data
    q = Int8Weight.quantize(weight)
    err = (q.dequantize(torch.float32) - weight.float()).abs()
    assert torch.all(err <= q.scale / 2 + 1e-6)


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
    layouts = np.array([[0, 0, 0, 0], [2, 0, 2, 0], [1, 2, 0, 2]], dtype=np.uint8)
    mixed.set_levels(layouts)
    batched = mixed(x)
    for b in range(3):
        mixed.set_levels(layouts[b])
        torch.testing.assert_close(batched[b : b + 1], mixed(x[b : b + 1]), **BF16_TOL)


def test_per_sample_layouts_reject_a_wrong_batch():
    mixed = MixedPrecisionLinear(make_linear(), block_rows=64)
    mixed.set_levels(np.array([[0, 2, 0, 2], [2, 0, 2, 0]], dtype=np.uint8))
    with pytest.raises(ValueError):
        mixed(make_input(batch=3))


def test_last_block_may_be_partial():
    mixed = MixedPrecisionLinear(make_linear(out_features=200), block_rows=64)
    assert mixed.n_blocks == 4
    assert mixed.block_sizes().tolist() == [64, 64, 64, 8]
    mixed.set_levels(np.array([0, 0, 0, 1], dtype=np.uint8))
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
    ctl.set_layout(np.array([[0, 0, 0, 0], [2, 2, 2, 2]], dtype=np.uint8))
    assert ctl.mean_bits().tolist() == [16.0, 4.0]
    ctl.set_layout(np.array([2, 0, 0, 0], dtype=np.uint8))
    assert ctl.layout() == {"layers.0.small": [2], "layers.1.big": [0, 0, 0]}
    with pytest.raises(ValueError):
        ctl.set_layout(np.zeros(5, dtype=np.uint8))


def test_layout_reports_actual_levels():
    ctl = Controller({"layers.0.a": MixedPrecisionLinear(make_linear(), block_rows=64)})
    ctl.set_blocks("layers.0.a", [0, 2], Level.NF4)
    assert ctl.layout() == {"layers.0.a": [2, 0, 2, 0]}
    with pytest.raises(KeyError):
        ctl.set_layer(5, Level.NF4)
