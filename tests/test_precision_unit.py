"""The precision controller on a synthetic linear layer: no model, seconds."""

import bitsandbytes.functional as bnbf
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from foqlens.precision import Controller, MixedPrecisionLinear
from foqlens.quant import NF4_BLOCKSIZE, Int8Weight, Level

pytestmark = pytest.mark.gpu

DEVICE = "cuda"


def make_linear(out_features: int = 200, in_features: int = 256, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=False, device=DEVICE, dtype=torch.bfloat16)
    nn.init.normal_(linear.weight, std=0.02)
    return linear


def make_input(in_features: int = 256) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(3, 5, in_features, device=DEVICE, dtype=torch.bfloat16)


def test_bf16_is_bit_exact_with_original_linear():
    linear = make_linear()
    x = make_input()
    expected = linear(x)
    mixed = MixedPrecisionLinear(linear, block_rows=64)
    assert torch.equal(mixed(x), expected)


def test_nf4_output_is_the_bnb_dequantized_weight():
    linear = make_linear()
    x = make_input()
    packed, state = bnbf.quantize_4bit(linear.weight.data.contiguous(), blocksize=NF4_BLOCKSIZE, quant_type="nf4")
    expected = F.linear(x, bnbf.dequantize_4bit(packed, state).to(torch.bfloat16))
    mixed = MixedPrecisionLinear(linear, block_rows=64)
    mixed.levels.fill_(int(Level.NF4))
    assert torch.equal(mixed(x), expected)
    assert not torch.equal(mixed(x), linear(x))


def test_int8_error_is_within_half_a_step():
    weight = make_linear().weight.data
    q = Int8Weight.quantize(weight)
    err = (q.dequantize(torch.float32) - weight.float()).abs()
    assert torch.all(err <= q.scale / 2 + 1e-6)


def test_switching_one_block_changes_only_its_rows():
    linear = make_linear(out_features=200)
    x = make_input()
    mixed = MixedPrecisionLinear(linear, block_rows=64)
    before = mixed(x)
    mixed.levels[1] = int(Level.NF4)  # rows 64..127
    after = mixed(x)
    assert torch.equal(after[..., :64], before[..., :64])
    assert torch.equal(after[..., 128:], before[..., 128:])
    assert not torch.equal(after[..., 64:128], before[..., 64:128])


def test_last_block_may_be_partial():
    mixed = MixedPrecisionLinear(make_linear(out_features=200), block_rows=64)
    assert mixed.n_blocks == 4
    assert mixed.block_sizes().tolist() == [64, 64, 64, 8]
    mixed.levels[-1] = int(Level.INT8)
    x = make_input()
    assert mixed(x).shape[-1] == 200


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


def test_layout_reports_actual_buffers():
    ctl = Controller({"layers.0.a": MixedPrecisionLinear(make_linear(), block_rows=64)})
    ctl.set_blocks("layers.0.a", [0, 2], Level.NF4)
    assert ctl.layout() == {"layers.0.a": [2, 0, 2, 0]}
    with pytest.raises(KeyError):
        ctl.set_layer(5, Level.NF4)
