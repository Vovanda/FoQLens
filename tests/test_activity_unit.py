"""Forward signals of the address and the projection onto every block (#18), on synthetic tensors: no model, CPU."""

import numpy as np
import pytest
import torch
from torch import nn

from foqlens.activity import (
    HeadEnergyScorer,
    NeuronActivityScorer,
    block_activity,
    block_offsets,
    head_energy,
    valid_tokens,
)
from foqlens.graph_zones import GraphZones
from foqlens.layouts import ProjectedStrength
from foqlens.precision import MixedPrecisionLinear
from foqlens.projection import Projection


def test_valid_tokens_leave_out_the_first_and_the_padding():
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    assert valid_tokens(mask).tolist() == [[0, 1, 1, 0], [0, 1, 1, 1]]


def test_block_activity_is_the_block_norm_averaged_over_valid_tokens():
    inputs = torch.zeros(1, 4, 130)  # three groups of 64 columns, the last one of 2
    inputs[0, 0] = 100.0  # the first token never counts
    inputs[0, 1, :64] = 1.0  # group 0 at one of two valid tokens: norm 8
    inputs[0, 2, 128:] = 3.0  # group 2 at the other: norm 3 * sqrt 2
    inputs[0, 3] = 50.0  # padding
    valid = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    torch.testing.assert_close(block_activity(inputs, valid, 64)[0], torch.tensor([4.0, 0.0, 1.5 * 2**0.5]))


def test_head_energy_splits_the_o_proj_input_into_heads_side_by_side():
    inputs = torch.zeros(1, 2, 8)  # two heads of 4
    inputs[0, 1, 4:] = 1.0  # head 1 at the only valid token: norm 2
    valid = torch.tensor([[0.0, 1.0]])
    assert head_energy(inputs, valid, n_heads=2)[0].tolist() == [0.0, 2.0]


def tiny_modules() -> dict[str, MixedPrecisionLinear]:
    """Two layers of a toy decoder: q of 2 heads x 128 rows, o, gate, up and down, blocks of 64 rows."""
    torch.manual_seed(0)
    shapes = {"self_attn.q_proj": (32, 256), "self_attn.o_proj": (256, 32), "mlp.gate_proj": (32, 192),
              "mlp.up_proj": (32, 192), "mlp.down_proj": (192, 32)}
    return {f"layers.{i}.{name}": MixedPrecisionLinear(nn.Linear(inp, out, bias=False, dtype=torch.bfloat16))
            for i in range(2) for name, (inp, out) in shapes.items()}


def test_neuron_activity_lands_on_the_gate_and_up_blocks_of_its_layer_only():
    modules = tiny_modules()
    scorer = NeuronActivityScorer(modules, layers={1})
    assert list(scorer.down) == ["layers.1.mlp.down_proj"]
    activity = {"layers.1.mlp.down_proj": torch.tensor([[1.0, 2.0, 3.0]])}
    out = scorer.assemble(activity, 1)[0]
    at = block_offsets(modules)
    assert out[at["layers.1.mlp.gate_proj"]: at["layers.1.mlp.gate_proj"] + 3].tolist() == [1.0, 2.0, 3.0]
    assert out[at["layers.1.mlp.up_proj"]: at["layers.1.mlp.up_proj"] + 3].tolist() == [1.0, 2.0, 3.0]
    assert out.sum() == pytest.approx(12.0)  # nothing anywhere else


def test_head_energy_lands_on_every_q_block_of_its_head():
    modules = tiny_modules()
    scorer = HeadEnergyScorer(modules, n_heads=2, layers={0})
    out = scorer.assemble({"layers.0.self_attn.o_proj": torch.tensor([[5.0, 7.0]])}, 1)[0]
    q = block_offsets(modules)["layers.0.self_attn.q_proj"]
    assert out[q: q + 4].tolist() == [5.0, 5.0, 7.0, 7.0] and out.sum() == pytest.approx(24.0)


def test_the_projection_recovers_an_exact_linear_map():
    rng = np.random.default_rng(0)
    signals = rng.standard_normal((200, 6))
    truth = rng.standard_normal((6, 30))
    scores = 2.0 + signals @ truth
    fitted = Projection.fit(signals, scores, alpha=0.0)
    np.testing.assert_allclose(fitted.weights.numpy(), truth, atol=1e-8)
    fresh = rng.standard_normal((5, 6))
    np.testing.assert_allclose(fitted.apply(fresh).numpy(), 2.0 + fresh @ truth, atol=1e-8)


def test_a_larger_ridge_never_gives_a_larger_map():
    rng = np.random.default_rng(1)
    signals, scores = rng.standard_normal((50, 8)), rng.standard_normal((50, 20))
    norms = [float(Projection.fit(signals, scores, alpha).weights.norm()) for alpha in (0.0, 1.0, 10.0, 100.0)]
    assert norms == sorted(norms, reverse=True)
    with pytest.raises(ValueError, match="alpha"):
        Projection.fit(signals, scores, -1.0)


def test_a_zone_is_as_strong_as_the_heads_that_point_at_its_center():
    weights = np.zeros((2, 10))
    weights[0, 3], weights[1, 7] = 1.0, 1.0  # head 0 points at block 3, head 1 at block 7
    strength = ProjectedStrength(signals=np.array([[0.5, 4.0]]), weights=weights, scale=2.0)
    zones_ = GraphZones(centers=np.array([3, 7, 1]), radii=np.ones(3))
    assert strength.strengths(0, zones_).tolist() == [0.25, 1.0, 0.0]  # 0.5 / 2, 4 / 2 clipped, nobody
