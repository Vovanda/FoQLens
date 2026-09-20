"""The signal-path graph of the weights (foqlens.coupling, M3 of #4): no model, CPU."""

import numpy as np
import torch

from foqlens.coupling import BLOCK_ROWS, _pairs, conductance, signal_path_table, source_rows, votes
from foqlens.metric import PAD

HIDDEN, N_HEADS, HEAD_DIM, KV_HEADS, NEURONS, CHANNELS = 128, 4, 32, 2, 192, 64


def toy_layer(layer: int, with_kv: bool = True, seed: int = 0) -> dict[str, torch.Tensor]:
    """One Gemma 4 decoder layer at toy size, in the controller's module order."""
    g = torch.Generator().manual_seed(seed + layer)
    shapes = {"self_attn.q_proj": (N_HEADS * HEAD_DIM, HIDDEN), "self_attn.k_proj": (KV_HEADS * HEAD_DIM, HIDDEN),
              "self_attn.v_proj": (KV_HEADS * HEAD_DIM, HIDDEN), "self_attn.o_proj": (HIDDEN, N_HEADS * HEAD_DIM),
              "mlp.gate_proj": (NEURONS, HIDDEN), "mlp.up_proj": (NEURONS, HIDDEN), "mlp.down_proj": (HIDDEN, NEURONS),
              "per_layer_input_gate": (CHANNELS, HIDDEN), "per_layer_projection": (HIDDEN, CHANNELS)}
    if not with_kv:  # a KV-shared layer has no k_proj and v_proj of its own
        shapes = {k: v for k, v in shapes.items() if k not in ("self_attn.k_proj", "self_attn.v_proj")}
    return {f"layers.{layer}.{k}": torch.randn(*s, generator=g) for k, s in shapes.items()}


def toy_model() -> dict[str, torch.Tensor]:
    return toy_layer(0) | toy_layer(1, with_kv=False)


def test_a_kv_head_feeds_the_o_columns_of_its_group_and_a_query_head_its_own():
    kv = source_rows("self_attn.v_proj", KV_HEADS * HEAD_DIM, N_HEADS * HEAD_DIM, N_HEADS)
    group = N_HEADS // KV_HEADS
    for h in range(N_HEADS):
        cols = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        assert kv[cols].tolist() == list(range((h // group) * HEAD_DIM, (h // group + 1) * HEAD_DIM)), h
    q = source_rows("self_attn.q_proj", N_HEADS * HEAD_DIM, N_HEADS * HEAD_DIM, N_HEADS)
    assert q.tolist() == list(range(N_HEADS * HEAD_DIM))


def test_the_conductance_is_the_upstream_norm_times_the_downstream_submatrix_it_feeds():
    model = toy_model()
    up, down = model["layers.0.self_attn.v_proj"], model["layers.0.self_attn.o_proj"]
    feeds = source_rows("self_attn.v_proj", up.shape[0], down.shape[1], N_HEADS)
    kappa = conductance(up, down, feeds)
    for a in range(up.shape[0] // BLOCK_ROWS):
        rho = torch.arange(a * BLOCK_ROWS, (a + 1) * BLOCK_ROWS)
        columns = torch.nonzero(torch.isin(feeds, rho)).flatten()
        for b in range(down.shape[0] // BLOCK_ROWS):
            sigma = slice(b * BLOCK_ROWS, (b + 1) * BLOCK_ROWS)
            direct = up[rho].norm() * down[sigma][:, columns].norm()
            assert torch.isclose(kappa[a, b], direct, rtol=1e-5), (a, b)


def test_scaling_a_module_scales_every_edge_through_it():
    model = toy_model()
    up, down = model["layers.0.mlp.gate_proj"], model["layers.0.mlp.down_proj"]
    feeds = source_rows("mlp.gate_proj", up.shape[0], down.shape[1], N_HEADS)
    torch.testing.assert_close(conductance(3.0 * up, down, feeds), 3.0 * conductance(up, down, feeds))
    torch.testing.assert_close(conductance(up, 3.0 * down, feeds), 3.0 * conductance(up, down, feeds))


def test_the_pairs_follow_the_order_of_a_decoder_layer():
    pairs = set(_pairs(list(toy_model())))
    expect = {
        ("layers.0.self_attn.q_proj", "layers.0.self_attn.o_proj"), ("layers.0.self_attn.v_proj", "layers.0.self_attn.o_proj"),
        ("layers.0.self_attn.o_proj", "layers.0.mlp.gate_proj"), ("layers.0.mlp.up_proj", "layers.0.mlp.down_proj"),
        ("layers.0.mlp.down_proj", "layers.0.per_layer_input_gate"),
        ("layers.0.per_layer_input_gate", "layers.0.per_layer_projection"),
        ("layers.0.per_layer_projection", "layers.1.self_attn.q_proj"),  # layer 1 shares its KV: q only
    }
    assert expect <= pairs
    assert ("layers.0.self_attn.o_proj", "layers.1.self_attn.q_proj") not in pairs  # depth stays a distance
    assert ("layers.0.per_layer_projection", "layers.1.self_attn.k_proj") not in pairs


def test_zones_on_the_signal_path_hold_the_ends_of_f_and_grow_along_the_layers():
    from foqlens.quant import Level
    from foqlens.strategies import Knobs, Space, zone_layout

    model = toy_layer(0) | toy_layer(1) | toy_layer(2)
    space = Space.signal_path(model, N_HEADS, strongest=4)
    n = space.table.shape[0]
    scores = np.zeros((1, n))
    scores[0, 0] = 1.0  # the first q block of layer 0 is the query's peak
    depths = (Level.D2, Level.D4, Level.D6, Level.D8)
    at = lambda f: zone_layout("m3", "query", scores, space, space.surface(), np.ones(n),  # noqa: E731
                               Knobs(Level.D2, f, 1.0), depths).levels(np.array([0]))[0]
    assert np.count_nonzero(at(0.0) > int(Level.D2)) == 1 and np.all(at(1.0) > int(Level.D2))
    lifted = [np.count_nonzero(at(f) > int(Level.D2)) for f in (0.1, 0.3, 0.6)]
    assert lifted == sorted(lifted) and lifted[0] < lifted[-1]


def test_the_table_is_symmetric_with_positive_lengths_and_no_block_its_own_neighbour():
    model = toy_model()
    table, lengths = signal_path_table(model, N_HEADS, strongest=3)
    n = sum(w.shape[0] // BLOCK_ROWS for w in model.values())
    assert table.shape[0] == n
    edges = {(a, int(b)) for a in range(n) for b in table[a] if b != PAD}
    assert all((b, a) in edges for a, b in edges)
    assert all(a != b for a, b in edges)
    assert torch.all(lengths[table != PAD] > 0)
    # every block keeps at least its strongest edge, and none more than the edges of both ends allow
    assert np.all((table != PAD).sum(dim=1).numpy() >= 1)


def a_graph() -> tuple[np.ndarray, np.ndarray]:
    """Four blocks in a line: 0 - 1 - 2 - 3, the middle edge twice as long as the outer ones."""
    near = np.array([[1, PAD], [0, 2], [1, 3], [2, PAD]])
    length = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
    return near, length


def test_a_blocks_votes_are_its_neighbours_activity_weighted_by_the_coupling():
    near, length = a_graph()
    loud = np.array([[0.0, 4.0, 1.0, 0.0]])  # block 1 is the loud one
    cast = votes(loud, near, length)
    assert cast[0, 0] == 4.0  # block 0 hears block 1 over an edge of length 1
    assert cast[0, 2] == 4.0 / 2  # block 2 hears the same block over an edge twice as long
    assert cast[0, 1] == 1.0 / 2  # block 1 hears its neighbours, never itself


def test_the_votes_are_linear_in_the_activity_and_empty_at_spread_zero():
    near, length = a_graph()
    activity = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, 1.0, 2.0]])
    assert np.allclose(votes(3 * activity, near, length), 3 * votes(activity, near, length))
    assert not votes(activity, near, length, spread=0.0).any()


def test_a_narrow_voter_keeps_its_strongest_edge_alone():
    near, length = a_graph()
    activity = np.ones((1, 4))
    narrow = votes(activity, near, length, spread=0.5)  # one edge of the two
    assert narrow[0, 1] == 1.0 and narrow[0, 2] == 1.0  # the short edge of each middle block, not the long one
    assert np.all(narrow <= votes(activity, near, length))
