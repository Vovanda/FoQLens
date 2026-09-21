"""The block graph of the signal's path through the weights (M3 of #4): who writes to whom, and how strongly.

Mechanism 5 grows a zone along the way a signal actually flows in the model, with no calibration data. A block a -
64 rows rho of module A - feeds a block b - 64 rows sigma of module B - through the columns C(rho) at which B reads
A's outputs, and the edge conducts

    kappa(a -> b) = ||A[rho, :]||_F * ||B[sigma, C(rho)]||_F,

an upper bound of what a passes to b at a unit input (the Frobenius norm is submultiplicative). Every column of B is
fed by one row of A (source_rows), so the second factor of every pair of blocks comes from one product: the squares
of B summed over its row blocks, [blocks of B, columns], times the indicator of which block of A feeds each column.

The pairs follow the order of a Gemma 4 decoder layer (modeling_gemma4.Gemma4TextDecoderLayer.forward): inside a
sub-block the readers feed the writer (q, k, v -> o; gate, up -> down; per_layer_input_gate -> per_layer_projection),
and a writer feeds the readers of the next sub-block (o -> gate, up; down -> per_layer_input_gate;
per_layer_projection -> q, k, v of the next layer). Further down the stream a signal travels along a chain of such
edges, so depth stays a distance - linking every early writer to every later reader would give ~10^7 edges (#4).

The graph keeps every block's STRONGEST edges, an edge staying if it is among the strongest of either end, and an
edge is as long as the median conductance over its own: a strong link is short. Distances along it are the geodesic
of metric.geodesic, as on the co-activation graph, so every zone rule works on it unchanged.
Derivations: .claude/session-context/docs/signal-path-metric.md (into the docs with the owner's word).

Invariant: kappa is the product of the upstream block's norm and the norm of the downstream submatrix it feeds -
checked against a direct computation.
Invariant: a KV head feeds the o_proj columns of every query head of its group, a query head only its own columns.
Invariant: the table is symmetric, no block is its own neighbour, and every edge has a positive length.
Invariant: scaling one module's weights by c > 0 scales the conductance of every edge through it by c.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from foqlens.metric import edge_table

BLOCK_ROWS = 64  # a block of the controller and the kernel
STRONGEST = 16  # edges every block keeps: the k of the co-activation graph (#4), so the two graphs are alike in degree

# A decoder layer as sub-blocks in order: (readers of the stream, the writer back to it).
SUB_BLOCKS = (
    (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "self_attn.o_proj"),
    (("mlp.gate_proj", "mlp.up_proj"), "mlp.down_proj"),
    (("per_layer_input_gate",), "per_layer_projection"),
)
KV = ("self_attn.k_proj", "self_attn.v_proj")


def source_rows(kind: str, upstream_rows: int, columns: int, n_heads: int) -> torch.Tensor:
    """For every input column of the downstream module, the row of the upstream module that feeds it: [columns].

    The stream, neurons, per-layer channels and q -> o keep their index. A KV head feeds the o_proj columns of the
    query heads of its group: column h * head_dim + w reads row (h // group) * head_dim + w (HF repeat_kv).
    """
    column = torch.arange(columns)
    if kind not in KV:
        return column
    head_dim = columns // n_heads
    group = n_heads // (upstream_rows // head_dim)
    return (column // head_dim) // group * head_dim + column % head_dim


def _row_blocks(rows: int, device: torch.device) -> torch.Tensor:
    """The block of every row: [rows]."""
    return torch.arange(rows, device=device) // BLOCK_ROWS


def block_norms(weight: torch.Tensor) -> torch.Tensor:
    """||W[rho, :]||_F of every block of rows: [blocks], on the weight's device."""
    squares = weight.float().square().sum(dim=1)
    return torch.zeros(-(-len(squares) // BLOCK_ROWS), device=weight.device).index_add_(
        0, _row_blocks(len(squares), weight.device), squares).sqrt()


def conductance(upstream: torch.Tensor, downstream: torch.Tensor, feeds: torch.Tensor) -> torch.Tensor:
    """kappa [blocks of upstream, blocks of downstream]; feeds[c] is the upstream row read at downstream column c.
    Computed on the downstream weight's device; the upstream and `feeds` are moved there."""
    device = downstream.device
    squares = downstream.float().square()
    by_block = torch.zeros(-(-downstream.shape[0] // BLOCK_ROWS), downstream.shape[1], device=device).index_add_(
        0, _row_blocks(downstream.shape[0], device), squares)  # [downstream blocks, columns]
    indicator = torch.zeros(downstream.shape[1], -(-upstream.shape[0] // BLOCK_ROWS), device=device)
    indicator[torch.arange(downstream.shape[1], device=device), feeds.to(device) // BLOCK_ROWS] = 1.0
    return block_norms(upstream.to(device))[:, None] * (by_block @ indicator).sqrt().T


def _module(layer: int, kind: str) -> str:
    return f"layers.{layer}.{kind}"


def _pairs(names: list[str]) -> list[tuple[str, str]]:
    """(upstream, downstream) module names in the order a signal flows, over every layer."""
    layers = sorted({int(n.split(".")[1]) for n in names})
    have = set(names)
    pairs = []
    for i, layer in enumerate(layers):
        steps = [(readers, writer) for readers, writer in SUB_BLOCKS if _module(layer, writer) in have]
        for s, (readers, writer) in enumerate(steps):
            source = _module(layer, writer)
            pairs += [(_module(layer, r), source) for r in readers if _module(layer, r) in have]
            # the writer feeds the readers of the next sub-block: of this layer, or the first of the next layer
            if s + 1 < len(steps):
                following = [_module(layer, r) for r in steps[s + 1][0]]
            elif i + 1 < len(layers):
                following = [_module(layers[i + 1], r) for r in SUB_BLOCKS[0][0]]
            else:
                following = []
            pairs += [(source, reader) for reader in following if reader in have]
    return pairs


def signal_path_table(weights: Mapping[str, torch.Tensor], n_heads: int,
                      strongest: int = STRONGEST) -> tuple[torch.Tensor, torch.Tensor]:
    """The block graph of M3: a symmetric neighbour table [n_blocks, width] padded with metric.PAD, and edge lengths.

    `weights` are the controlled modules' [out, in] weights in the controller's order - blocks are numbered along it.
    """
    names = list(weights)
    offsets = dict(zip(names, np.cumsum([0] + [-(-w.shape[0] // BLOCK_ROWS) for w in weights.values()])))
    n = int(sum(-(-w.shape[0] // BLOCK_ROWS) for w in weights.values()))
    heads, tails, kappas = [], [], []
    for up, down in _pairs(names):
        kind = up.split(".", 2)[2]
        feeds = source_rows(kind, weights[up].shape[0], weights[down].shape[1], n_heads)
        # the products run where the weights lie; the graph itself - a few hundred thousand edges - is built on the CPU
        kappa = conductance(weights[up], weights[down], feeds).cpu()
        a, b = torch.nonzero(kappa > 0, as_tuple=True)
        heads.append(a + int(offsets[up]))
        tails.append(b + int(offsets[down]))
        kappas.append(kappa[a, b])
    heads, tails, kappas = torch.cat(heads), torch.cat(tails), torch.cat(kappas)
    # undirected: an edge stays if it is among the strongest of either end
    both_heads, both_tails, both = torch.cat([heads, tails]), torch.cat([tails, heads]), torch.cat([kappas, kappas])
    # entry i of the first half is edge i seen from its upstream end, of the second half from its downstream end
    edge = _strongest_per_end(both_heads, both, strongest).view(2, -1).any(dim=0).repeat(2)
    lengths = both.median() / both
    return edge_table(both_heads[edge], both_tails[edge], lengths[edge], n)


def votes(activity: np.ndarray, near: np.ndarray, length: np.ndarray, spread: float = 1.0) -> np.ndarray:
    """Every block's votes: the activity of the blocks it is joined to, weighted by the coupling - the shorter the
    edge, the stronger the vote: [questions, blocks].

    `spread` is how wide a voter looks (Volodya 20.09): the share of its own edges a block's vote is gathered over,
    the strongest first. At 0 nobody votes, and at 1 every edge of a block carries one. This is the one source that
    looks ahead per question (docs/precision-regulator.md): a block is scored by what feeds it, not by itself.

    Invariant: a block's votes do not read its own activity - only its neighbours'.
    Invariant: the votes are linear in the activity - scaling it by c scales every vote by c.
    Invariant: at spread 0 every vote is zero.
    """
    weight = np.where(near >= 0, 1.0 / np.maximum(length, 1e-12), 0.0)
    kept = max(int(round(spread * near.shape[1])), 0)
    order = np.argsort(-weight, axis=1)
    out = np.zeros_like(activity, dtype=float)
    for rank in range(kept):
        column = np.take_along_axis(near, order[:, rank:rank + 1], axis=1)[:, 0]
        edge = np.take_along_axis(weight, order[:, rank:rank + 1], axis=1)[:, 0]
        held = column >= 0
        out[:, held] += activity[:, column[held]] * edge[held]
    return out


def _strongest_per_end(ends: torch.Tensor, kappa: torch.Tensor, k: int) -> torch.Tensor:
    """bool [edges]: whether an edge is among the k strongest edges of its `ends` block."""
    order = torch.argsort(kappa, descending=True, stable=True)
    ends_sorted = ends[order]
    grouped = torch.argsort(ends_sorted, stable=True)  # by block, strongest first within a block
    blocks = ends_sorted[grouped]
    start = torch.searchsorted(blocks, blocks, side="left")
    rank = torch.arange(len(blocks)) - start
    keep = torch.zeros(len(ends), dtype=torch.bool)
    keep[order[grouped[rank < k]]] = True
    return keep
