"""Neighbourhoods of blocks, for widening the sharp windows of a mask (dilation).

Structural neighbours share what their rows compute: the gate and up rows of one block are the
same MLP neurons; the modules that write into the residual stream (o_proj, down_proj,
per_layer_projection) write the same stream coordinates under the same block index, in the same
layer and in the layers next to it. Index neighbours are the nearest blocks of the same matrix - a
control with exactly as many neighbours for every block and no shared meaning.

A neighbour table is [n_blocks, k] of block indices in the controller's block order, padded with PAD.

Invariant: structural neighbours are symmetric and a block is never its own neighbour.
Invariant: index_matched gives every block exactly as many neighbours as the table it matches.
"""

from __future__ import annotations

import numpy as np

PAIRED = (("mlp.gate_proj", "mlp.up_proj"),)
WRITERS = ("self_attn.o_proj", "mlp.down_proj", "per_layer_projection")
LAYER_REACH = 1  # a writer is linked to the writers of the layers this far away on either side
PAD = -1


def _split(name: str) -> tuple[int, str]:
    """'layers.12.mlp.down_proj' -> (12, 'mlp.down_proj')."""
    _, layer, module = name.split(".", 2)
    return int(layer), module


def _starts(blocks: dict[str, int]) -> dict[str, int]:
    starts = np.cumsum([0, *blocks.values()])[:-1]
    return dict(zip(blocks, starts.tolist()))


def _table(sets: list[set[int]]) -> np.ndarray:
    k = max((len(s) for s in sets), default=0)
    table = np.full((len(sets), k), PAD, dtype=np.int64)
    for g, s in enumerate(sets):
        table[g, : len(s)] = sorted(s)
    return table


def structural(blocks: dict[str, int]) -> np.ndarray:
    """Structural neighbour table of the blocks of {module name: n_blocks}, in that order."""
    start = _starts(blocks)
    where = {_split(name): name for name in blocks}
    sets: list[set[int]] = [set() for _ in range(sum(blocks.values()))]

    def link(a: str, b: str) -> None:
        for i in range(min(blocks[a], blocks[b])):
            sets[start[a] + i].add(start[b] + i)
            sets[start[b] + i].add(start[a] + i)

    for name in blocks:
        layer, module = _split(name)
        for first, second in PAIRED:
            if module == first and (layer, second) in where:
                link(name, where[(layer, second)])
        if module in WRITERS:
            for step in range(LAYER_REACH + 1):
                for other in WRITERS:
                    target = where.get((layer + step, other))
                    if target is not None and target != name:
                        link(name, target)
    return _table(sets)


def index_matched(blocks: dict[str, int], table: np.ndarray) -> np.ndarray:
    """For every block as many neighbours as `table` gives it: the nearest blocks of its matrix, b+1, b-1, b+2, ..."""
    start = _starts(blocks)
    sets = []
    for name, n in blocks.items():
        for b in range(n):
            k = int((table[start[name] + b] != PAD).sum())
            near = [b + sign * d for d in range(1, n) for sign in (1, -1) if 0 <= b + sign * d < n][:k]
            sets.append({start[name] + j for j in near})
    return _table(sets)
