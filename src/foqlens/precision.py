"""Precision controller: bit depth per layer, module and block of weight rows.

Every linear module of the text decoder is replaced by a MixedPrecisionLinear. It keeps the
original bf16 weight and, for the levels some layout has used, a packed int8 or nf4 copy; the level
is set per block of block_rows output rows, either one layout for the whole batch or one layout
per sample of the batch.

Hot-path rule: levels live on the CPU as a numpy array, so forward never synchronizes with the
GPU to find out what to compute. A mixed layout computes the output once per level in use and
selects rows with torch.where.

Invariants (each one has a test):
- Invariant: an all-bf16 layout is bit-exact with the original nn.Linear.
- Invariant: the rows of a block depend only on that block's level.
- Invariant: with per-sample layouts, sample b is bit-exact with sample b of the same batch run
  under layout b for every sample - a per-sample layout changes nothing but the selection.
- Invariant: forward never reads levels from the GPU.
- Invariant: a packed copy exists only for a level some layout has used - a bench that only
  reads bf16 and ZERO holds no int8 or nf4 copy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from foqlens.model import text_layers
from foqlens.quant import Int8Weight, Level, Nf4Weight

# Linear modules inside a decoder layer. KV-shared layers have no k/v_proj - those are skipped.
CONTROLLED = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "per_layer_input_gate",
    "per_layer_projection",
)

DEFAULT_BLOCK_ROWS = 64
BITS_BY_CODE = np.array([lv.bits for lv in Level], dtype=np.float64)
# Levels read from a packed copy of the weight; bf16 reads the weight itself, ZERO reads nothing.
PACKED = {Level.INT8: Int8Weight, Level.NF4: Nf4Weight}


class MixedPrecisionLinear(nn.Module):
    """An nn.Linear whose every block of output rows is read at its own precision."""

    def __init__(self, linear: nn.Linear, block_rows: int = DEFAULT_BLOCK_ROWS):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.block_rows = block_rows
        self.n_blocks = math.ceil(self.out_features / block_rows)
        self.weight = linear.weight
        self.weight.requires_grad_(False)
        self.bias = linear.bias
        self._packed: dict[Level, Int8Weight | Nf4Weight] = {}
        self.set_levels(Level.BF16)

    @property
    def levels(self) -> np.ndarray:
        """Level code per block: [n_blocks], or [batch, n_blocks] for per-sample layouts. A copy."""
        return self._levels.copy()

    @property
    def packed_levels(self) -> tuple[Level, ...]:
        """Levels whose packed copy of the weight is held, in the order they were first used."""
        return tuple(self._packed)

    def _materialize(self, level: Level) -> None:
        """Quantize the weight into the level on its first use; bf16 and ZERO need no copy."""
        if level in PACKED and level not in self._packed:
            self._packed[level] = PACKED[level].quantize(self.weight.data)

    def set_levels(self, levels: Level | int | np.ndarray) -> None:
        """One level for all blocks, a level per block, or a level per block per sample of the batch."""
        arr = np.asarray(levels, dtype=np.uint8)
        if arr.ndim == 0:
            arr = np.full(self.n_blocks, arr, dtype=np.uint8)
        if arr.ndim not in (1, 2) or arr.shape[-1] != self.n_blocks:
            raise ValueError(f"levels of shape {arr.shape} for {self.n_blocks} blocks")
        self._levels = arr.copy()
        self._used = tuple(Level(int(c)) for c in np.unique(arr))
        for level in self._used:
            self._materialize(level)
        self._rows = None if len(self._used) == 1 else self._row_codes(arr)

    def _row_codes(self, arr: np.ndarray) -> torch.Tensor:
        """Level code per output row on the GPU: [out] or [batch, 1, out], broadcast over tokens."""
        rows = np.repeat(arr, self.block_rows, axis=-1)[..., : self.out_features]
        codes = torch.as_tensor(rows, device=self.weight.device)
        return codes if arr.ndim == 1 else codes.unsqueeze(1)

    def block_sizes(self) -> np.ndarray:
        """Number of rows in every block; the last one may be partial."""
        sizes = np.full(self.n_blocks, self.block_rows, dtype=np.int64)
        sizes[-1] = self.out_features - self.block_rows * (self.n_blocks - 1)
        return sizes

    def output_at(self, level: Level, x: torch.Tensor) -> torch.Tensor:
        """The whole output as if every block were read at this level."""
        if level is Level.BF16:
            return F.linear(x, self.weight, self.bias)
        if level in PACKED:
            self._materialize(level)
            return self._packed[level].matmul(x, self.bias)
        out = x.new_zeros(*x.shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._rows is None:
            return self.output_at(self._used[0], x)
        if self._levels.ndim == 2 and x.shape[0] != self._levels.shape[0]:
            raise ValueError(f"batch of {x.shape[0]} for per-sample layouts of {self._levels.shape[0]}")
        out = self.output_at(self._used[0], x)
        for level in self._used[1:]:
            out = torch.where(self._rows == int(level), self.output_at(level, x), out)
        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, blocks={self.n_blocks}x{self.block_rows}"


class Controller:
    """Precision layout over the bench: set levels and read back what is actually in place."""

    def __init__(self, modules: dict[str, MixedPrecisionLinear]):
        self.modules = modules
        self._bounds = np.cumsum([0] + [m.n_blocks for m in modules.values()])

    @property
    def n_blocks(self) -> int:
        return int(self._bounds[-1])

    def names(self, layer: int | None = None) -> list[str]:
        if layer is None:
            return list(self.modules)
        prefix = f"layers.{layer}."
        return [n for n in self.modules if n.startswith(prefix)]

    def set_all(self, level: Level) -> None:
        for module in self.modules.values():
            module.set_levels(level)

    def set_layer(self, layer: int, level: Level) -> None:
        names = self.names(layer)
        if not names:
            raise KeyError(f"no layer {layer}")
        for name in names:
            self.modules[name].set_levels(level)

    def set_module(self, name: str, level: Level) -> None:
        self.modules[name].set_levels(level)

    def set_blocks(self, name: str, blocks: Iterable[int], level: Level) -> None:
        levels = self.modules[name].levels
        levels[..., list(blocks)] = int(level)
        self.modules[name].set_levels(levels)

    def set_layout(self, levels: np.ndarray) -> None:
        """Levels over all blocks of all modules in module order: [n_blocks] or [batch, n_blocks]."""
        levels = np.asarray(levels, dtype=np.uint8)
        if levels.shape[-1] != self.n_blocks:
            raise ValueError(f"layout of {levels.shape[-1]} blocks for {self.n_blocks}")
        for (start, stop), module in zip(zip(self._bounds[:-1], self._bounds[1:]), self.modules.values()):
            module.set_levels(levels[..., start:stop])

    def layout(self) -> dict[str, list]:
        """Level of every block of every module - the state the modules compute with."""
        return {name: m.levels.tolist() for name, m in self.modules.items()}

    def mean_bits(self) -> float | np.ndarray:
        """Mean nominal bits per weight over the controlled modules, weighted by weight count.

        A float for one layout, one value per sample for per-sample layouts. Nominal: 16 / 8 / 4
        without the overhead of group scales. Embeddings, lm_head and per_layer_model_projection
        are not controlled and do not enter the mean.
        """
        total_bits, total_weights = 0.0, 0
        for m in self.modules.values():
            weights_per_block = m.block_sizes() * m.in_features
            total_bits = total_bits + (BITS_BY_CODE[m.levels] * weights_per_block).sum(axis=-1)
            total_weights += m.in_features * m.out_features
        result = total_bits / total_weights
        return float(result) if np.ndim(result) == 0 else result


def _resolve(parent: nn.Module, dotted: str) -> tuple[nn.Module, str] | None:
    *path, leaf = dotted.split(".")
    for part in path:
        parent = getattr(parent, part, None)
        if parent is None:
            return None
    child = getattr(parent, leaf, None)
    return (parent, leaf) if isinstance(child, nn.Linear) else None


@torch.no_grad()
def install(model: nn.Module, block_rows: int = DEFAULT_BLOCK_ROWS) -> Controller:
    """Replace the text decoder's linear modules with controlled ones. Everything starts in bf16."""
    modules: dict[str, MixedPrecisionLinear] = {}
    for i, layer in enumerate(text_layers(model)):
        for dotted in CONTROLLED:
            found = _resolve(layer, dotted)
            if found is None:
                continue
            parent, leaf = found
            mixed = MixedPrecisionLinear(getattr(parent, leaf), block_rows)
            setattr(parent, leaf, mixed)
            modules[f"layers.{i}.{dotted}"] = mixed
    return Controller(modules)
