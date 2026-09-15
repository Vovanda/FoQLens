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
- Invariant: drop_bf16 changes no output of a read depth; afterwards the module holds its sliced
  copy alone, and a level it cannot read is refused when set, not when computed.
- Invariant: with depth caps a block stores only its first cap slices, reads within its cap
  exactly as before, and a read deeper than its cap is refused when set.
- Invariant: a baked level is bit-exact with reading the same depth from the sliced copy; afterwards
  the module holds that weight alone and reads that level alone - bf16 is refused when set.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from foqlens.model import text_layers
from foqlens.quant import N_SLICES, SLICE_BITS, CappedSlicedWeight, Int8Weight, Level, Nf4Weight, SlicedWeight

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
# Where a level reads from: a packed copy of its own, or the one sliced copy every read depth
# shares. bf16 reads the weight itself, ZERO reads nothing.
STORAGE = {Level.INT8: Int8Weight, Level.NF4: Nf4Weight} | {lv: SlicedWeight for lv in Level if lv.slices}
# Slices a level reads, by code; the levels that are not read depths count as the full copy.
SLICES_BY_CODE = np.array([lv.slices if lv.slices or lv is Level.ZERO else N_SLICES for lv in Level], dtype=np.uint8)


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
        self._device = linear.weight.device
        self._native = Level.BF16  # the level the weight itself holds: bf16, or the read depth baked into it
        self._packed: dict[type, Int8Weight | Nf4Weight | SlicedWeight] = {}
        self.readable: frozenset[Level] = frozenset(Level)
        self._caps = np.full(self.n_blocks, N_SLICES, dtype=np.uint8)  # slices every block stores
        self.set_levels(Level.BF16)

    @property
    def levels(self) -> np.ndarray:
        """Level code per block: [n_blocks], or [batch, n_blocks] for per-sample layouts. A copy."""
        return self._levels.copy()

    @property
    def storages(self) -> tuple[type, ...]:
        """Kinds of quantized copies of the weight that are held, in the order they were first needed."""
        return tuple(self._packed)

    def _materialize(self, level: Level) -> None:
        """Quantize the weight into the level's storage on its first use; bf16, ZERO and a baked level need no copy."""
        kind = STORAGE.get(level)
        if kind is not None and kind not in self._packed and level is not self._native:
            self._packed[kind] = kind.quantize(self.weight.data)

    def set_levels(self, levels: Level | int | np.ndarray, codes: torch.Tensor | None = None) -> None:
        """One level for all blocks, a level per block, or a level per block per sample of the batch.

        `codes`, when given, are the same levels already on the device (Controller.set_layout uploads a
        whole layout once instead of once per module).
        """
        arr = np.asarray(levels, dtype=np.uint8)
        if arr.ndim == 0:
            arr = np.full(self.n_blocks, arr, dtype=np.uint8)
        if arr.ndim not in (1, 2) or arr.shape[-1] != self.n_blocks:
            raise ValueError(f"levels of shape {arr.shape} for {self.n_blocks} blocks")
        used = tuple(Level(int(c)) for c in np.unique(arr))
        if not self.readable.issuperset(used):
            raise ValueError(f"levels {[lv.name for lv in used if lv not in self.readable]} are not held by this module")
        if (SLICES_BY_CODE[arr] > self._caps).any():
            raise ValueError("a block is read deeper than the depth it stores")
        self._levels = arr.copy()
        self._used = used
        for level in self._used:
            self._materialize(level)
        self._rows = None if len(self._used) == 1 else self._row_codes(arr, codes)

    def _row_codes(self, arr: np.ndarray, codes: torch.Tensor | None = None) -> torch.Tensor:
        """Level code per output row on the GPU: [out] or [batch, 1, out], broadcast over tokens.

        Blocks are expanded to rows on the device: one small copy of block codes, not of row codes.
        """
        blocks = torch.as_tensor(arr, device=self._device) if codes is None else codes
        rows = blocks.repeat_interleave(self.block_rows, dim=-1)[..., : self.out_features]
        return rows if arr.ndim == 1 else rows.unsqueeze(1)

    def block_sizes(self) -> np.ndarray:
        """Number of rows in every block; the last one may be partial."""
        sizes = np.full(self.n_blocks, self.block_rows, dtype=np.int64)
        sizes[-1] = self.out_features - self.block_rows * (self.n_blocks - 1)
        return sizes

    # What a module still reads after drop_bf16: the depths of its sliced copy, and nothing.
    RESIDENT = frozenset({Level.ZERO, *(lv for lv in Level if lv.slices)})

    def drop_bf16(self) -> None:
        """Keep only the sliced copy: the bf16 weight and every other copy leave the GPU.

        Afterwards the module reads ZERO and the read depths only; the current layout must be one of those.
        """
        if not self.RESIDENT.issuperset(self._used):
            raise ValueError("switch to ZERO or read depths before dropping bf16")
        self._materialize(Level.D8)
        self._packed = {SlicedWeight: self._packed[SlicedWeight]}
        self.weight = None
        self.readable = self.RESIDENT

    def bake(self, level: Level) -> None:
        """Read one depth at the cost of bf16: the weight read to it replaces the bf16 weight and every copy.

        A uniform level then unpacks once instead of on every call. Afterwards the module reads that
        level alone; the bf16 weight is gone, so asking for bf16 is refused rather than answered wrongly.
        """
        if not level.slices:
            raise ValueError(f"only a read depth is baked, not {level.name}")
        if self._native is not Level.BF16 or self.weight is None:
            raise ValueError("a level is baked once, from the bf16 weight")
        self._materialize(level)
        baked = self._packed[SlicedWeight].dequantize(self.weight.dtype, level.slices)
        self.weight = nn.Parameter(baked, requires_grad=False)
        self._packed = {}
        self._native = level
        self.readable = frozenset({level})
        self.set_levels(level)

    def set_caps(self, caps: np.ndarray) -> None:
        """Store every block only to its depth cap (slices, 0 ... N_SLICES); the deeper slices leave the GPU.

        Only on a resident module (after drop_bf16), once, from its full sliced copy; the current
        layout must not read any block deeper than its new cap.
        """
        caps = np.asarray(caps, dtype=np.uint8)
        if self.weight is not None:
            raise ValueError("depth caps need a resident module: call drop_bf16 first")
        if caps.shape != (self.n_blocks,) or (caps > N_SLICES).any():
            raise ValueError(f"caps of shape {caps.shape} for {self.n_blocks} blocks, each 0 ... {N_SLICES}")
        full = self._packed[SlicedWeight]
        if not isinstance(full, SlicedWeight):
            raise ValueError("depth caps are set once, from the full sliced copy")
        if (SLICES_BY_CODE[self._levels] > caps).any():
            raise ValueError("the current layout reads a block deeper than its new cap")
        block_caps = torch.as_tensor(caps, device=self._device)
        self._packed[SlicedWeight] = CappedSlicedWeight.from_sliced(full, block_caps, self.block_rows)
        self._caps = caps.copy()

    @property
    def caps(self) -> np.ndarray:
        """Slices every block stores. A copy."""
        return self._caps.copy()

    def stored_bytes(self) -> int:
        """Bytes of the sliced copy held, without its scales: every slice, or only those under the caps."""
        store = self._packed.get(SlicedWeight)
        return 0 if store is None else store.nbytes

    def output_at(self, level: Level, x: torch.Tensor) -> torch.Tensor:
        """The whole output as if every block were read at this level."""
        if level not in self.readable:
            raise ValueError(f"{level.name} is not held by this module")
        if level is self._native:
            return F.linear(x, self.weight, self.bias)
        if level in STORAGE:
            self._materialize(level)
            store = self._packed[STORAGE[level]]
            return store.matmul(x, self.bias, level.slices) if level.slices else store.matmul(x, self.bias)
        out = x.new_zeros(*x.shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._rows is None:
            return self.output_at(self._used[0], x)
        if self._levels.ndim == 2 and x.shape[0] != self._levels.shape[0]:
            raise ValueError(f"batch of {x.shape[0]} for per-sample layouts of {self._levels.shape[0]}")
        outputs = self._outputs_at(self._used, x)
        out = outputs[self._used[0]]
        for level in self._used[1:]:
            out = torch.where(self._rows == int(level), outputs[level], out)
        return out

    def _outputs_at(self, levels: tuple[Level, ...], x: torch.Tensor) -> dict[Level, torch.Tensor]:
        """output_at for every level; several read depths come from one accumulation of the sliced copy."""
        depths = [lv for lv in levels if lv.slices]
        outputs = {lv: self.output_at(lv, x) for lv in levels if not lv.slices}
        if len(depths) < 2:
            return outputs | {lv: self.output_at(lv, x) for lv in depths}
        self._materialize(depths[0])
        store = self._packed[SlicedWeight]
        return outputs | dict(zip(depths, store.linear_at_depths(x, self.bias, [lv.slices for lv in depths])))

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
        # one copy of the whole layout to the device; every module takes a view of its own columns
        codes = torch.as_tensor(levels, device=next(iter(self.modules.values()))._device)
        for (start, stop), module in zip(zip(self._bounds[:-1], self._bounds[1:]), self.modules.values()):
            module.set_levels(levels[..., start:stop], codes[..., start:stop])

    def drop_bf16(self) -> None:
        """Every module keeps only its sliced copy (MixedPrecisionLinear.drop_bf16); the cache returns the freed memory."""
        for module in self.modules.values():
            module.drop_bf16()
        torch.cuda.empty_cache()

    def bake(self, level: Level) -> None:
        """Every module reads `level` from a weight unpacked once (MixedPrecisionLinear.bake); the cache returns the freed memory."""
        for module in self.modules.values():
            module.bake(level)
        torch.cuda.empty_cache()

    def set_caps(self, caps: np.ndarray) -> None:
        """Depth caps over all blocks in module order (see MixedPrecisionLinear.set_caps); the cache returns the freed memory."""
        caps = np.asarray(caps, dtype=np.uint8)
        if caps.shape != (self.n_blocks,):
            raise ValueError(f"caps of shape {caps.shape} for {self.n_blocks} blocks")
        for (start, stop), module in zip(zip(self._bounds[:-1], self._bounds[1:]), self.modules.values()):
            module.set_caps(caps[start:stop])
        torch.cuda.empty_cache()

    def stored_bits(self) -> float:
        """Mean stored bits per weight of the sliced copies, weighted by weight count: SLICE_BITS per slice kept, no scales."""
        total_bits, total_weights = 0.0, 0
        for m in self.modules.values():
            total_bits += float((m.caps.astype(np.int64) * SLICE_BITS * m.block_sizes() * m.in_features).sum())
            total_weights += m.in_features * m.out_features
        return total_bits / total_weights

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
