"""Precision controller: bit depth per layer, module and block of weight rows.

Every linear module of the text decoder is replaced by a MixedPrecisionLinear. It keeps the
original bf16 weight and, for the levels some layout has used, a packed int8 or nf4 copy; the level
is set per block of block_rows output rows, either one layout for the whole batch or one layout
per sample of the batch.

Hot-path rule: forward never synchronizes with the GPU to find out what to compute. A layout set from the host
(`set_levels`) lives on the CPU as a numpy array; one decided inside a pass (`set_levels_on_device`, foqlens.layerwise)
stays on the card and is read back only when someone asks for it. A mixed layout computes the output once per level in
use and
selects rows with torch.where. A layout of depths and ZERO over a k-quant copy is read by the kernel instead
(kernels/kquant.py), every block of rows straight from the copy's bytes to its depth; an input longer than
KERNEL_MAX_TOKENS - a prefill - multiplies in one GEMM by the copy the kernel unpacks, every block to its depth.

Invariants (each one has a test):
- Invariant: an all-bf16 layout is bit-exact with the original nn.Linear.
- Invariant: the rows of a block depend only on that block's level.
- Invariant: with per-sample layouts, sample b is bit-exact with sample b of the same batch run
  under layout b for every sample - a per-sample layout changes nothing but the selection.
- Invariant: inside `samples(model, part)` the samples `part` of the batch read their own layouts,
  bit-exact with the same inputs under those layouts set alone.
- Invariant: forward never reads levels from the GPU.
- Invariant: a packed copy exists only for a level some layout has used - a bench that only
  reads bf16 and ZERO holds no int8 or nf4 copy.
- Invariant: drop_bf16 changes no output of a read depth; afterwards the module holds its refined
  copy alone, and a level it cannot read is refused when set, not when computed.
- Invariant: a resident module (MixedPrecisionLinear.resident), built from a copy with no bf16 weight ever loaded,
  reads every depth exactly as a module built from bf16 reads it after drop_bf16, and refuses a depth its copy does
  not hold.
- Invariant: with depth caps a block stores only the depths up to its cap, reads within its cap
  exactly as before, and a read deeper than its cap is refused when set. A layout set on the card names its ladder
  instead of its blocks, so it is refused as a whole when any level of that ladder is deeper than some block stores.
- Invariant: the same codes through `set_levels_on_device` and through `set_levels` give the same output of forward.
- Invariant: a baked level is bit-exact with reading the same depth from the refined copy; afterwards
  the module holds that weight alone and reads that level alone - bf16 is refused when set.
- Invariant: a weight baked from another source (bake_weight) is read as given at its level, bit-exact
  with F.linear on that weight; setting the level again does not quantize it again.
- Invariant: a layout read by the kernel gives the unpacked output within a bf16 step of the largest output, and a
  token's output does not depend on the other tokens of the batch; a baked level and bf16 never go to the kernel.
- Invariant: a layout read by the kernel on an input past KERNEL_MAX_TOKENS is bit-exact with F.linear over the weight
  read block by block to its depth in torch, and a sample of a per-sample layout reads its own layout - also when the
  batch holds fewer levels than samples and the copy is unpacked once per level instead of once per sample.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
import functools
from functools import partial
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from foqlens.kernels.kquant import TILE_ROWS as KERNEL_TILE_ROWS, kernel_reads_format, kquant_matmul, kquant_unpack
from foqlens.refinements import KQuantLadder, KRefinedWeight
from foqlens.model import text_layers
from foqlens.quant import MAX_DEPTH, DEPTH_BITS, CappedRefinedWeight, Int8Weight, Level, Nf4Weight, RefinedWeight, _DepthReader

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
# Where a level reads from: a packed copy of its own, or the one refined copy every read depth
# shares. bf16 reads the weight itself, ZERO reads nothing.
STORAGE = {Level.INT8: Int8Weight, Level.NF4: Nf4Weight} | {lv: RefinedWeight for lv in Level if lv.depth}
# The depth a level reads, by code; the levels that are not read depths count as the full copy.
DEPTH_BY_CODE = np.array([lv.depth if lv.depth or lv is Level.ZERO else MAX_DEPTH for lv in Level], dtype=np.uint8)
DEEPEST = max((lv for lv in Level if lv.depth), key=lambda lv: lv.depth)  # the deepest read depth, D8
# A layout of depths and ZERO over a k-quant copy is read by the kernel, straight from the copy's bytes, up to this many
# tokens; past it the kernel unpacks the copy, every block to its depth, for one GEMM (kquant_unpack). On a 12288x1536
# module at D8 the two meet at 32 tokens, 0.138 against 0.139 ms, and at 256 the GEMM is 3.6x faster
# (scripts/kernel_speed.py); a step of E2B-it at a mixed layout takes 28.4 ms at a batch of 64 and 40.0 at 128 with the
# threshold at 32, against 42.5 and 70.2 at 256, and 27.0 at 32 either way (scripts/decode_step_speed.py, 2026-09-19).
# Off, every read unpacks in torch.
KERNEL = True
KERNEL_MAX_TOKENS = 32


@functools.cache
def _depth_table(device: torch.device) -> torch.Tensor:
    """uint8 by level code: the depth the kernel reads a block to, 0 for ZERO."""
    return torch.as_tensor(DEPTH_BY_CODE, device=device)


class RefinedCopy(Protocol):
    """How a module's refined copy is built: the module's name and bf16 weight in, a copy read at every depth out."""

    def quantize(self, name: str, weight: torch.Tensor) -> _DepthReader: ...


class WeightSource(Protocol):
    """Where a baked level takes its weight from: the module's name and bf16 weight in, the weight read at `level` out."""

    def read(self, name: str, weight: torch.Tensor, level: Level) -> torch.Tensor: ...


class MixedPrecisionLinear(nn.Module):
    """An nn.Linear whose every block of output rows is read at its own precision."""

    def __init__(self, linear: nn.Linear, block_rows: int = DEFAULT_BLOCK_ROWS,
                 copy: Callable[[torch.Tensor], _DepthReader] = RefinedWeight.quantize):
        super().__init__()
        self._copy = copy  # builds the refined copy every read depth shares, from the bf16 weight
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.block_rows = block_rows
        self.n_blocks = math.ceil(self.out_features / block_rows)
        self.weight = linear.weight
        self.weight.requires_grad_(False)
        self.bias = linear.bias
        self._device = linear.weight.device
        self._native = Level.BF16  # the level the weight itself holds: bf16, or the read depth baked into it
        self._packed: dict[type, Int8Weight | Nf4Weight | RefinedWeight] = {}
        self.readable: frozenset[Level] = frozenset(Level)
        self._caps = np.full(self.n_blocks, MAX_DEPTH, dtype=np.uint8)  # the depth every block stores
        self._shallowest_cap = MAX_DEPTH  # the depth every block is sure to store; a layout on the card is held to it
        self._samples = slice(None)  # the samples of a per-sample layout the next forwards read
        self._codes: torch.Tensor | None = None  # the layout on the card, when it was set there
        self._shape: tuple[int, ...] = (self.n_blocks,)
        self.set_levels(Level.BF16)

    @property
    def levels(self) -> np.ndarray:
        """Level code per block: [n_blocks], or [batch, n_blocks] for per-sample layouts. A copy.

        A layout set on the device is read back here and nowhere else: the pass itself never waits for the card. It is
        read afresh every time and never kept - a replayed graph decides it again without any Python running, so a
        kept copy would answer for the layout of some earlier step.
        """
        if self._levels is None:
            return self._codes.to("cpu", torch.uint8).numpy()
        return self._levels.copy()

    @property
    def storages(self) -> tuple[type, ...]:
        """Kinds of quantized copies of the weight that are held, in the order they were first needed."""
        return tuple(self._packed)

    @property
    def refined(self) -> _DepthReader | None:
        """The refined copy every read depth shares, built on the first need as a read of a depth builds it (from the
        model's file, or quantized from bf16); None for a module that reads a baked level and holds no copy."""
        if RefinedWeight not in self._packed and self._native is Level.BF16 and self.weight is not None:
            self._materialize(DEEPEST)
        return self._packed.get(RefinedWeight)

    def _materialize(self, level: Level) -> None:
        """Quantize the weight into the level's storage on its first use; bf16, ZERO and a baked level need no copy."""
        kind = STORAGE.get(level)
        if kind is not None and kind not in self._packed and level is not self._native:
            self._packed[kind] = (self._copy if kind is RefinedWeight else kind.quantize)(self.weight.data)

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
        if (DEPTH_BY_CODE[arr] > self._caps).any():
            raise ValueError("a block is read deeper than the depth it stores")
        self._levels = arr.copy()
        self._shape = arr.shape
        self._used = used
        for level in self._used:
            self._materialize(level)
        self._codes = torch.as_tensor(arr, device=self._device) if codes is None else codes
        self._rows = None  # built on the first read that needs it (_row_layout)
        self._depths = self._block_depths(self._codes) if self._kernel_reads() else None

    def set_levels_on_device(self, codes: torch.Tensor, used: tuple[Level, ...]) -> None:
        """Levels that stay on the card: `codes` [n_blocks] or [batch, n_blocks] of level codes, and the levels the
        layout may hold, named by the caller instead of read off the card.

        A hot path decides a layout inside the pass (foqlens.layerwise), and reading the codes back to name the levels
        used would stop the pipeline at every module. The caller states the ladder once; every level of it is
        materialized whether or not this layout reaches it, and the checks that read values are left to `set_levels`.

        Invariant: the same codes through this and through `set_levels` give the same output of `forward`.
        """
        if codes.ndim not in (1, 2) or codes.shape[-1] != self.n_blocks:
            raise ValueError(f"levels of shape {tuple(codes.shape)} for {self.n_blocks} blocks")
        if not self.readable.issuperset(used):
            raise ValueError(f"levels {[lv.name for lv in used if lv not in self.readable]} are not held by this module")
        # which block holds which code is on the card, so the depth a block stores is held against the whole ladder:
        # the caller may lay any of its levels on any block
        if max((int(DEPTH_BY_CODE[int(level)]) for level in used), default=0) > self._shallowest_cap:
            raise ValueError("a level of the ladder is read deeper than a block of this module stores")
        self._levels, self._shape, self._used = None, tuple(codes.shape), tuple(used)
        for level in self._used:
            self._materialize(level)
        self._codes = codes
        self._rows = None  # a layout the kernel reads never needs the row codes, and this one is set every pass
        self._depths = self._block_depths(codes) if self._kernel_reads() else None

    def _kernel_reads(self) -> bool:
        """Whether the layout is read by the kernel: a k-quant copy on a base it reads, every level a depth or ZERO,
        blocks of its rows."""
        copy = self._packed.get(RefinedWeight)
        return (KERNEL and self._native is Level.BF16 and self.block_rows == KERNEL_TILE_ROWS
                and isinstance(copy, KRefinedWeight) and kernel_reads_format(copy.fmt)
                and all(level is Level.ZERO or level.depth for level in self._used))

    def _block_depths(self, codes: torch.Tensor) -> torch.Tensor:
        """The depth every block is read to on the device, [n_blocks] or [batch, n_blocks]; ZERO is 0."""
        return _depth_table(self._device)[codes.long()]

    def _row_layout(self) -> torch.Tensor | None:
        """The level code per output row of the current layout, or None where one level covers the module.

        Built on the first read that needs it and kept until the next layout is set: a layout the kernel reads never
        asks for it, and a regulator sets a layout in front of every layer of every pass (foqlens.layerwise).
        """
        if self._rows is None and len(self._used) > 1:
            self._rows = self._row_codes(self._codes)
        return self._rows

    def _row_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Level code per output row on the GPU: [out] or [batch, 1, out], broadcast over tokens.

        Blocks are expanded to rows on the device: one small copy of block codes, not of row codes.
        """
        rows = codes.repeat_interleave(self.block_rows, dim=-1)[..., : self.out_features]
        return rows if codes.ndim == 1 else rows.unsqueeze(1)

    def block_sizes(self) -> np.ndarray:
        """Number of rows in every block; the last one may be partial."""
        sizes = np.full(self.n_blocks, self.block_rows, dtype=np.int64)
        sizes[-1] = self.out_features - self.block_rows * (self.n_blocks - 1)
        return sizes

    # What a module still reads after drop_bf16: the depths of its refined copy, and nothing.
    RESIDENT = frozenset({Level.ZERO, *(lv for lv in Level if lv.depth)})

    @classmethod
    def resident(cls, linear: nn.Linear, copy: _DepthReader, device: str,
                 block_rows: int = DEFAULT_BLOCK_ROWS) -> MixedPrecisionLinear:
        """A module that holds `copy` alone from the start, reading its deepest depth: the state drop_bf16 leaves,
        with no bf16 weight ever loaded. `linear` gives the shape and the bias; its weight may be on meta."""
        module = cls(linear, block_rows, lambda weight: copy)  # the copy is given, never built from the weight
        module._device = torch.device(device)  # the weight may be on meta; the rows are read where the copy lies
        module.set_levels(DEEPEST)
        module.drop_bf16()
        # a copy cut short (scripts/cut_model.py --depth) refuses what it does not hold, instead of reading shallower
        module.readable = frozenset(lv for lv in module.readable if lv.depth <= copy.depth)
        module.set_levels(max(module.readable, key=lambda lv: lv.depth))
        return module

    def drop_bf16(self) -> None:
        """Keep only the refined copy: the bf16 weight and every other copy leave the GPU.

        Afterwards the module reads ZERO and the read depths only; the current layout must be one of those.
        """
        if not self.RESIDENT.issuperset(self._used):
            raise ValueError("switch to ZERO or read depths before dropping bf16")
        self._materialize(Level.D8)
        self._packed = {RefinedWeight: self._packed[RefinedWeight]}
        self.weight = None
        self.readable = self.RESIDENT

    def read_weight(self, level: Level) -> torch.Tensor:
        """The dense weight every block reads at `level` (a read depth), in the weight's dtype - for a signal that needs
        what a level costs a block (its quantization error), not for the forward pass."""
        if not level.depth:
            raise ValueError(f"{level.name} is not a read depth of the refined copy")
        self._materialize(level)
        return self._packed[RefinedWeight].dequantize(self.weight.dtype, level.depth)

    def bake(self, level: Level) -> None:
        """Read one depth at the cost of bf16: the weight read to it replaces the bf16 weight and every copy.

        A uniform level then unpacks once instead of on every call. Afterwards the module reads that
        level alone; the bf16 weight is gone, so asking for bf16 is refused rather than answered wrongly.
        """
        self._check_bakeable(level)
        self._materialize(level)
        self.bake_weight(self._packed[RefinedWeight].dequantize(self.weight.dtype, level.depth), level)

    def bake_weight(self, weight: torch.Tensor, level: Level) -> None:
        """As bake, with the weight read to `level` given by another source of weights (WeightSource).

        The module then reads `weight` as it is whenever `level` is set - it is never quantized again.
        """
        self._check_bakeable(level)
        if weight.shape != self.weight.shape:
            raise ValueError(f"a weight of shape {tuple(weight.shape)} for {tuple(self.weight.shape)}")
        self.weight = nn.Parameter(weight.to(self.weight.device, self.weight.dtype), requires_grad=False)
        self._packed = {}
        self._native = level
        self.readable = frozenset({level})
        self.set_levels(level)

    def _check_bakeable(self, level: Level) -> None:
        if not level.depth:
            raise ValueError(f"only a read depth is baked, not {level.name}")
        if self._native is not Level.BF16 or self.weight is None:
            raise ValueError("a level is baked once, from the bf16 weight")

    def set_caps(self, caps: np.ndarray) -> None:
        """Store every block only to its depth cap (0 ... MAX_DEPTH); the deeper depths leave the GPU.

        Only on a resident module (after drop_bf16), once, from its full refined copy; the current
        layout must not read any block deeper than its new cap.
        """
        caps = np.asarray(caps, dtype=np.uint8)
        if self.weight is not None:
            raise ValueError("depth caps need a resident module: call drop_bf16 first")
        if caps.shape != (self.n_blocks,) or (caps > MAX_DEPTH).any():
            raise ValueError(f"caps of shape {caps.shape} for {self.n_blocks} blocks, each 0 ... {MAX_DEPTH}")
        full = self._packed[RefinedWeight]
        if not isinstance(full, RefinedWeight):
            raise ValueError("depth caps are set once, from the full refined copy")
        if (DEPTH_BY_CODE[self.levels] > caps).any():
            raise ValueError("the current layout reads a block deeper than its new cap")
        block_caps = torch.as_tensor(caps, device=self._device)
        self._packed[RefinedWeight] = CappedRefinedWeight.from_full(full, block_caps, self.block_rows)
        self._caps = caps.copy()
        self._shallowest_cap = int(caps.min())

    @property
    def caps(self) -> np.ndarray:
        """The depth every block stores. A copy."""
        return self._caps.copy()

    def stored_bytes(self) -> int:
        """Bytes of the refined copy held, without its scales: every depth, or only those under the caps."""
        store = self._packed.get(RefinedWeight)
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
            return store.matmul(x, self.bias, level.depth) if level.depth else store.matmul(x, self.bias)
        out = x.new_zeros(*x.shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias

    def read_samples(self, part: slice) -> None:
        """Per-sample layouts serve only the samples `part` of the batch until set back to slice(None)."""
        self._samples = part

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._depths is not None:
            return self._kernel_forward(x) if x.numel() // x.shape[-1] <= KERNEL_MAX_TOKENS else self._unpacked_forward(x)
        rows = self._row_layout()
        if rows is None:
            return self.output_at(self._used[0], x)
        if len(self._shape) == 2:
            read = range(*self._samples.indices(self._shape[0]))
            if x.shape[0] != len(read):
                raise ValueError(f"batch of {x.shape[0]} for per-sample layouts of {len(read)}")
            rows = rows[self._samples]
        outputs = self._outputs_at(self._used, x)
        out = outputs[self._used[0]]
        for level in self._used[1:]:
            out = torch.where(rows == int(level), outputs[level], out)
        return out

    def _kernel_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The output read by the kernel straight from the copy, every block to its depth (kernels/kquant.py)."""
        depths = self._depths
        if depths.ndim == 2:
            depths = depths[self._samples]
            if x.shape[0] != depths.shape[0]:
                raise ValueError(f"batch of {x.shape[0]} for per-sample layouts of {depths.shape[0]}")
            # every token of a sample reads its sample's layout
            depths = depths[:, None, :].expand(-1, x[0, ..., 0].numel(), -1).reshape(-1, self.n_blocks)
        out = kquant_matmul(self._packed[RefinedWeight], x, depths)
        return out if self.bias is None else out + self.bias

    def _unpacked_forward(self, x: torch.Tensor) -> torch.Tensor:
        """A long input times the copy unpacked by the kernel, every block to its depth, in one GEMM; under per-sample
        layouts one unpacking and one GEMM per sample, or per level in use when there are fewer levels than samples."""
        copy = self._packed[RefinedWeight]
        if self._depths.ndim == 1:
            return F.linear(x, kquant_unpack(copy, self._depths), self.bias)
        depths = self._depths[self._samples]
        if x.shape[0] != depths.shape[0]:
            raise ValueError(f"batch of {x.shape[0]} for per-sample layouts of {depths.shape[0]}")
        if len(self._used) < depths.shape[0]:
            return self._per_level_forward(x, copy)
        return torch.stack([F.linear(sample, kquant_unpack(copy, d), self.bias) for sample, d in zip(x, depths)])

    def _per_level_forward(self, x: torch.Tensor, copy: KRefinedWeight) -> torch.Tensor:
        """Per-sample layouts over few levels - many variants of one prompt: the whole batch times the copy unpacked
        once per level, every sample's rows selected from its own level (a row depends only on its block's level)."""
        table = _depth_table(self._device)
        outputs = [F.linear(x, kquant_unpack(copy, table[torch.full((self.n_blocks,), int(level), device=self._device)]),
                            self.bias) for level in self._used]
        if self._row_layout() is None:
            return outputs[0]
        rows = self._rows[self._samples]
        out = outputs[0]
        for level, output in zip(self._used[1:], outputs[1:]):
            out = torch.where(rows == int(level), output, out)
        return out

    def _outputs_at(self, levels: tuple[Level, ...], x: torch.Tensor) -> dict[Level, torch.Tensor]:
        """output_at for every level; several read depths come from one accumulation of the refined copy."""
        depths = [lv for lv in levels if lv.depth]
        outputs = {lv: self.output_at(lv, x) for lv in levels if not lv.depth}
        if len(depths) < 2:
            return outputs | {lv: self.output_at(lv, x) for lv in depths}
        self._materialize(depths[0])
        store = self._packed[RefinedWeight]
        return outputs | dict(zip(depths, store.linear_at_depths(x, self.bias, [lv.depth for lv in depths])))

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
        """Every module keeps only its refined copy (MixedPrecisionLinear.drop_bf16); the cache returns the freed memory."""
        for module in self.modules.values():
            module.drop_bf16()
        torch.cuda.empty_cache()

    def bake(self, level: Level) -> None:
        """Every module reads `level` from a weight unpacked once (MixedPrecisionLinear.bake); the cache returns the freed memory."""
        for module in self.modules.values():
            module.bake(level)
        torch.cuda.empty_cache()

    def bake_from(self, source: WeightSource, level: Level) -> None:
        """Every module reads `level` from the weight `source` gives for it, in place of its own refined copy."""
        for name, module in self.modules.items():
            module.bake_weight(source.read(name, module.weight.data, level), level)
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
        """Mean stored bits per weight of the refined copies, weighted by weight count: DEPTH_BITS per depth kept, no scales."""
        total_bits, total_weights = 0.0, 0
        for m in self.modules.values():
            total_bits += float((m.caps.astype(np.int64) * DEPTH_BITS * m.block_sizes() * m.in_features).sum())
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


@contextmanager
def samples(model: nn.Module, part: slice) -> Iterator[None]:
    """The model's per-sample layouts serve only the samples `part` of the batch: a pass over some rows of it,
    as a prefill in chunks of rows. A model without controlled modules is left as it is."""
    modules = [m for m in model.modules() if isinstance(m, MixedPrecisionLinear)]
    for module in modules:
        module.read_samples(part)
    try:
        yield
    finally:
        for module in modules:
            module.read_samples(slice(None))


def _resolve(parent: nn.Module, dotted: str) -> tuple[nn.Module, str] | None:
    *path, leaf = dotted.split(".")
    for part in path:
        parent = getattr(parent, part, None)
        if parent is None:
            return None
    child = getattr(parent, leaf, None)
    return (parent, leaf) if isinstance(child, nn.Linear) else None


@torch.no_grad()
@dataclass(frozen=True)
class SymmetricCopy:
    """quant.RefinedWeight for every module: the copy depth caps are cut from (CappedRefinedWeight has no k-quant form yet)."""

    def quantize(self, name: str, weight: torch.Tensor) -> RefinedWeight:
        return RefinedWeight.quantize(weight)


# The bench's one copy (E002): a k-quant base with refinements, the sensitive classes on Q4_K.
BENCH_COPY = KQuantLadder()


def install(model: nn.Module, block_rows: int = DEFAULT_BLOCK_ROWS, copy: RefinedCopy = BENCH_COPY) -> Controller:
    """Replace the text decoder's linear modules with controlled ones. Everything starts in bf16.

    `copy` builds each module's refined copy by its name; the bench's own is the k-quant ladder.
    """
    return _replace(model, lambda name, linear: MixedPrecisionLinear(linear, block_rows, partial(copy.quantize, name)))


def install_resident(model: nn.Module, copy_of: Callable[[str], _DepthReader], device: str,
                     block_rows: int = DEFAULT_BLOCK_ROWS) -> Controller:
    """As install, with every module holding the refined copy `copy_of` gives for its name and no bf16 weight at all -
    drop_bf16's state from the start. The model's linear weights need not exist: they may still be on meta."""
    return _replace(model, lambda name, linear: MixedPrecisionLinear.resident(linear, copy_of(name), device, block_rows))


def _replace(model: nn.Module, make: Callable[[str, nn.Linear], MixedPrecisionLinear]) -> Controller:
    modules: dict[str, MixedPrecisionLinear] = {}
    for i, layer in enumerate(text_layers(model)):
        for dotted in CONTROLLED:
            found = _resolve(layer, dotted)
            if found is None:
                continue
            parent, leaf = found
            name = f"layers.{i}.{dotted}"
            mixed = make(name, getattr(parent, leaf))
            setattr(parent, leaf, mixed)
            modules[name] = mixed
    return Controller(modules)
