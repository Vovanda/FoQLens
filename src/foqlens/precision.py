"""Precision controller: bit depth per layer, module and block of weight rows.

Every linear module of the text decoder is replaced by a MixedPrecisionLinear. It keeps the
original bf16 weight and its packed int8 and nf4 copies, and the level is set per block of
block_rows output rows. A block of rows is the unit of address: "where to sharpen" is given by it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

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


class MixedPrecisionLinear(nn.Module):
    """An nn.Linear whose every block of output rows is read at its own precision."""

    def __init__(self, linear: nn.Linear, block_rows: int = DEFAULT_BLOCK_ROWS):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.block_rows = block_rows
        self.weight = linear.weight
        self.weight.requires_grad_(False)
        self.bias = linear.bias
        self.int8 = Int8Weight.quantize(self.weight.data)
        self.nf4 = Nf4Weight.quantize(self.weight.data)
        n_blocks = math.ceil(self.out_features / block_rows)
        self.register_buffer(
            "levels", torch.full((n_blocks,), int(Level.BF16), dtype=torch.uint8, device=self.weight.device)
        )

    @property
    def n_blocks(self) -> int:
        return self.levels.numel()

    def block_sizes(self) -> torch.Tensor:
        """Number of rows in every block; the last one may be partial."""
        sizes = torch.full((self.n_blocks,), self.block_rows, dtype=torch.long)
        sizes[-1] = self.out_features - self.block_rows * (self.n_blocks - 1)
        return sizes

    def dequantized(self, level: Level) -> torch.Tensor:
        if level is Level.BF16:
            return self.weight
        if level is Level.INT8:
            return self.int8.dequantize(self.weight.dtype)
        return self.nf4.dequantize(self.weight.dtype)

    def effective_weight(self) -> torch.Tensor:
        """The weight forward actually computes with under the current level layout."""
        used = torch.unique(self.levels).tolist()
        if len(used) == 1:
            return self.dequantized(Level(used[0]))
        rows = self.levels.repeat_interleave(self.block_rows)[: self.out_features]
        weight = self.weight.clone()
        for code in used:
            if code == Level.BF16:
                continue
            mask = rows == code
            weight[mask] = self.dequantized(Level(code))[mask]
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, blocks={self.n_blocks}x{self.block_rows}"


class Controller:
    """Precision layout over the bench: set levels and read back what is actually in place."""

    def __init__(self, modules: dict[str, MixedPrecisionLinear]):
        self.modules = modules

    def names(self, layer: int | None = None) -> list[str]:
        if layer is None:
            return list(self.modules)
        prefix = f"layers.{layer}."
        return [n for n in self.modules if n.startswith(prefix)]

    def set_all(self, level: Level) -> None:
        for module in self.modules.values():
            module.levels.fill_(int(level))

    def set_layer(self, layer: int, level: Level) -> None:
        names = self.names(layer)
        if not names:
            raise KeyError(f"no layer {layer}")
        for name in names:
            self.modules[name].levels.fill_(int(level))

    def set_module(self, name: str, level: Level) -> None:
        self.modules[name].levels.fill_(int(level))

    def set_blocks(self, name: str, blocks: Iterable[int], level: Level) -> None:
        idx = torch.as_tensor(list(blocks), dtype=torch.long, device=self.modules[name].levels.device)
        self.modules[name].levels[idx] = int(level)

    def layout(self) -> dict[str, list[int]]:
        """Level of every block of every module - the actual state of the buffers."""
        return {name: m.levels.tolist() for name, m in self.modules.items()}

    def mean_bits(self) -> float:
        """Mean nominal bits per weight over the controlled modules, weighted by weight count.

        Nominal: 16 / 8 / 4 without the overhead of group scales. Embeddings, lm_head and
        per_layer_model_projection are not controlled and do not enter the mean.
        """
        bits_by_code = torch.tensor([int(lv.bits) for lv in Level], dtype=torch.float64)
        total_bits = 0.0
        total_weights = 0
        for m in self.modules.values():
            weights_per_block = m.block_sizes().to(torch.float64) * m.in_features
            total_bits += float((weights_per_block * bits_by_code[m.levels.long().cpu()]).sum())
            total_weights += m.in_features * m.out_features
        return total_bits / total_weights


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
