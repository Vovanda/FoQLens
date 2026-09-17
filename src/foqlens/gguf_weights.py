"""The controlled weights of a published GGUF file, dequantized: a WeightSource to bake a level from.

Infrastructure: it reads a file. The embeddings and norms stay as the bench loaded them; only the controlled
matrices (precision.CONTROLLED) come from the file, so a GGUF level and a level of our own ladder differ in those
matrices alone.

Invariant: read(name, weight, level) equals gguf-py's dequantizer on the file's tensor for that module, shaped as
the weight, whatever the level.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

# The bench's module class -> the llama.cpp tensor name inside a block (gemma3n / gemma4 conversion).
GGUF_NAMES = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
              "mlp.down_proj": "ffn_down", "per_layer_input_gate": "inp_gate", "per_layer_projection": "proj"}
_MODULE = re.compile(r"layers\.(\d+)\.(.+)$")


class GgufWeights:
    """The dequantized controlled matrices of one GGUF file, read lazily per module."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._tensors = {t.name: t for t in GGUFReader(self.path).tensors}

    def tensor_name(self, name: str) -> str:
        layer, kind = _MODULE.search(name).groups()
        return f"blk.{layer}.{GGUF_NAMES[kind]}.weight"

    def read(self, name: str, weight: torch.Tensor, level) -> torch.Tensor:
        t = self._tensors[self.tensor_name(name)]
        array = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(weight.shape)
        return torch.from_numpy(array).to(weight.device, weight.dtype)
