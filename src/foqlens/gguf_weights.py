"""The controlled weights of a published GGUF file: dequantized, a WeightSource to bake a level from, or as their
k-quant blocks, the base of a stack (refinements.PublishedBaseLadder).

Infrastructure: it reads a file. The embeddings and norms stay as the bench loaded them; only the controlled
matrices (precision.CONTROLLED) come from the file, so a GGUF level and a level of our own ladder differ in those
matrices alone.

Invariant: read(name, weight, level) equals gguf-py's dequantizer on the file's tensor for that module, shaped as
the weight, whatever the level.
Invariant: base_blocks gives a k-quant tensor's bytes as they lie in the file; read as a base they equal read().
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from huggingface_hub import hf_hub_download

from foqlens.kquant import FORMATS, QK_K, KFormat

# The bench's module class -> the llama.cpp tensor name inside a block (gemma3n / gemma4 conversion).
GGUF_NAMES = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
              "mlp.down_proj": "ffn_down", "per_layer_input_gate": "inp_gate", "per_layer_projection": "proj"}
_MODULE = re.compile(r"layers\.(\d+)\.(.+)$")

# Published GGUF files of E2B-it the bench compares with or builds on, pinned like the checkpoints (model.REVISIONS):
# name -> (repository, revision, file). The Q2_K of bartowski is k-quant throughout, so every controlled tensor of it
# is a base the stack refines; unsloth's UD-Q2_K_XL holds 40 of them in IQ types, which have no block step.
PUBLISHED = {
    "bartowski-Q2_K": ("bartowski/google_gemma-4-E2B-it-GGUF", "81012ba3538e061d5ee003f11f25335b17f82e2d",
                       "google_gemma-4-E2B-it-Q2_K.gguf"),
    "unsloth-UD-Q2_K_XL": ("unsloth/gemma-4-E2B-it-GGUF", "0314792d7f1f7e229411f620751375812bb9faf2",
                           "gemma-4-E2B-it-UD-Q2_K_XL.gguf"),
    "unsloth-Q4_K_M": ("unsloth/gemma-4-E2B-it-GGUF", "0314792d7f1f7e229411f620751375812bb9faf2",
                       "gemma-4-E2B-it-Q4_K_M.gguf"),
}


def published_path(name: str) -> Path:
    """The local path of a PUBLISHED file, downloaded at its pinned revision if it is not in the cache yet."""
    repository, revision, file = PUBLISHED[name]
    return Path(hf_hub_download(repository, file, revision=revision))


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

    def base_blocks(self, name: str, shape: tuple[int, int]) -> tuple[KFormat, torch.Tensor] | None:
        """The module's tensor as a base for the stack: its k-quant format and ggml blocks [out, super-blocks, bytes]
        as they lie in the file, or None for a type the stack cannot refine (float, or IQ with no block step).
        A refinements.BaseBlocks."""
        t = self._tensors[self.tensor_name(name)]
        fmt = FORMATS.get(t.tensor_type.name)
        if fmt is None:
            return None
        out, inp = shape
        blocks = np.ascontiguousarray(t.data, dtype=np.uint8).reshape(out, inp // QK_K, -1)
        return fmt, torch.from_numpy(blocks.copy())
