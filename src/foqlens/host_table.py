"""The per-layer embedding table of Gemma 4 kept in the host's memory: 4.37 GiB of E2B-it off the card (#27).

Gemma 4 looks every token up in a second table, one row of num_layers x 256 per token (embed_tokens_per_layer,
262,144 x 8,960 bf16 for E2B-it). The regulator never reads it, and a decoding step needs only the rows of its tokens:
32 tokens take 573 KB. So the table lies in pinned host memory and kernels/host_gather.cu fetches those rows over
PCIe; under unified addressing the kernel reads the host pointer directly, so the lookup stays inside a CUDA graph.

Invariant: HostEmbedding's output equals Gemma4TextScaledWordEmbedding's bit for bit - the same bf16 rows times the
same scale in bf16 (tests/test_host_table_gpu.py).
Invariant: the module keeps no copy of the table on the device.
"""

from __future__ import annotations

import torch
from torch import nn

from foqlens.kernels import launch, load

GATHER = "host_gather"
WORD_BYTES = 16  # the kernel copies a row 16 bytes a thread at a time
THREADS = 256  # threads of a block, one block a row: 1,120 words a row of E2B-it


class HostEmbedding(nn.Module):
    """A scaled embedding whose table lies in pinned host memory; forward returns device rows, as the original does."""

    def __init__(self, weight: torch.Tensor, embed_scale: torch.Tensor):
        super().__init__()
        if weight.shape[1] * weight.element_size() % WORD_BYTES:
            raise ValueError(f"a row of {weight.shape[1]} {weight.dtype} is not a whole number of {WORD_BYTES}-byte words")
        self.weight = weight.to("cpu").contiguous().pin_memory()  # not a Parameter: nothing moves it back to the card
        self.embed_scale = nn.Buffer(embed_scale.clone(), persistent=False)

    @classmethod
    def from_embedding(cls, module: nn.Embedding) -> HostEmbedding:
        return cls(module.weight.data, module.embed_scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.reshape(-1).to(torch.long)
        rows = torch.empty(ids.numel(), self.weight.shape[1], dtype=self.weight.dtype, device=input_ids.device)
        words = self.weight.shape[1] * self.weight.element_size() // WORD_BYTES
        launch(load(GATHER), (ids.numel(), 1, 1), (THREADS, 1, 1), self.weight, ids, rows, words)
        return rows.view(*input_ids.shape, -1) * self.embed_scale.to(self.weight.dtype)


def per_layer_table_on_host(model: nn.Module) -> HostEmbedding:
    """Move the text decoder's per-layer embedding table to pinned host memory; returns the module put in its place."""
    decoder = model.model.language_model
    host = HostEmbedding.from_embedding(decoder.embed_tokens_per_layer).to(next(decoder.parameters()).device)
    decoder.embed_tokens_per_layer = host
    torch.cuda.empty_cache()
    return host
