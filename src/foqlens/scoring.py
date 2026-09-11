"""Step 1: the naive per-block score, as preregistered and extended by prereg/ADDENDUM-01.md.

For one query, one forward pass in bf16 with eager attention gives three mask vectors - one per
way of choosing where the query's meaning sits:

- "norm" (variant A, the preregistered one): centers are the top-k tokens by hidden-state norm
  at the middle decoder layer;
- "pooled" (variant B): every token;
- "attention" (variant C): centers are the top-k tokens by the attention they receive at the
  middle decoder layer, normalized by how many queries the causal mask lets look at them.

The first token (<bos>) is never used. The score of a block of output rows of a linear module is
the L2 norm of that block's output slice, averaged over the chosen tokens; the mask vector is the
block scores of all controlled modules in a fixed order. Background subtraction (the prepared
next move) is centering these vectors by their mean over queries, done in foqlens.separation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from foqlens import model as fm
from foqlens.precision import MixedPrecisionLinear

DEFAULT_TOP_K = 4
MODES = ("norm", "pooled", "attention")


def center_positions(token_scores: torch.Tensor, k: int = DEFAULT_TOP_K) -> torch.Tensor:
    """Indices of the k highest-scoring tokens, [seq] -> [k] sorted, never position 0."""
    scores = token_scores.float().clone()
    scores[0] = -math.inf
    k = min(k, scores.shape[0] - 1)
    return torch.topk(scores, k).indices.sort().values


def attention_received(attn: torch.Tensor) -> torch.Tensor:
    """Mean attention each key token receives, [heads, seq, seq] -> [seq].

    Summed over heads and queries, then divided by heads times the number of queries the causal
    mask allows to attend to that key (seq - j), so early tokens do not win by position alone.
    """
    heads, seq, _ = attn.shape
    received = attn.float().sum(dim=(0, 1))
    allowed = torch.arange(seq, 0, -1, device=attn.device, dtype=torch.float32)
    return received / (heads * allowed)


def token_weights(mode: str, hidden: torch.Tensor, attn: torch.Tensor | None, k: int = DEFAULT_TOP_K) -> torch.Tensor:
    """Weights over tokens [seq] summing to 1: how much each token counts in the block scores."""
    seq = hidden.shape[0]
    weights = torch.zeros(seq, device=hidden.device)
    if mode == "pooled":
        weights[1:] = 1.0 / (seq - 1)
        return weights
    if mode == "norm":
        token_scores = hidden.float().norm(dim=-1)
    elif mode == "attention":
        if attn is None:
            raise ValueError("attention mode needs attention weights: load the model with attn_implementation='eager'")
        token_scores = attention_received(attn)
    else:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    centers = center_positions(token_scores, k)
    weights[centers] = 1.0 / centers.numel()
    return weights


def block_scores(output: torch.Tensor, weights: torch.Tensor, block_rows: int) -> torch.Tensor:
    """Scores of the row blocks of one module: output [seq, out] -> [n_blocks], weighted mean over tokens of the block L2 norm."""
    out = output.float()
    seq, n_out = out.shape
    n_blocks = math.ceil(n_out / block_rows)
    padded = torch.zeros(seq, n_blocks * block_rows, device=out.device)
    padded[:, :n_out] = out
    per_token = padded.view(seq, n_blocks, block_rows).norm(dim=-1)  # [seq, n_blocks]
    return weights.to(per_token.device) @ per_token


def gini(x: np.ndarray) -> float:
    """Gini coefficient of non-negative scores: 0 for a flat mask, close to 1 when all mass sits in one block."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def normalized_entropy(x: np.ndarray) -> float:
    """Entropy of the score distribution over blocks divided by its maximum: 1 for a flat mask, 0 for one block."""
    p = np.asarray(x, dtype=np.float64)
    p = p / p.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(p.size))


class BlockScorer:
    """Collects module outputs with forward hooks and turns one query into its mask vectors."""

    def __init__(self, modules: dict[str, MixedPrecisionLinear], top_k: int = DEFAULT_TOP_K):
        self.modules = modules
        self.top_k = top_k
        self._outputs: dict[str, torch.Tensor] = {}

    @property
    def n_blocks(self) -> int:
        return sum(m.n_blocks for m in self.modules.values())

    def _hook(self, name: str):
        def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> None:
            self._outputs[name] = output[0].detach()

        return hook

    @torch.no_grad()
    def score(self, model: nn.Module, tokenizer, text: str, modes: tuple[str, ...] = MODES) -> dict[str, tuple[np.ndarray, list[int]]]:
        """For every mode: the mask vector [n_blocks] and the token positions it was built on."""
        handles = [m.register_forward_hook(self._hook(n)) for n, m in self.modules.items()]
        try:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            out = model(**inputs, output_hidden_states=True, output_attentions="attention" in modes)
        finally:
            for h in handles:
                h.remove()
        layer = len(fm.text_layers(model)) // 2
        hidden = out.hidden_states[layer + 1][0]
        attn = out.attentions[layer][0] if out.attentions is not None else None
        result = {}
        for mode in modes:
            weights = token_weights(mode, hidden, attn, self.top_k)
            parts = [block_scores(self._outputs[n], weights, m.block_rows) for n, m in self.modules.items()]
            result[mode] = (torch.cat(parts).cpu().numpy(), torch.nonzero(weights).flatten().tolist())
        self._outputs.clear()
        return result
