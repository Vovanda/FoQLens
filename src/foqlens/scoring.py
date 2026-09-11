"""Mask sources: how much each block of weight rows matters for a query.

Two instruments, both producing one score per block over all controlled modules, in module order:

- BlockScorer - the naive score of experiments/E001-run1-exploration/ADDENDUM-01.md: the L2 norm of a block's output,
  averaged over the tokens a center mode selects ("norm", "pooled", "attention");
- GradientScorer - gradient x activation of experiments/E002-gradient-score/ADDENDUM-02.md: the first-order estimate of
  the change in the query's own language-model loss if the block's output were zeroed.

Both work on right-padded batches. Background subtraction is done by the caller.

Invariants:
- Invariant: padding and the first token (<bos>) never contribute to a mask.
- Invariant: the same texts in the same batch give identical masks - with sdpa attention too: the
  gradient pass runs attention on GRADIENT_ATTENTION, whose backward is deterministic.
- Invariant (approximate, bf16): a text's mask in a batch points the same way as its mask alone,
  cosine >= 0.99 - batched kernels accumulate differently, which moves token states by up to ~2%.
  The discrete center modes ("norm", "attention") may swap one center of four on near-ties.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from foqlens import model as fm
from foqlens.precision import MixedPrecisionLinear

DEFAULT_TOP_K = 4

# The attention backend of the gradient pass. The default sdpa backend on long sequences is memory-efficient
# attention, whose backward accumulates with atomics: the same batch gave masks differing by up to 9e-3 of the
# largest score (E2B, 8 questions of 131 tokens). The math backend is deterministic, at +5% time and +0.8 GiB peak
# on that batch (RTX 3090 Ti, 2026-09-12). Forward-only passes stay on the default: its forward is deterministic.
GRADIENT_ATTENTION = SDPBackend.MATH
MODES = ("norm", "pooled", "attention")

Scored = dict[str, tuple[np.ndarray, list[int]]]  # mode -> (mask vector, token positions used)


# --- per-token selection -------------------------------------------------------------------


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
    """Weights over the tokens of one unpadded sequence [seq], summing to 1."""
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


def batch_token_weights(
    mode: str, hidden: torch.Tensor, attn: torch.Tensor | None, lengths: list[int], k: int = DEFAULT_TOP_K
) -> torch.Tensor:
    """token_weights for every sequence of a right-padded batch: [batch, seq], zero on padding."""
    out = torch.zeros(hidden.shape[:2], device=hidden.device)
    for b, n in enumerate(lengths):
        a = attn[b, :, :n, :n] if attn is not None else None
        out[b, :n] = token_weights(mode, hidden[b, :n], a, k)
    return out


# --- per-block reductions ------------------------------------------------------------------


def _blocks(values: torch.Tensor, block_rows: int) -> torch.Tensor:
    """[..., out] -> [..., n_blocks, block_rows], zero-padding a partial last block."""
    n_out = values.shape[-1]
    n_blocks = math.ceil(n_out / block_rows)
    return F.pad(values, (0, n_blocks * block_rows - n_out)).unflatten(-1, (n_blocks, block_rows))


def block_norms(output: torch.Tensor, block_rows: int) -> torch.Tensor:
    """L2 norm of every block's output per token: [batch, seq, out] -> [batch, seq, n_blocks]."""
    return _blocks(output.float(), block_rows).norm(dim=-1)


def weighted_block_scores(norms: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean over tokens: norms [batch, seq, n_blocks], weights [batch, seq] -> [batch, n_blocks]."""
    return torch.einsum("bsn,bs->bn", norms, weights)


def block_scores(output: torch.Tensor, weights: torch.Tensor, block_rows: int) -> torch.Tensor:
    """One sequence: output [seq, out], weights [seq] -> [n_blocks]."""
    return weighted_block_scores(block_norms(output[None], block_rows), weights[None])[0]


def taylor_block_scores(
    output: torch.Tensor, grad: torch.Tensor, block_rows: int, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """|sum over valid tokens and block rows of grad * output|: [(batch,) seq, out] -> [(batch,) n_blocks].

    valid [(batch,) seq] marks counted tokens; by default every token except position 0.
    """
    single = output.dim() == 2
    if single:
        output, grad = output[None], grad[None]
        valid = None if valid is None else valid[None]
    if valid is None:
        valid = torch.ones(output.shape[:2], device=output.device)
        valid[:, 0] = 0
    prod = grad.float() * output.float() * valid[..., None].float()
    scores = _blocks(prod, block_rows).sum(dim=(1, 3)).abs()
    return scores[0] if single else scores


def sequence_losses(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy of every sequence of a right-padded batch: [batch]."""
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()
    ce = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), targets.flatten(), reduction="none")
    return (ce.view_as(mask) * mask).sum(dim=1) / mask.sum(dim=1)


# Positions whose vocabulary logits exist at once: 256 x 262k in fp32 is 0.27 GB, against 2+ GB for a whole batch.
LOSS_CHUNK = 256


def head_logits(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """The model's own head on hidden states: lm_head, then the final logit softcapping if the model has one."""
    logits = model.lm_head(hidden)
    cap = model.config.get_text_config().final_logit_softcapping
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


def _chunk_ce(hidden: torch.Tensor, targets: torch.Tensor, model: nn.Module) -> torch.Tensor:
    return F.cross_entropy(head_logits(model, hidden).float(), targets, reduction="none")


def chunked_sequence_losses(
    model: nn.Module, hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, chunk: int = LOSS_CHUNK
) -> torch.Tensor:
    """sequence_losses from the last hidden state, LOSS_CHUNK positions at a time: [batch].

    Every chunk runs under activation checkpointing, so its logits are recomputed in backward and
    the [batch, seq, vocab] logits never exist - neither in forward nor in backward.
    """
    batch, seq = input_ids.shape
    targets = input_ids[:, 1:].reshape(-1)
    mask = attention_mask[:, 1:].float()
    flat = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    owner = torch.arange(batch, device=hidden.device).repeat_interleave(seq - 1)
    totals = torch.zeros(batch, device=hidden.device, dtype=torch.float32)
    for start in range(0, targets.numel(), chunk):
        end = start + chunk
        ce = checkpoint(_chunk_ce, flat[start:end], targets[start:end], model, use_reentrant=False)
        totals = totals.index_add(0, owner[start:end], ce * mask.reshape(-1)[start:end])
    return totals / mask.sum(dim=1)


# --- recording: reduce inside the hooks, never keep whole module outputs ---------------------


class _HookSet:
    """Forward hooks on the given modules for the duration of a with-block."""

    def __init__(self, modules: dict[str, MixedPrecisionLinear]):
        self.modules = modules
        self._handles: list = []

    def __enter__(self):
        self._handles = [m.register_forward_hook(self._hook(n, m)) for n, m in self.modules.items()]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, name: str, module: MixedPrecisionLinear):
        raise NotImplementedError


class BlockNormRecorder(_HookSet):
    """Per-token block norms of every module output, [batch, seq, n_blocks], reduced in the forward hook.

    The output itself (block_rows times larger) is dropped as soon as the module returns.
    """

    def __init__(self, modules: dict[str, MixedPrecisionLinear]):
        super().__init__(modules)
        self.norms: dict[str, torch.Tensor] = {}

    def _hook(self, name: str, module: MixedPrecisionLinear):
        def hook(_m: nn.Module, _inputs, output: torch.Tensor) -> None:
            self.norms[name] = block_norms(output.detach(), module.block_rows)

        return hook


class TaylorRecorder(_HookSet):
    """Gradient x activation block scores of every module, [batch, n_blocks], reduced in a gradient hook.

    The module's output gradient is consumed where backward produces it and never retained.
    valid [batch, seq] marks the tokens that count.
    """

    def __init__(self, modules: dict[str, MixedPrecisionLinear], valid: torch.Tensor):
        super().__init__(modules)
        self.valid = valid
        self.scores: dict[str, torch.Tensor] = {}

    def _hook(self, name: str, module: MixedPrecisionLinear):
        def hook(_m: nn.Module, _inputs, output: torch.Tensor) -> None:
            activation = output.detach()

            def on_grad(grad: torch.Tensor) -> None:
                self.scores[name] = taylor_block_scores(activation, grad, module.block_rows, self.valid)

            output.register_hook(on_grad)

        return hook


def _lengths(attention_mask: torch.Tensor) -> list[int]:
    return attention_mask.sum(dim=1).tolist()


def _positions(weights: torch.Tensor) -> list[list[int]]:
    return [torch.nonzero(w).flatten().tolist() for w in weights]


# --- instruments ---------------------------------------------------------------------------


class BlockScorer:
    """The naive score: block output norms averaged over the tokens a center mode selects."""

    def __init__(self, modules: dict[str, MixedPrecisionLinear], top_k: int = DEFAULT_TOP_K):
        self.modules = modules
        self.top_k = top_k

    @property
    def n_blocks(self) -> int:
        return sum(m.n_blocks for m in self.modules.values())

    @torch.no_grad()
    def score_batch(self, model: nn.Module, tokenizer, texts: list[str], modes: tuple[str, ...] = MODES) -> list[Scored]:
        enc = fm.encode(tokenizer, texts, model.device)
        with BlockNormRecorder(self.modules) as rec:
            out = model(**enc, output_hidden_states=True, output_attentions="attention" in modes)
        layer = len(fm.text_layers(model)) // 2
        hidden = out.hidden_states[layer + 1]
        attn = out.attentions[layer] if out.attentions is not None else None
        lengths = _lengths(enc["attention_mask"])
        norms = [rec.norms[n] for n in self.modules]
        results: list[Scored] = [{} for _ in texts]
        for mode in modes:
            weights = batch_token_weights(mode, hidden, attn, lengths, self.top_k)
            vectors = torch.cat([weighted_block_scores(n, weights) for n in norms], dim=1).cpu().numpy()
            for b, positions in enumerate(_positions(weights)):
                results[b][mode] = (vectors[b], positions)
        return results

    def score(self, model: nn.Module, tokenizer, text: str, modes: tuple[str, ...] = MODES) -> Scored:
        return self.score_batch(model, tokenizer, [text], modes)[0]


class GradientScorer:
    """Gradient x activation per block of each query's own language-model loss."""

    def __init__(self, model: nn.Module, modules: dict[str, MixedPrecisionLinear], loss_chunk: int = LOSS_CHUNK):
        self.modules = modules
        self.loss_chunk = loss_chunk
        # parameters stay frozen: the graph runs through activations, starting at the embedding output
        model.requires_grad_(False)
        fm.text_embeddings(model).register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))

    @property
    def n_blocks(self) -> int:
        return sum(m.n_blocks for m in self.modules.values())

    def score_batch(self, model: nn.Module, tokenizer, texts: list[str]) -> list[Scored]:
        enc = fm.encode(tokenizer, texts, model.device)
        valid = enc["attention_mask"].clone()
        valid[:, 0] = 0
        with torch.enable_grad(), sdpa_kernel(GRADIENT_ATTENTION), TaylorRecorder(self.modules, valid) as rec:
            hidden = model.model(**enc).last_hidden_state
            # summed per-sequence means: each sequence's gradient is that of its own mean loss
            losses = chunked_sequence_losses(model, hidden, enc["input_ids"], enc["attention_mask"], self.loss_chunk)
            losses.sum().backward()
        vectors = torch.cat([rec.scores[n] for n in self.modules], dim=1).cpu().numpy()
        return [{"gradient": (vectors[b], list(range(1, n)))} for b, n in enumerate(_lengths(enc["attention_mask"]))]

    def score(self, model: nn.Module, tokenizer, text: str) -> Scored:
        return self.score_batch(model, tokenizer, [text])[0]


# --- concentration -------------------------------------------------------------------------


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
