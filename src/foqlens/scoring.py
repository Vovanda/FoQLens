"""Mask sources: how much each block of weight rows matters for a query.

Two instruments, both producing one score per block over all controlled modules, in module order:

- BlockScorer - the naive score: the L2 norm of a block's output,
  averaged over the tokens a center mode selects ("norm", "pooled", "attention");
- GradientScorer - gradient x activation: the first-order estimate of
  the change in the query's own language-model loss if the block's output were zeroed. In two forms from one
  backward pass: "gradient", |sum| of the block's terms, and "gradient_magnitude", the sum of their |.|, whose
  terms cannot cancel (issue #17). Given a rung gap (low, high) the same pass also gives "quant_gap", the signed
  first-order change of the loss when the block is read at low instead of high (QuantGapRecorder).

Both work on right-padded batches. Background subtraction is done by the caller.

Invariants:
- Invariant: "gradient_magnitude" >= "gradient" for every block, equal where all the block's terms share a sign.
- Invariant: padding and the first token (<bos>) never contribute to a mask.
- Invariant: "quant_gap" is 0 for a block whose two levels read the same weights, and equals
  sum grad * (W_low - W_high) x over its valid tokens and rows.
- Invariant: the same texts in the same batch give identical masks - with sdpa attention too: the
  gradient pass runs attention on GRADIENT_ATTENTION, whose backward is deterministic.
- Invariant (approximate, bf16): a text's mask in a batch points the same way as its mask alone,
  cosine >= 0.99 - batched kernels accumulate differently, which moves token states by up to ~2%.
  The discrete center modes ("norm", "attention") may swap one center of four on near-ties.
"""

from __future__ import annotations

import math
from contextlib import ExitStack

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from foqlens import model as fm
from foqlens.precision import MixedPrecisionLinear
from foqlens.quant import Level

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


def taylor_terms(
    output: torch.Tensor, grad: torch.Tensor, block_rows: int, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """grad * output of every valid token and row, by block: [(batch,) seq, out] -> [batch, seq, n_blocks, block_rows].

    valid [(batch,) seq] marks counted tokens; by default every token except position 0. A single
    sequence comes back with a batch of one.
    """
    if output.dim() == 2:
        output, grad = output[None], grad[None]
        valid = None if valid is None else valid[None]
    if valid is None:
        valid = torch.ones(output.shape[:2], device=output.device)
        valid[:, 0] = 0
    return _blocks(grad.float() * output.float() * valid[..., None].float(), block_rows)


def taylor_block_scores(
    output: torch.Tensor, grad: torch.Tensor, block_rows: int, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """|sum over valid tokens and block rows of grad * output|: [(batch,) seq, out] -> [(batch,) n_blocks]."""
    scores = taylor_terms(output, grad, block_rows, valid).sum(dim=(1, 3)).abs()
    return scores[0] if output.dim() == 2 else scores


def taylor_block_magnitudes(
    output: torch.Tensor, grad: torch.Tensor, block_rows: int, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """sum over valid tokens and block rows of |grad * output|: [(batch,) seq, out] -> [(batch,) n_blocks].

    The form whose terms cannot cancel (issue #17): |sum| over a large group correlates poorly with
    importance because its terms of opposite sign cancel (Molchanov et al. 2019).
    """
    scores = taylor_terms(output, grad, block_rows, valid).abs().sum(dim=(1, 3))
    return scores[0] if output.dim() == 2 else scores


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


def answer_losses(model: nn.Module, hidden: torch.Tensor, enc, starts: list[int]) -> torch.Tensor:
    """The mean negative log-likelihood of every answer's tokens, from the last hidden state of prompt + answer on a
    right-padded batch; `starts` are the prompts' lengths in tokens: [batch]. Logits are taken at the answer's
    positions only, so a long prompt costs no [seq, vocab] logits."""
    ends = enc["attention_mask"].sum(dim=1).tolist()
    out = []
    for b, (start, end) in enumerate(zip(starts, ends)):
        positions = torch.arange(start - 1, end - 1, device=hidden.device)  # each predicts the next token
        logits = head_logits(model, hidden[b, positions]).float()
        out.append(F.cross_entropy(logits, enc["input_ids"][b, positions + 1]))
    return torch.stack(out)


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

    Both forms come from the one tensor of terms: scores |sum| (taylor_block_scores) and magnitudes
    sum |.| (taylor_block_magnitudes); the absolute value is taken in place after the signed sum, so
    the second form costs no second copy. The module's output gradient is consumed where backward
    produces it and never retained. valid [batch, seq] marks the tokens that count.
    """

    def __init__(self, modules: dict[str, MixedPrecisionLinear], valid: torch.Tensor):
        super().__init__(modules)
        self.valid = valid
        self.scores: dict[str, torch.Tensor] = {}
        self.magnitudes: dict[str, torch.Tensor] = {}

    def _hook(self, name: str, module: MixedPrecisionLinear):
        def hook(_m: nn.Module, _inputs, output: torch.Tensor) -> None:
            activation = output.detach()

            def on_grad(grad: torch.Tensor) -> None:
                terms = taylor_terms(activation, grad, module.block_rows, self.valid)
                self.scores[name] = terms.sum(dim=(1, 3)).abs()
                self.magnitudes[name] = terms.abs_().sum(dim=(1, 3))

            output.register_hook(on_grad)

        return hook


class QuantGapRecorder(_HookSet):
    """The first-order change of the loss when a block is read at `low` instead of `high`, [batch, n_blocks]:
    sum over valid tokens and the block's rows of grad * (W_low - W_high) x - signed, positive where the coarser
    level raises the loss. The gradient's other half: gradient x activation says how much the loss listens to a
    block's output, this also says how far the rung gap really moves it. The change (W_low - W_high) x is taken in
    the forward hook and consumed in the gradient hook; the gap weight is unpacked per batch and dropped at once,
    so no second model is held.
    """

    def __init__(self, modules: dict[str, MixedPrecisionLinear], valid: torch.Tensor, low: Level, high: Level):
        super().__init__(modules)
        self.valid, self.low, self.high = valid, low, high
        self.change: dict[str, torch.Tensor] = {}

    def _hook(self, name: str, module: MixedPrecisionLinear):
        def hook(_m: nn.Module, inputs, output: torch.Tensor) -> None:
            with torch.no_grad():
                moved = F.linear(inputs[0].detach(), module.read_weight(self.low) - module.read_weight(self.high))

            def on_grad(grad: torch.Tensor) -> None:
                self.change[name] = taylor_terms(moved, grad, module.block_rows, self.valid).sum(dim=(1, 3))

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

    def __init__(self, model: nn.Module, modules: dict[str, MixedPrecisionLinear], loss_chunk: int = LOSS_CHUNK,
                 gap: tuple[Level, Level] | None = None):
        self.modules = modules
        self.loss_chunk = loss_chunk
        self.gap = gap  # (low, high): the same backward pass also gives the "quant_gap" form (QuantGapRecorder)
        # parameters stay frozen: the graph runs through activations, starting at the embedding output
        model.requires_grad_(False)
        fm.text_embeddings(model).register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))

    @property
    def n_blocks(self) -> int:
        return sum(m.n_blocks for m in self.modules.values())

    def _forms(self, model: nn.Module, enc, losses_of) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Every form of every block, [batch, n_blocks], for the per-sequence losses `losses_of(hidden)` gives, and
        those losses: [batch]."""
        valid = enc["attention_mask"].clone()
        valid[:, 0] = 0
        with ExitStack() as stack:
            stack.enter_context(torch.enable_grad())
            stack.enter_context(sdpa_kernel(GRADIENT_ATTENTION))
            rec = stack.enter_context(TaylorRecorder(self.modules, valid))
            gap = stack.enter_context(QuantGapRecorder(self.modules, valid, *self.gap)) if self.gap else None
            hidden = model.model(**enc).last_hidden_state
            losses = losses_of(hidden)
            # summed per-sequence means: each sequence's gradient is that of its own mean loss
            losses.sum().backward()
        read = [("gradient", rec.scores), ("gradient_magnitude", rec.magnitudes)]
        read += [("quant_gap", gap.change)] if gap else []
        forms = {form: torch.cat([rec_form[n] for n in self.modules], dim=1).cpu().numpy() for form, rec_form in read}
        return forms, losses.detach().float().cpu().numpy()

    def score_batch(self, model: nn.Module, tokenizer, texts: list[str]) -> list[Scored]:
        enc = fm.encode(tokenizer, texts, model.device)
        forms, _ = self._forms(model, enc, lambda hidden: chunked_sequence_losses(
            model, hidden, enc["input_ids"], enc["attention_mask"], self.loss_chunk))
        return [{form: (vectors[b], list(range(1, n))) for form, vectors in forms.items()}
                for b, n in enumerate(_lengths(enc["attention_mask"]))]

    def answer_batch(self, model: nn.Module, tokenizer, prompts: list[str],
                     answers: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Both forms for the loss of every answer after its prompt (answer_losses) - the objective the oracles by
        trying measure (foqlens.group_oracle), so the three are read against one target: [batch, n_blocks]; and the
        answers' NLL the gradient is taken of: [batch], to be checked against the oracle by trying at the same level."""
        enc = fm.encode(tokenizer, [p + a for p, a in zip(prompts, answers)], model.device)
        starts = [len(tokenizer(p)["input_ids"]) for p in prompts]
        return self._forms(model, enc, lambda hidden: answer_losses(model, hidden, enc, starts))

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
