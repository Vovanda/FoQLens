"""Greedy decoding over a static KV cache, every step after the first replayed as one CUDA graph.

The dynamic loop (generation.greedy_tokens) is bound by the host, not by the card: a step launches
~3400 kernels from Python and costs ~270 ms at any batch while the GPU sits at 13% (2026-09-14). Here
the cache is allocated once for the whole answer, every tensor a step reads or writes keeps its
address, and the step is captured once per batch and replayed: the host launches one graph a token.

What makes the step capturable:
- The cache: a transformers StaticLayer for each layer that owns its keys and values; the layers past
  them read those through shared_kv_states (modeling_gemma4.py), so the model's interface is kept.
  The write position is a counter on the device, advanced in place. The sliding window is kept by the
  mask rather than by a ring: every key stays in the cache and a sliding layer does not look more than
  a window back. HF's StaticSlidingWindowLayer branches in Python on whether its window is full, and a
  graph would freeze that branch.
- The masks: ours, 4D boolean on the device, rewritten in place from the cache slot and handed to the
  model as a dict, so Gemma 4 builds none of its own - its builder reads the device twice a step to
  see whether any padding is left (masking_utils).
- The fed token, its position, its cache slot and the column the step writes: static buffers that the
  step advances itself.

The prefill runs eagerly, with masks of the same rule over the prompt. The first step after it runs
eagerly too, on the stream the capture uses: it warms what a first call allocates (cuBLAS workspaces,
cached constants) and is a real step, not a throwaway. A graph holds the kernels of the precision
layout set when it was captured, so a batch is decoded at one layout; the graph is released at the end
of its batch.

Invariant: the graph writes exactly what the same static step run eagerly (graph=False) writes, token for token.
Invariant: the masks select the keys HF's masks select - the prompt's real tokens and everything written after
it, up to the query, and for a sliding layer only the last `window` of those (tests/test_graph_decode_unit.py).
Invariant (approximate, bf16): a row writes what the dynamic loop writes on the same batch until the two best
tokens nearly tie - the attention sums keys that sit at other places in the cache, in another order. Measured
2026-09-14: 13 of 16 rows the same 48 tokens, the first token the same in all; the three that part do so where
the dynamic loop's top two logits were 0.06-0.25 apart, below the 5th percentile of all steps (0.48)
(tests/test_graph_decode_gpu.py).
Invariant: the host reads the device once every check_every steps and once at the end, never more.
Invariant: memory does not grow from batch to batch - every capture runs on one side stream per device
(capture_stream); measured 2026-09-15: 40 batches in a row, 0.00 MiB a batch (tests/test_graph_decode_gpu.py).
Invariant: the prefill runs on the prefill kernels of the decoder's attention plan and every step, eager or
captured, on its decode kernels (foqlens.attention; tests/test_padded_batch_gpu.py).
Invariant (approximate, bf16): a prefill in chunks of rows opens every row as one pass does, up to bf16's batch
noise, and peaks no higher (tests/test_padded_batch_gpu.py).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from torch.nn.attention import sdpa_kernel
from transformers import Cache, StaticLayer

from foqlens.attention import MATH_ONLY, AttentionPlan
from foqlens.generation import STOP_CHECK_EVERY, ends_after, mask_positions

FULL, SLIDING = "full_attention", "sliding_attention"
# Prompt tokens, padding included, that one prefill pass holds. The math prefill (#14) of a batch filled to
# answering.BATCH_TOKENS (32k) passed the 21.6 GiB a run at share 0.9 may hold, on HotpotQA's passages (E016 at
# D6, 2026-09-16: 876 MiB asked with 20.67 GiB allocated); the rows past it are prefilled in chunks into one cache,
# and the decode batch stays whole.
PREFILL_TOKENS = 16384


def own_kv_layers(config) -> int:
    """Layers that compute and cache their own keys and values; the rest reuse the last of them."""
    return config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)


class CacheMasks:
    """Boolean attention masks over a static cache of `length` slots, True where a query may look.

    Slot k holds the prompt's token k for k < prompt length, and after it the tokens the steps feed,
    one slot each. A query at slot q sees slot k when k holds a real token and k <= q; a sliding layer
    also needs k > q - window.
    """

    def __init__(self, prompt_mask: torch.Tensor, length: int, window: int):
        batch, self.prompt_length = prompt_mask.shape
        device = prompt_mask.device
        real = torch.ones(batch, length, dtype=torch.bool, device=device)
        real[:, : self.prompt_length] = prompt_mask.bool()
        self.real = real[:, None, None, :]  # [batch, 1, 1, length]
        self.slot = torch.arange(length, device=device)
        self.window = window
        self.full = torch.empty(batch, 1, 1, length, dtype=torch.bool, device=device)
        self.sliding = torch.empty_like(self.full)

    def prompt(self) -> dict[str, torch.Tensor]:
        """Masks of the prefill, a query at every prompt slot: [batch, 1, prompt length, length]."""
        query = self.slot[: self.prompt_length, None]
        full = self.real & (self.slot <= query)
        return {FULL: full, SLIDING: full & (self.slot > query - self.window)}

    def at(self, slot: torch.Tensor) -> dict[str, torch.Tensor]:
        """Masks of one query at cache `slot` ([1] on the device), rewritten in place: [batch, 1, 1, length]."""
        torch.logical_and(self.real, self.slot <= slot, out=self.full)
        torch.logical_and(self.full, self.slot > slot - self.window, out=self.sliding)
        return {FULL: self.full, SLIDING: self.sliding}


class StaticRun:
    """One batch over a static cache: the prefill at construction, then `advance` one step at a time.

    Every buffer `advance` reads or writes is allocated here, so the step can be captured and replayed.
    """

    @torch.no_grad()
    def __init__(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int,
                 stop: torch.Tensor, prefill_tokens: int = PREFILL_TOKENS):
        config = model.config.get_text_config(decoder=True)
        batch, prompt = input_ids.shape
        device = input_ids.device
        # a slot for the prompt and one for every token fed back; the last token written is never fed
        length = prompt + max_new_tokens
        layers = own_kv_layers(config)
        self.model, self.stop = model, stop
        self.cache = Cache(layers=[StaticLayer(length) for _ in range(layers)])
        self.masks = CacheMasks(attention_mask, length, config.sliding_window)
        self.tokens = torch.empty(batch, max_new_tokens, dtype=torch.long, device=device)
        positions = mask_positions(attention_mask)
        masks = self.masks.prompt()
        rows = max(1, prefill_tokens // prompt)
        logits = []
        for start in range(0, batch, rows):
            part = slice(start, start + rows)
            # one chunk reads straight into the run's cache; several read into their own and are copied in
            cache = self.cache if rows >= batch else Cache(layers=[StaticLayer(length) for _ in range(layers)])
            out = model(input_ids=input_ids[part], attention_mask={k: m[part] for k, m in masks.items()},
                        position_ids=positions[part], past_key_values=cache, use_cache=True, logits_to_keep=1)
            logits.append(out.logits[:, -1])
            if cache is not self.cache:
                self._copy_rows(cache, part, batch)
        self.token = torch.cat(logits).argmax(-1, keepdim=True)  # [batch, 1]: written at step 0, fed next
        self.tokens[:, :1] = self.token
        self.done = (self.token == stop).any(-1)
        self.position = positions[:, -1:] + 1
        self.slot = torch.full((1,), prompt, dtype=torch.long, device=device)  # where the fed token is cached
        self.column = torch.ones(1, dtype=torch.long, device=device)  # the step the next token is written at

    def _copy_rows(self, part_cache: Cache, rows: slice, batch: int) -> None:
        """A chunk's keys and values into the run's cache at its rows; the run's cache is allocated at the first chunk."""
        for whole, part in zip(self.cache.layers, part_cache.layers, strict=True):
            if not whole.is_initialized:
                # lazy_initialization reads only shape, dtype and device: an expanded view gives the batch's shape
                whole.lazy_initialization(part.keys[:1].expand(batch, -1, -1, -1),
                                          part.values[:1].expand(batch, -1, -1, -1))
            whole.keys[rows] = part.keys
            whole.values[rows] = part.values
            whole.cumulative_length.copy_(part.cumulative_length)

    @torch.no_grad()
    def advance(self) -> None:
        """Feed the last token written and write the next one; only in-place writes to the run's buffers."""
        out = self.model(input_ids=self.token, attention_mask=self.masks.at(self.slot), position_ids=self.position,
                         past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        self.tokens.index_copy_(1, self.column, token)
        self.done |= (token == self.stop).any(-1)
        self.token.copy_(token)
        self.position += 1
        self.slot += 1
        self.column += 1


@functools.cache
def capture_stream(device: torch.device) -> torch.cuda.Stream:
    """The side stream every capture on `device` runs on, made once.

    cuBLAS keeps a workspace for every stream it has run on and never gives it back, so a new stream per
    batch leaked one per batch: +6.5 MiB a batch, linear over 40 batches, against 0.00 on one stream
    (2026-09-15) - a night of stage 1 would have lost ~2 GiB.
    """
    return torch.cuda.Stream(device)


class GraphedStep:
    """StaticRun.advance as a CUDA graph: the first call runs it eagerly on the capture stream and captures
    it, every later call replays it. `release` frees the graph's memory pool."""

    def __init__(self, run: StaticRun):
        self.run = run
        self.graph: torch.cuda.CUDAGraph | None = None

    def __call__(self) -> None:
        if self.graph is not None:
            self.graph.replay()
            return
        stream = capture_stream(self.run.tokens.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.run.advance()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.run.advance()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = graph

    def release(self) -> None:
        if self.graph is not None:
            self.graph.reset()
            self.graph = None


@dataclass(frozen=True)
class StaticDecoder:
    """Decoding over a static cache. `graph` replays the step as a CUDA graph; off, the same step runs
    eagerly - the reference the graph is held to, bit for bit."""

    graph: bool = True
    check_every: int = STOP_CHECK_EVERY
    # The kernels of the prefill and of the steps (foqlens.attention); a graph keeps those it was captured with.
    attention: AttentionPlan = MATH_ONLY
    prefill_tokens: int = PREFILL_TOKENS  # prompt tokens one prefill pass holds; more are read in chunks of rows

    def tokens(self, model, input_ids, attention_mask, max_new_tokens, stop):
        with sdpa_kernel(list(self.attention.prefill)):
            try:
                run = StaticRun(model, input_ids, attention_mask, max_new_tokens, stop, self.prefill_tokens)
            except torch.OutOfMemoryError:
                # The share cap counts reserved memory, and the math prefill asks for one large score matrix:
                # fragments earlier batches left in the cache blocked 744 MiB with 2.47 GiB reserved and free
                # (E016 at D6, 2026-09-16). Handing them back and asking once more costs nothing when it succeeds.
                torch.cuda.empty_cache()
                run = StaticRun(model, input_ids, attention_mask, max_new_tokens, stop, self.prefill_tokens)
        step = GraphedStep(run) if self.graph else run.advance
        try:
            with sdpa_kernel(list(self.attention.decode)):
                for written in range(max_new_tokens):
                    if ends_after(written, max_new_tokens, run.done, self.check_every):
                        return run.tokens[:, : written + 1]
                    step()
            return run.tokens
        finally:
            if self.graph:
                step.release()


STATIC = StaticDecoder()
