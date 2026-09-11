"""Loading Gemma 4 and access to its text decoder."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoTokenizer, BatchEncoding, Gemma4ForConditionalGeneration, PreTrainedTokenizerBase

E2B = "google/gemma-4-E2B"
E4B = "google/gemma-4-E4B"

# Pinned Hugging Face commits: every run of the bench reads exactly these weights.
# Bump only with a separate commit - results before and after a bump are not comparable.
REVISIONS = {
    E2B: "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
    E4B: "411aa17b749aa952df1359d2dcea73917a544d9a",
}


# Left outside the allocator cap for memory CUDA libraries take past the torch allocator.
SPILL_MARGIN_BYTES = 256 * 2**20
TOWERS = ("vision_tower", "embed_vision", "audio_tower", "embed_audio")


def forbid_spill(device: str = "cuda") -> None:
    """Cap this process's allocator at the VRAM it holds plus what is free now.

    Invariant: the bench never spills into shared system memory. Past the cap torch raises OOM;
    without it the Windows driver silently moves allocations to system RAM over PCIe and a run
    crawls on (6.4 GB spilled in the first backbone run).
    """
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    free, total = torch.cuda.mem_get_info(index)
    usable = free + torch.cuda.memory_reserved(index) - SPILL_MARGIN_BYTES
    torch.cuda.set_per_process_memory_fraction(max(usable, 0) / total, index)


def drop_towers(model: Gemma4ForConditionalGeneration) -> None:
    """Remove the vision and audio towers (~0.9 GiB for E2B): text-only input never calls them."""
    for name in TOWERS:
        setattr(model.model, name, None)
    torch.cuda.empty_cache()


def load(
    model_id: str = E2B,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = None,
    text_only: bool = True,
) -> tuple[Gemma4ForConditionalGeneration, PreTrainedTokenizerBase]:
    """The model at its pinned revision and its tokenizer, in eval mode, with the allocator capped (forbid_spill).

    text_only drops the vision and audio towers. attn_implementation="eager" is needed wherever
    attention weights are read (step 1).
    """
    # bf16 matmuls accumulate partial sums in fp32: halves the batch-size dependence of the
    # numbers (letter log-probabilities 0.19 -> 0.09 apart between a batch and one by one).
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    forbid_spill(device)
    revision = REVISIONS[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    # Gemma 4 derives position ids from arange(seq), not from the attention mask, so left padding
    # would shift the positions of real tokens. Batches are padded on the right.
    tokenizer.padding_side = "right"
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=dtype, device_map=device, attn_implementation=attn_implementation
    )
    model.eval()
    if text_only:
        drop_towers(model)
    return model, tokenizer


def text_layers(model: Gemma4ForConditionalGeneration) -> nn.ModuleList:
    """Text decoder layers - the only part the bench controls."""
    return model.model.language_model.layers


def text_embeddings(model: Gemma4ForConditionalGeneration) -> nn.Module:
    """The token embedding of the text decoder - where a gradient graph through activations starts."""
    return model.model.language_model.embed_tokens


def encode(tokenizer: PreTrainedTokenizerBase, texts: list[str], device: torch.device | str) -> BatchEncoding:
    """A right-padded batch on the device (see load: left padding would shift positions)."""
    if tokenizer.padding_side != "right":
        raise ValueError("batches must be padded on the right: load the tokenizer with foqlens.model.load")
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)


@torch.no_grad()
def hidden_states(
    model: Gemma4ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> tuple[torch.Tensor, ...]:
    """Representations after every layer: [0] is the embeddings, [i + 1] is the output of layer i. Shape [seq, hidden]."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    return tuple(h[0] for h in out.hidden_states)


@torch.no_grad()
def logits(
    model: Gemma4ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> torch.Tensor:
    """Logits at every position, shape [seq, vocab]."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    return model(**inputs).logits[0]


@torch.no_grad()
def perplexity(
    model: Gemma4ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> float:
    """Perplexity on a text: exp of the mean next-token cross-entropy."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model(**inputs, labels=inputs["input_ids"])
    return float(torch.exp(out.loss))
