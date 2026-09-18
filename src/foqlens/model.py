"""Loading Gemma 4 and access to its text decoder."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoConfig, AutoTokenizer, BatchEncoding, Gemma4ForConditionalGeneration, PreTrainedTokenizerBase

from foqlens.prompting import PLAIN, ChatFormat, PromptFormat
from foqlens.safetensors_io import read_all

E2B = "google/gemma-4-E2B"
E4B = "google/gemma-4-E4B"
# Instruction-tuned: they answer a question asked as it is and follow a request (write the solution,
# justify the answer), which a base checkpoint does only when shown worked examples.
E2B_IT = "google/gemma-4-E2B-it"
E4B_IT = "google/gemma-4-E4B-it"

# Pinned Hugging Face commits: every run of the bench reads exactly these weights.
# Bump only with a separate commit - results before and after a bump are not comparable.
REVISIONS = {
    E2B: "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
    E4B: "411aa17b749aa952df1359d2dcea73917a544d9a",
    E2B_IT: "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    E4B_IT: "ee0ef6023621cff504d758262d4e04895a5af4a2",
}


# Left outside the allocator cap for memory CUDA libraries take past the torch allocator.
SPILL_MARGIN_BYTES = 256 * 2**20
TOWERS = ("vision_tower", "embed_vision", "audio_tower", "embed_audio")
CHECKPOINT_WEIGHTS = "*.safetensors"


def forbid_spill(device: str = "cuda", share: float = 1.0) -> None:
    """Cap this process's allocator at the VRAM it holds plus what is free now, and at `share` of the card.

    Invariant: the bench never spills into shared system memory. Past the cap torch raises OOM;
    without it the Windows driver silently moves allocations to system RAM over PCIe and a run
    crawls on (6.4 GB spilled in the first backbone run).
    Invariant: the bench never holds more than `share` of the card (gpu_share.py).
    """
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    free, total = torch.cuda.mem_get_info(index)
    usable = min(free + torch.cuda.memory_reserved(index) - SPILL_MARGIN_BYTES, share * total)
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
    gpu_share: float = 1.0,
) -> tuple[Gemma4ForConditionalGeneration, PreTrainedTokenizerBase]:
    """The model at its pinned revision (scripts/download_models.py) and its tokenizer, in eval mode, with the allocator
    capped (forbid_spill).

    text_only drops the vision and audio towers. attn_implementation="eager" is needed wherever
    attention weights are read (step 1). gpu_share caps the VRAM at that share of the card.

    Invariant: sdpa attention runs on the math kernel - a padded row of a batch reads what it reads alone,
    within bf16's batch noise (tests/test_padded_batch_gpu.py).
    Invariant: the model is the one from_pretrained builds from the same checkpoint - its logits bit for bit
    (tests/test_refocustensors_gpu.py) - loaded at the commit of one copy of the weights (load_state).
    """
    source = Path(snapshot_download(model_id, revision=REVISIONS[model_id], local_files_only=True))
    weights = read_all(sorted(source.glob(CHECKPOINT_WEIGHTS)), device)
    return load_state(source, weights, device, dtype, attn_implementation, text_only, gpu_share)


def load_state(
    directory: str | Path,
    state_dict: dict[str, torch.Tensor],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = None,
    text_only: bool = True,
    gpu_share: float = 1.0,
    fill: Callable[[Gemma4ForConditionalGeneration], None] | None = None,
) -> tuple[Gemma4ForConditionalGeneration, PreTrainedTokenizerBase]:
    """As load, with the weights given and the config and tokenizer read from `directory` - a Hugging Face snapshot or
    a cut model's folder (foqlens.refocustensors). `fill` puts in place what the weights leave empty - the resident
    modules of a model whose controlled weights are read from their copies (precision.install_resident).

    The weights become the model's parameters as they are: the model is built with its parameters on meta and the
    tensors are assigned, never copied. from_pretrained copies them and holds every weight twice on the way - 22 GiB of
    commit for an 8.6 GiB model, against 13.5 here.
    """
    _configure(device, gpu_share)
    tokenizer = _tokenizer(AutoTokenizer.from_pretrained(directory))
    config = AutoConfig.from_pretrained(directory)
    with init_empty_weights(include_buffers=False):  # the buffers (rotary frequencies, scales) are computed for real
        model = Gemma4ForConditionalGeneration._from_config(config, dtype=dtype, attn_implementation=attn_implementation)
    loaded = model.load_state_dict(state_dict, strict=False, assign=True)
    model.tie_weights()
    if fill is not None:
        fill(model)
    # The checkpoint holds k/v projections of the layers that share another layer's keys and values; the model lists
    # them to be dropped, as from_pretrained drops them. Nothing else may be left over.
    ignored = [re.compile(p) for m in model.modules() for p in getattr(m, "_keys_to_ignore_on_load_unexpected", None) or ()]
    unexpected = [key for key in loaded.unexpected_keys if not any(p.search(key) for p in ignored)]
    empty = [name for name, t in (*model.named_parameters(), *model.named_buffers()) if t.is_meta]
    if unexpected or empty:
        raise ValueError(f"weights that fit no parameter: {unexpected[:5]}; parameters left empty: {empty[:5]}")
    return _finish(model.to(device), text_only), tokenizer


def _configure(device: str, gpu_share: float) -> None:
    # bf16 matmuls accumulate partial sums in fp32: halves the batch-size dependence of the
    # numbers (letter log-probabilities 0.19 -> 0.09 apart between a batch and one by one).
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # The memory-efficient sdpa kernel collapses every padded row of some left-padded batches into one and
    # the same state, whatever its prompt: E001 at D8 opened whole batches with 令, stage 1 at bf16 with
    # <h2> (issue #14, 2026-09-15). This build of torch has no flash kernel and cuDNN refuses Gemma 4's
    # head_dim of 256 and 512, so sdpa falls back to math.
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    forbid_spill(device, gpu_share)


def _tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    # A plain forward takes position ids from arange(seq), not from the attention mask, so left padding
    # would shift the positions of real tokens: forward batches are padded on the right. generate()
    # builds the positions from the mask itself, so generation pads on the left (encode_left).
    tokenizer.padding_side = "right"
    return tokenizer


def _finish(model: Gemma4ForConditionalGeneration, text_only: bool) -> Gemma4ForConditionalGeneration:
    model.eval()
    if text_only:
        drop_towers(model)
    return model


CHAT_MODELS = frozenset({E2B_IT, E4B_IT})


def prompt_format(model_id: str, tokenizer: PreTrainedTokenizerBase) -> PromptFormat:
    """How this checkpoint reads a prompt: its chat template if it is instruction-tuned, a plain document otherwise."""
    return ChatFormat(tokenizer) if model_id in CHAT_MODELS else PLAIN


def text_layers(model: Gemma4ForConditionalGeneration) -> nn.ModuleList:
    """Text decoder layers - the only part the bench controls."""
    return model.model.language_model.layers


def text_embeddings(model: Gemma4ForConditionalGeneration) -> nn.Module:
    """The token embedding of the text decoder - where a gradient graph through activations starts."""
    return model.model.language_model.embed_tokens


def encode(tokenizer: PreTrainedTokenizerBase, texts: list[str], device: torch.device | str) -> BatchEncoding:
    """A right-padded batch on the device, for a plain forward (see load: left padding would shift positions)."""
    if tokenizer.padding_side != "right":
        raise ValueError("batches must be padded on the right: load the tokenizer with foqlens.model.load")
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)


def encode_left(tokenizer: PreTrainedTokenizerBase, texts: list[str], device: torch.device | str) -> BatchEncoding:
    """A left-padded batch for generate(): every prompt ends where its answer starts.

    generate() builds position ids from the attention mask (cumsum - 1) and advances them per row
    (transformers 5.17, generation/utils.py), so left padding does not shift them.
    """
    return tokenizer(texts, return_tensors="pt", padding=True, padding_side="left").to(device)


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
