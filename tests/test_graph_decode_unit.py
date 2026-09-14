"""The masks of the static cache select the keys HF's own masks select, at the prefill and at every later step."""

import pytest
import torch
from transformers.masking_utils import causal_mask_function, sdpa_mask, sliding_window_causal_mask_function

from foqlens.graph_decode import FULL, SLIDING, CacheMasks

WINDOW = 4
NEW = 6  # steps after the prompt, so the query moves past the window
# left-padded rows: some padding, none, most of the row
PROMPT_MASK = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1]])
BATCH, PROMPT = PROMPT_MASK.shape
LENGTH = PROMPT + NEW
RULES = {FULL: causal_mask_function, SLIDING: sliding_window_causal_mask_function(WINDOW)}


def hf_mask(kind: str, q_length: int, kv_length: int, q_offset: int) -> torch.Tensor:
    """HF's sdpa mask over the first kv_length slots: the prompt's padding, then real tokens."""
    padding = torch.cat([PROMPT_MASK, torch.ones(BATCH, kv_length - PROMPT, dtype=PROMPT_MASK.dtype)], dim=1).bool()
    return sdpa_mask(batch_size=BATCH, q_length=q_length, kv_length=kv_length, q_offset=q_offset,
                     mask_function=RULES[kind], attention_mask=padding, allow_is_causal_skip=False)


@pytest.fixture
def masks() -> CacheMasks:
    return CacheMasks(PROMPT_MASK, LENGTH, WINDOW)


@pytest.mark.parametrize("kind", [FULL, SLIDING])
def test_the_prefill_masks_are_hfs_and_never_look_past_the_prompt(masks, kind):
    ours = masks.prompt()[kind]
    assert ours.shape == (BATCH, 1, PROMPT, LENGTH)
    assert torch.equal(ours[..., :PROMPT], hf_mask(kind, PROMPT, PROMPT, 0))
    assert not ours[..., PROMPT:].any()


@pytest.mark.parametrize("kind", [FULL, SLIDING])
def test_a_step_mask_is_hfs_over_the_slots_written_so_far(masks, kind):
    for slot in range(PROMPT, LENGTH):
        ours = masks.at(torch.tensor([slot]))[kind]
        assert ours.shape == (BATCH, 1, 1, LENGTH)
        assert torch.equal(ours[..., : slot + 1], hf_mask(kind, 1, slot + 1, slot)), slot
        assert not ours[..., slot + 1 :].any(), slot


def test_a_step_rewrites_its_masks_in_place(masks):
    first = masks.at(torch.tensor([PROMPT]))
    pointers = {kind: m.data_ptr() for kind, m in first.items()}
    later = masks.at(torch.tensor([LENGTH - 1]))
    assert {kind: m.data_ptr() for kind, m in later.items()} == pointers
