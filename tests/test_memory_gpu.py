"""Memory discipline of the bench without a model: the allocator cap."""

import pytest
import torch

from foqlens import model as fm

pytestmark = pytest.mark.gpu


def test_allocation_past_free_vram_raises_instead_of_spilling():
    free, _ = torch.cuda.mem_get_info()
    fm.forbid_spill()
    try:
        # without the cap the Windows driver would place this in shared system memory and succeed
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(free + 2**30, dtype=torch.uint8, device="cuda")
        small = torch.empty(2**20, dtype=torch.uint8, device="cuda")  # within the cap all is as before
        assert small.numel() == 2**20
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
        torch.cuda.empty_cache()
