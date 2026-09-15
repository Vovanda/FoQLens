"""The static decoder asks for its prefill once more after an out-of-memory, having handed the cache back."""

import pytest
import torch

from foqlens import graph_decode
from foqlens.graph_decode import StaticDecoder


class FakeRun:
    """A run whose every row is done at once: the decoder returns after the prefill."""

    def __init__(self, model, input_ids, attention_mask, max_new_tokens, stop):
        self.tokens = torch.zeros(input_ids.shape[0], max_new_tokens, dtype=torch.long)
        self.done = torch.ones(input_ids.shape[0], dtype=torch.bool)

    def advance(self) -> None:
        raise AssertionError("no step after every row is done")


def failing(times: int):
    calls = {"n": 0}

    def make(*args):
        calls["n"] += 1
        if calls["n"] <= times:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return FakeRun(*args)

    return make, calls


def decode(monkeypatch, times: int):
    make, calls = failing(times)
    emptied = []
    monkeypatch.setattr(graph_decode, "StaticRun", make)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    ids = torch.zeros(2, 3, dtype=torch.long)
    out = StaticDecoder(graph=False).tokens(None, ids, torch.ones_like(ids), 1, torch.tensor([0]))
    return out, calls["n"], len(emptied)


def test_a_prefill_that_runs_out_of_memory_once_is_asked_again_after_the_cache_is_handed_back(monkeypatch):
    out, calls, emptied = decode(monkeypatch, times=1)
    assert out.shape == (2, 1) and calls == 2 and emptied == 1


def test_a_prefill_that_fits_is_asked_once_and_the_cache_kept(monkeypatch):
    _, calls, emptied = decode(monkeypatch, times=0)
    assert calls == 1 and emptied == 0


def test_a_second_out_of_memory_is_raised(monkeypatch):
    with pytest.raises(torch.OutOfMemoryError):
        decode(monkeypatch, times=2)
