"""The bench's mask sources by name (pipeline.Bench.source): each name gives its source, an unknown one is refused."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from foqlens.pipeline import Bench
from foqlens.precision import Controller, MixedPrecisionLinear

NAMES = ("pooled", "neuron_activity", "head_energy")  # forward sources; the gradient ones need a real model (GPU files)


def bench() -> Bench:
    text = SimpleNamespace(num_attention_heads=8)
    model = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda decoder: text))
    linear = MixedPrecisionLinear(nn.Linear(64, 64, bias=False, dtype=torch.bfloat16), 64)
    return Bench(model, tokenizer=None, ctl=Controller({"layers.0.self_attn.o_proj": linear}))


@pytest.mark.parametrize("name", NAMES)
def test_every_source_is_made_by_its_name_with_its_batch(name):
    made = bench().source(name, batch_size=4, layers={0})
    assert made.name == name and made.batch_size == 4


def test_an_unknown_source_is_refused_with_the_names_known():
    with pytest.raises(ValueError, match="neuron_activity"):
        bench().source("entropy", batch_size=4)
