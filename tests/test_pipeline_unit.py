"""The bench's mask sources by name (pipeline.ADDRESS_SOURCES): each name gives its source, an unknown one is refused."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from foqlens.pipeline import ADDRESS_SOURCES, Bench
from foqlens.precision import Controller, MixedPrecisionLinear

BACKWARD = ("gradient", "gradient_magnitude")  # they need a real model (GPU files)
FORWARD = tuple(name for name in ADDRESS_SOURCES if name not in BACKWARD)
BATCH = 4
HEADS = 8
WIDTH = 64


def bench() -> Bench:
    text = SimpleNamespace(num_attention_heads=HEADS)
    model = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda decoder: text))
    linear = MixedPrecisionLinear(nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.bfloat16), WIDTH)
    return Bench(model, tokenizer=None, ctl=Controller({"layers.0.self_attn.o_proj": linear}))


@pytest.mark.parametrize("name", FORWARD)
def test_every_source_is_made_by_its_name_with_its_batch(name):
    layers = {0} if ADDRESS_SOURCES[name].reads_layers else None
    made = bench().source(name, batch_size=BATCH, layers=layers)
    assert made.name == name and made.batch_size == BATCH
    assert bench().source(name).batch_size == ADDRESS_SOURCES[name].batch


def test_a_source_that_reads_every_layer_refuses_to_read_the_first_ones():
    with pytest.raises(ValueError, match="reads every layer"):
        bench().source("pooled", layers={0})


def test_an_unknown_source_is_refused_with_the_names_known():
    with pytest.raises(ValueError, match="neuron_activity"):
        bench().source("entropy", batch_size=BATCH)
