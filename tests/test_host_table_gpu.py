"""The per-layer embedding table in host memory (foqlens.host_table) reads as the table on the card, bit for bit."""

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

from foqlens import model as fm
from foqlens.host_table import HostEmbedding, per_layer_table_on_host

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
ROWS, WIDTH = 4096, 35 * 256  # a small vocabulary, E2B's row of 35 layers x 256
SCALE = 256**0.5  # Gemma 4's per-layer scale, sqrt(hidden_size_per_layer_input)
GIB = 2**30
TABLE_GIB = 4.3  # E2B-it's table is 4.37 GiB; the card must give back at least this much


def scaled_embedding() -> Gemma4TextScaledWordEmbedding:
    torch.manual_seed(0)
    module = Gemma4TextScaledWordEmbedding(ROWS, WIDTH, padding_idx=0, embed_scale=SCALE)
    return module.to(DEVICE, torch.bfloat16)


def test_the_host_table_reads_the_rows_the_card_table_reads():
    card = scaled_embedding()
    host = HostEmbedding.from_embedding(card).to(DEVICE)
    ids = torch.randint(0, ROWS, (3, 17), device=DEVICE)
    ids[0, :4] = torch.tensor([5, 5, 0, ROWS - 1])  # a repeated row, the padding row, the last row
    assert torch.equal(host(ids), card(ids))
    assert not host.weight.is_cuda and host.weight.is_pinned()


def test_the_host_table_is_read_inside_a_cuda_graph():
    card = scaled_embedding()
    host = HostEmbedding.from_embedding(card).to(DEVICE)
    ids = torch.zeros(8, 1, dtype=torch.long, device=DEVICE)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        host(ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = host(ids)
    ids.copy_(torch.randint(0, ROWS, ids.shape, device=DEVICE))  # the graph reads the ids where they lie
    graph.replay()
    assert torch.equal(out, card(ids))


def test_the_model_answers_the_same_with_its_table_on_the_host(e2b_it_refocused):
    model, tokenizer, _ = e2b_it_refocused
    ids = fm.encode(tokenizer, ["The capital of France is", "Two plus two equals"], DEVICE)
    with torch.no_grad():
        before = model(**ids).logits
    held = torch.cuda.memory_allocated()
    per_layer_table_on_host(model)
    assert (held - torch.cuda.memory_allocated()) / GIB >= TABLE_GIB
    with torch.no_grad():
        after = model(**ids).logits
    assert torch.equal(before, after)
