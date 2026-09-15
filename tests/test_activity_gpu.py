"""The forward signals of the address on Gemma 4 E2B (#18): the invariants of activity.py on the real model."""

import numpy as np
import pytest

from foqlens.activity import HeadEnergyScorer, NeuronActivityScorer, block_offsets
from foqlens.quant import Level

pytestmark = pytest.mark.gpu

TEXTS = ["Photosynthesis converts light energy into chemical energy stored in glucose.",
         "What is the derivative of x squared?",
         "Name the longest river in Africa and the countries it flows through."]
EARLY = {0, 1, 2, 3}  # the working role reads the first layers only


@pytest.fixture(autouse=True)
def bf16(e2b_eager):
    e2b_eager[2].set_all(Level.BF16)


def spans(ctl, suffix: str, layers=None) -> list[tuple[int, int]]:
    at = block_offsets(ctl.modules)
    return [(at[n], at[n] + m.n_blocks) for n, m in ctl.modules.items()
            if n.endswith(suffix) and (layers is None or int(n.split(".")[1]) in layers)]


def only_on(vector: np.ndarray, allowed: list[tuple[int, int]]) -> bool:
    mask = np.zeros(len(vector), dtype=bool)
    for a, b in allowed:
        mask[a:b] = True
    return bool(np.all(vector[~mask] == 0) and np.all(vector[mask] >= 0) and vector[mask].sum() > 0)


def test_neuron_activity_lies_on_gate_and_up_only_and_the_two_agree(e2b_eager):
    model, tokenizer, ctl = e2b_eager
    scores = NeuronActivityScorer(ctl.modules).score_batch(model, tokenizer, TEXTS)
    gate, up = spans(ctl, "mlp.gate_proj"), spans(ctl, "mlp.up_proj")
    for row in scores:
        assert only_on(row, gate + up)
        for (g0, g1), (u0, u1) in zip(gate, up):
            assert np.array_equal(row[g0:g1], row[u0:u1])


def test_head_energy_lies_on_q_only_one_value_per_head(e2b_eager):
    model, tokenizer, ctl = e2b_eager
    n_heads = model.config.get_text_config().num_attention_heads
    scores = HeadEnergyScorer(ctl.modules, n_heads).score_batch(model, tokenizer, TEXTS)
    q = spans(ctl, "self_attn.q_proj")
    for row in scores:
        assert only_on(row, q)
        for a, b in q:
            per_head = row[a:b].reshape(n_heads, -1)
            assert np.all(per_head == per_head[:, :1])


def test_the_working_role_reads_only_its_layers(e2b_eager):
    model, tokenizer, ctl = e2b_eager
    n_heads = model.config.get_text_config().num_attention_heads
    neurons = NeuronActivityScorer(ctl.modules, layers=EARLY).score_batch(model, tokenizer, TEXTS)
    heads = HeadEnergyScorer(ctl.modules, n_heads, layers=EARLY).score_batch(model, tokenizer, TEXTS)
    for row in neurons:
        assert only_on(row, spans(ctl, "mlp.gate_proj", EARLY) + spans(ctl, "mlp.up_proj", EARLY))
    for row in heads:
        assert only_on(row, spans(ctl, "self_attn.q_proj", EARLY))


def test_the_same_batch_gives_identical_signals(e2b_eager):
    model, tokenizer, ctl = e2b_eager
    n_heads = model.config.get_text_config().num_attention_heads
    for scorer in (NeuronActivityScorer(ctl.modules), HeadEnergyScorer(ctl.modules, n_heads)):
        assert np.array_equal(scorer.score_batch(model, tokenizer, TEXTS), scorer.score_batch(model, tokenizer, TEXTS))
