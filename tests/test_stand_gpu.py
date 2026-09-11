"""Bench health on Gemma 4 E2B: precision control really changes the computation where it is set and only there."""

import pytest
import torch

from foqlens import model as fm
from foqlens.precision import CONTROLLED, install
from foqlens.quant import Level

pytestmark = pytest.mark.gpu

TEXT = (
    "Water boils at 100 degrees Celsius at sea level. The boiling point drops at higher altitude "
    "because the air pressure is lower, so food takes longer to cook in the mountains."
)


@pytest.fixture(scope="module")
def stand():
    model, tokenizer = fm.load(fm.E2B, text_only=False)
    # the reference is taken on the untouched model: dropping the towers and installing the
    # controller are both checked against it
    ref_logits = fm.logits(model, tokenizer, TEXT)
    ref_hidden = fm.hidden_states(model, tokenizer, TEXT)
    ref_ppl = fm.perplexity(model, tokenizer, TEXT)
    fm.drop_towers(model)
    ctl = install(model)
    yield model, tokenizer, ctl, ref_logits, ref_hidden, ref_ppl
    del model
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_to_bf16(stand):
    yield
    stand[2].set_all(Level.BF16)


def test_install_covers_every_linear_of_the_text_decoder(stand):
    model, _, ctl, *_ = stand
    cfg = model.config.text_config
    first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers
    shared_skip = {"self_attn.k_proj", "self_attn.v_proj"}
    expected = [
        f"layers.{i}.{name}"
        for i in range(cfg.num_hidden_layers)
        for name in CONTROLLED
        if not (i >= first_shared and name in shared_skip)
    ]
    assert sorted(ctl.names()) == sorted(expected)
    # not a single plain nn.Linear is left inside the decoder layers
    leftover = [
        n for n, m in fm.text_layers(model).named_modules() if type(m) is torch.nn.Linear
    ]
    assert leftover == []


def test_all_bf16_is_bit_exact_with_the_original_model(stand):
    model, tokenizer, ctl, ref_logits, *_ = stand
    assert ctl.mean_bits() == 16.0
    assert torch.equal(fm.logits(model, tokenizer, TEXT), ref_logits)


@pytest.mark.parametrize("layer", [3, 14, 20])
def test_layer_switch_changes_only_that_layer_and_later(stand, layer):
    model, tokenizer, ctl, _, ref_hidden, _ = stand
    ctl.set_layer(layer, Level.NF4)
    hidden = fm.hidden_states(model, tokenizer, TEXT)
    for i in range(layer + 1):
        assert torch.equal(hidden[i], ref_hidden[i]), f"hidden_states[{i}] changed although layer {layer} comes later"
    assert not torch.equal(hidden[layer + 1], ref_hidden[layer + 1]), f"layer {layer} is nf4 but its output is the same"


def test_block_switch_inside_the_model_changes_only_its_channels(stand):
    model, tokenizer, ctl, *_ = stand
    name = "layers.20.mlp.down_proj"
    module = ctl.modules[name]
    captured = []
    handle = module.register_forward_hook(lambda _m, _i, out: captured.append(out[0].clone()))
    try:
        fm.logits(model, tokenizer, TEXT)
        ctl.set_blocks(name, [2], Level.NF4)
        fm.logits(model, tokenizer, TEXT)
    finally:
        handle.remove()
    before, after = captured
    rows = slice(2 * module.block_rows, 3 * module.block_rows)
    assert not torch.equal(after[:, rows], before[:, rows])
    assert torch.equal(after[:, : rows.start], before[:, : rows.start])
    assert torch.equal(after[:, rows.stop :], before[:, rows.stop :])


def test_mean_bits_follows_the_layout(stand):
    _, _, ctl, *_ = stand
    ctl.set_all(Level.NF4)
    assert ctl.mean_bits() == 4.0
    ctl.set_all(Level.INT8)
    assert ctl.mean_bits() == 8.0
    ctl.set_all(Level.BF16)
    ctl.set_layer(20, Level.NF4)
    assert 4.0 < ctl.mean_bits() < 16.0


def test_each_read_depth_costs_less_than_the_shallower_one(stand):
    model, tokenizer, ctl, _, _, ref_ppl = stand
    ppl = {}
    for level in (Level.D2, Level.D4, Level.D6, Level.D8, Level.NF4):
        ctl.set_all(level)
        ppl[level] = fm.perplexity(model, tokenizer, TEXT)
    print({lv.name: round(p, 3) for lv, p in ppl.items()}, "bf16", round(ref_ppl, 3))
    assert ppl[Level.D2] > ppl[Level.D4] > ppl[Level.D6] > ppl[Level.D8]
    assert ppl[Level.D2] > ppl[Level.NF4]  # 2 bits read are worse than 4
    assert abs(ppl[Level.D8] - ref_ppl) < abs(ppl[Level.NF4] - ref_ppl)


def test_coarser_weights_cost_perplexity(stand):
    model, tokenizer, ctl, _, _, ref_ppl = stand
    ctl.set_all(Level.NF4)
    nf4_ppl = fm.perplexity(model, tokenizer, TEXT)
    ctl.set_all(Level.INT8)
    int8_ppl = fm.perplexity(model, tokenizer, TEXT)
    assert nf4_ppl > ref_ppl
    # int8 stays close to bf16: strictly less damage than nf4
    assert abs(int8_ppl - ref_ppl) < abs(nf4_ppl - ref_ppl)
