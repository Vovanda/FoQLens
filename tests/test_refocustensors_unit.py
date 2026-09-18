"""A model folder cut from a small fake checkpoint and read back: no model, seconds."""

import json

import pytest
import torch
from safetensors.torch import save_file
from test_kquant_unit import DEVICE

from foqlens.kquant import QK_K
from foqlens.quant import MAX_DEPTH
from foqlens.refinements import KQuantLadder, ulp_order
from foqlens.refocustensors import FILE, FileCopy, ModelFile, controlled_name, write

pytestmark = pytest.mark.gpu

CONTROLLED_KEYS = [
    "model.language_model.layers.0.self_attn.q_proj.weight",  # a Q2_K base
    "model.language_model.layers.0.mlp.down_proj.weight",  # a sensitive class, Q4_K
    "model.language_model.layers.1.per_layer_projection.weight",
]
PASSED_KEYS = [
    "model.language_model.layers.0.input_layernorm.weight",  # a weight the regulator does not read
    "model.language_model.layers.0.layer_scalar",  # a scalar, as in Gemma 4
    "model.language_model.embed_tokens.weight",
]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    source = tmp_path_factory.mktemp("source")
    tensors = {key: (torch.distributions.StudentT(3.0).sample((128, 2 * QK_K)) * 0.02).to(torch.bfloat16)
               for key in CONTROLLED_KEYS}
    tensors |= {PASSED_KEYS[0]: torch.randn(QK_K).to(torch.bfloat16), PASSED_KEYS[1]: torch.tensor(0.5, dtype=torch.bfloat16),
                PASSED_KEYS[2]: torch.randn(64, QK_K, dtype=torch.float32)}
    save_file(tensors, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps({"model_type": "fake"}), encoding="utf-8")
    (source / "README.md").write_text("not needed to run the model", encoding="utf-8")
    out = tmp_path_factory.mktemp("cut")
    write(source, out, "fake/model@0")
    return tensors, out


def test_only_the_decoder_weights_the_regulator_reads_are_controlled():
    assert [controlled_name(key) for key in CONTROLLED_KEYS] == [
        "layers.0.self_attn.q_proj", "layers.0.mlp.down_proj", "layers.1.per_layer_projection"]
    assert all(controlled_name(key) is None for key in PASSED_KEYS)
    assert controlled_name("model.vision_tower.layers.0.self_attn.q_proj.weight") is None


def test_the_folder_holds_the_model_file_and_what_runs_it(checkpoint):
    _, out = checkpoint
    assert sorted(p.name for p in out.iterdir()) == sorted([FILE, "config.json"])


def test_the_source_reads_back_bit_for_bit(checkpoint):
    tensors, out = checkpoint
    state = ModelFile(out / FILE, device=DEVICE).source_state()
    assert sorted(state) == sorted(tensors)
    for key, source in tensors.items():
        assert state[key].dtype == source.dtype, key
        assert torch.equal(ulp_order(state[key].cpu()), ulp_order(source)), key


def test_a_module_reads_the_copy_the_bench_builds_from_the_source(checkpoint):
    tensors, out = checkpoint
    copy = FileCopy(ModelFile(out / FILE, device=DEVICE))
    ladder = KQuantLadder()
    for key in CONTROLLED_KEYS:
        name = controlled_name(key)
        built = ladder.quantize(name, tensors[key].to(DEVICE))
        read = copy.quantize(name, weight=None)
        for depth in range(1, MAX_DEPTH + 1):
            assert torch.equal(read.dequantize(torch.float32, depth), built.dequantize(torch.float32, depth)), (name, depth)


def test_a_file_of_another_format_is_refused(tmp_path):
    save_file({"w": torch.zeros(1)}, tmp_path / FILE, metadata={"format": "pt"})
    with pytest.raises(ValueError):
        ModelFile(tmp_path / FILE)


def test_a_stack_cut_to_a_depth_holds_no_tail_and_refuses_the_source(checkpoint, tmp_path):
    tensors, full = checkpoint
    source = tmp_path / "source"
    source.mkdir()
    save_file(tensors, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps({"model_type": "fake"}), encoding="utf-8")
    cut = ModelFile(write(source, tmp_path / "cut", "fake/model@0", depth=2), device=DEVICE)
    assert not cut.holds_source and ModelFile(full / FILE).holds_source
    assert not any(".exact" in key for key in cut._keys)
    with pytest.raises(ValueError, match="resident"):
        cut.source_state()
    for key in CONTROLLED_KEYS:
        assert cut.copy(controlled_name(key)).depth == 2, key  # Q2_K: base and one refinement; Q4_K: its base alone
    assert (full / FILE).stat().st_size > (tmp_path / "cut" / FILE).stat().st_size
