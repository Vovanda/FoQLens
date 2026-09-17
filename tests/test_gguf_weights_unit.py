"""GgufWeights on a tiny GGUF file written here: no model, seconds."""

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import dequantize, quantize

from foqlens.gguf_weights import GgufWeights
from foqlens.quant import Level

ROWS, COLS = 4, 64  # Q8_0 blocks of 32 weights along a row


def test_a_module_reads_its_tensor_dequantized_and_shaped_as_the_weight(tmp_path):
    rng = np.random.default_rng(0)
    array = rng.standard_normal((ROWS, COLS)).astype(np.float32) * 0.02
    packed = quantize(array, GGMLQuantizationType.Q8_0)
    path = tmp_path / "tiny.gguf"
    writer = GGUFWriter(path, "gemma4")
    writer.add_tensor("blk.2.attn_output.weight", packed, raw_shape=packed.shape, raw_dtype=GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    source = GgufWeights(path)
    weight = torch.zeros(ROWS, COLS, dtype=torch.bfloat16)
    read = source.read("model.language_model.layers.2.self_attn.o_proj", weight, Level.D2)
    expected = dequantize(packed, GGMLQuantizationType.Q8_0).reshape(ROWS, COLS)
    assert read.shape == weight.shape and read.dtype == weight.dtype
    np.testing.assert_array_equal(read.float().numpy(), torch.from_numpy(expected.astype(np.float32)).to(torch.bfloat16).float().numpy())
