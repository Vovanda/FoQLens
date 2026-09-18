"""safetensors read and written one tensor at a time: no model, milliseconds."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from foqlens.safetensors_io import SafetensorsReader, SafetensorsWriter, read_all

TENSORS = {
    "wide": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    "half": torch.tensor([-0.0, 1.5, -2.25], dtype=torch.bfloat16),
    "scalar": torch.tensor(0.5, dtype=torch.bfloat16),
    "bytes": torch.arange(7, dtype=torch.uint8),
}


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def test_a_written_file_reads_back_through_the_reader_and_the_library(tmp_path):
    path = tmp_path / "t.safetensors"
    with SafetensorsWriter(path, n_tensors=len(TENSORS)) as writer:
        for name, tensor in TENSORS.items():
            writer.write(name, tensor)
        writer.metadata = {"format": "test"}
    reader = SafetensorsReader(path)
    assert reader.metadata == {"format": "test"} and reader.keys() == list(TENSORS)
    library = load_file(path)
    for name, tensor in TENSORS.items():
        assert same(reader.read(name), tensor) and same(library[name], tensor), name


def test_the_reader_reads_a_file_the_library_wrote(tmp_path):
    save_file({k: v for k, v in TENSORS.items() if k != "scalar"}, tmp_path / "a.safetensors")
    save_file({"scalar": TENSORS["scalar"]}, tmp_path / "b.safetensors")
    state = read_all([tmp_path / "a.safetensors", tmp_path / "b.safetensors"])
    assert sorted(state) == sorted(TENSORS)
    assert all(same(state[name], tensor) for name, tensor in TENSORS.items())


def test_a_header_past_its_room_fails_and_leaves_no_file(tmp_path):
    path = tmp_path / "t.safetensors"
    with pytest.raises(ValueError):
        with SafetensorsWriter(path, n_tensors=0) as writer:
            for i in range(100):
                writer.write(f"tensor-with-a-long-name-{i:04d}", torch.zeros(1))
    assert not path.exists()
