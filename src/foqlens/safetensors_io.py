"""safetensors files read and written one tensor at a time.

The safetensors library maps a whole file into memory to read it, and Windows charges such a mapping of a 10 GB file
against the commit limit in full, the device's memory counting there too: two open mappings and a model on the card ran
out of it. Its save_file needs every tensor in memory at once - the whole model. Here the reader seeks to a tensor's
bytes and reads them alone, and the writer writes each tensor as it comes, so the host holds one tensor at a time.

The format: 8 bytes of header length, a JSON header - every tensor's dtype, shape and [begin, end) offsets in the data,
an optional __metadata__ of strings - then the data, the tensors back to back. The header may be padded with trailing
spaces, which is what lets the writer reserve its room first and fill it in last.

Invariant: a file written by SafetensorsWriter is read back by SafetensorsReader and by the safetensors library alike,
tensor for tensor, bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

SAFETENSORS_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.uint8: "U8",
                     torch.int16: "I16", torch.int32: "I32", torch.int64: "I64"}
SAFETENSORS_TYPES = {name: dtype for dtype, name in SAFETENSORS_NAMES.items()}
LENGTH_BYTES = 8
HEADER_BYTES_PER_TENSOR = 400  # name, dtype, shape and offsets of one entry, with room to spare
_HEADER_ALIGN = 8  # the data starts on an 8-byte boundary
_HEADER_SLACK = 4096


class SafetensorsReader:
    """Any safetensors file read tensor by tensor from its offsets, never mapped."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with open(path, "rb") as f:
            length = int.from_bytes(f.read(LENGTH_BYTES), "little")
            header = json.loads(f.read(length))
        self.metadata: dict[str, str] = header.pop("__metadata__", {})
        self._entries: dict[str, dict] = header
        self._data_start = LENGTH_BYTES + length

    def keys(self) -> list[str]:
        return list(self._entries)

    def dtype(self, key: str) -> torch.dtype:
        return SAFETENSORS_TYPES[self._entries[key]["dtype"]]

    def nbytes(self, key: str) -> int:
        """The bytes the tensor `key` takes in the file, read from the header alone."""
        begin, end = self._entries[key]["data_offsets"]
        return end - begin

    def read(self, key: str, device: str = "cpu") -> torch.Tensor:
        entry = self._entries[key]
        begin, end = entry["data_offsets"]
        data = torch.empty(end - begin, dtype=torch.uint8)
        with open(self.path, "rb") as f:
            f.seek(self._data_start + begin)
            if f.readinto(memoryview(data.numpy())) != end - begin:
                raise ValueError(f"{self.path} ends inside {key}")
        return data.view(self.dtype(key)).reshape(entry["shape"]).to(device)


def read_all(paths: list[Path], device: str = "cpu") -> dict[str, torch.Tensor]:
    """Every tensor of the files, each read alone and moved to the device."""
    readers = [SafetensorsReader(path) for path in paths]
    return {key: reader.read(key, device) for reader in readers for key in reader.keys()}


class SafetensorsWriter:
    """A safetensors file written tensor by tensor, in the order given; set `metadata` before the file closes.

    The header's room is reserved for `n_tensors` entries and `metadata_bytes` of metadata; a header that outgrows it
    fails the write and leaves no file. Writing wider types first keeps every tensor aligned to its element size.
    """

    def __init__(self, path: Path, n_tensors: int, metadata_bytes: int = 0) -> None:
        room = HEADER_BYTES_PER_TENSOR * n_tensors + metadata_bytes + _HEADER_SLACK
        self.room = -(-room // _HEADER_ALIGN) * _HEADER_ALIGN
        self.path = path
        self.entries: dict[str, dict] = {}
        self.metadata: dict[str, str] = {}
        self.end = 0

    def __enter__(self) -> SafetensorsWriter:
        self.file = open(self.path, "wb")
        self.file.write(b"\0" * (LENGTH_BYTES + self.room))
        return self

    def write(self, name: str, tensor: torch.Tensor) -> None:
        data = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()
        self.file.write(memoryview(data))
        self.entries[name] = {"dtype": SAFETENSORS_NAMES[tensor.dtype], "shape": list(tensor.shape),
                              "data_offsets": [self.end, self.end + data.nbytes]}
        self.end += data.nbytes

    def __exit__(self, kind, value, trace) -> None:
        if kind is not None:
            self._abandon()
            return
        header = json.dumps({"__metadata__": self.metadata} | self.entries, separators=(",", ":")).encode()
        if len(header) > self.room:
            self._abandon()
            raise ValueError(f"a header of {len(header)} bytes for {self.room} reserved")
        self.file.seek(0)
        self.file.write(self.room.to_bytes(LENGTH_BYTES, "little"))
        self.file.write(header.ljust(self.room, b" "))
        self.file.close()

    def _abandon(self) -> None:
        self.file.close()
        self.path.unlink(missing_ok=True)
