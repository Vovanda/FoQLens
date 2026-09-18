"""A model stored as .refocustensors: every controlled weight as a k-quant base, its refinements and the exact tail
over them, every other tensor as the source holds it. The folder of a model is self-sufficient - its config, its
tokenizer and one .refocustensors file - and reads back the source model bit for bit, with no Hugging Face checkpoint.

The container is safetensors: a JSON header, then raw tensors, read tensor by tensor from their offsets. What is the
bench's own is the names and the metadata:

    <key>.base            uint8 [out, super-blocks, bytes]   the base as ggml blocks (kquant.gguf_blocks)
    <key>.refinement.<k>  uint8 [out, in / 4]                 refinement k, four 2-bit codes to a byte
    <key>.exact.widths    uint8 [out, in / 16]                the exact tail's width per group of 16 weights of a row
    <key>.exact           uint8 [bytes]                       the exact tail (refinements.ExactTail)
    <key>                 as in the source                    every tensor the regulator does not read

where <key> is the source's own name of the weight. The metadata names the format and its version, the source and its
revision, and for every controlled weight its base format, shape and source type.

Invariant: a module read from the file is bit for bit the copy the bench builds from the source weight
(refinements.KQuantLadder), and the source weights read back are bit for bit the source checkpoint's.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from foqlens import model as fm
from foqlens.kquant import KFormat, Q2_K, Q4_K
from foqlens.model import REVISIONS
from foqlens.precision import BENCH_COPY, CONTROLLED, Controller, install_resident
from foqlens.quant import MAX_DEPTH
from foqlens.refinements import ExactTail, KQuantLadder, KRefinedWeight
from foqlens.safetensors_io import SafetensorsReader, SafetensorsWriter

FORMAT = "refocustensors"
VERSION = 1
FILE = f"model.{FORMAT}"
SOURCE_WEIGHTS = fm.CHECKPOINT_WEIGHTS
# Files of a Hugging Face snapshot that are not needed to run the model.
NOT_COPIED = frozenset({"README.md", ".gitattributes"})
TEXT_DECODER = "model.language_model."  # the source's key prefix of text_layers(model)'s parent
WEIGHT = ".weight"
BASE_FORMATS = {fmt.name: fmt for fmt in (Q2_K, Q4_K)}
DTYPES = {str(dtype).removeprefix("torch."): dtype for dtype in (torch.bfloat16, torch.float16, torch.float32)}
# Where cut models live: outside the repository, as the Hugging Face cache is; FOQLENS_HOME moves it.
HOME = Path(os.environ.get("FOQLENS_HOME", Path.home() / ".cache" / "foqlens"))
_MAX_PARTS = 2 + MAX_DEPTH + 1  # base, refinements, the tail's widths and bytes - a bound for the header's room
_METADATA_BYTES_PER_MODULE = 400  # one module's entry in the metadata


def model_directory(model_id: str, level: str | None = None) -> Path:
    """The folder of a model cut from its pinned revision: <HOME>/models/<name>@<revision>, and -<level> after it for
    a stack cut to a depth, so the full one is never overwritten."""
    name = f"{model_id.rsplit('/', 1)[-1]}@{REVISIONS[model_id][:8]}"
    return HOME / "models" / (name if level is None else f"{name}-{level}")


def source_id(model_id: str) -> str:
    return f"{model_id}@{REVISIONS[model_id]}"


def controlled_name(key: str) -> str | None:
    """The controller's name of a source weight (precision.install), or None if the regulator does not read it."""
    if not (key.startswith(TEXT_DECODER + "layers.") and key.endswith(WEIGHT)):
        return None
    name = key[len(TEXT_DECODER):-len(WEIGHT)]
    return name if name.split(".", 2)[2] in CONTROLLED else None


def write(source: Path, out: Path, source_id: str, copy: KQuantLadder = BENCH_COPY, device: str = "cuda",
          depth: int | None = None) -> Path:
    """Cut the source checkpoint in `source` into a model folder `out`; returns the .refocustensors file.

    Every controlled weight is quantized on `device`, as the bench quantizes it, so the file holds the bench's copy.
    With `depth` the stack stops there and holds no exact tail: the file is read resident only (load_resident).
    """
    out.mkdir(parents=True, exist_ok=True)
    keys = _source_keys(source)
    controlled = [key for key in keys if controlled_name(key) is not None]
    passed = sorted((key for key in keys if controlled_name(key) is None), key=lambda k: -keys[k].itemsize)
    modules: dict[str, dict] = {}
    path = out / FILE
    with SafetensorsWriter(path, n_tensors=len(passed) + len(controlled) * _MAX_PARTS,
                           metadata_bytes=_METADATA_BYTES_PER_MODULE * len(controlled)) as stream:
        # the widest types first, the byte parts last: every tensor stays aligned to its own element size
        for key, weight in _source_tensors(source, passed, device):
            stream.write(key, weight)
        for key, weight in _source_tensors(source, controlled, device):
            name = controlled_name(key)
            refined = copy.quantize(name, weight, MAX_DEPTH if depth is None else depth)
            parts = refined.tensors()
            if depth is None:
                parts |= ExactTail.encode(weight, refined.prediction()).tensors()
            for part, tensor in parts.items():
                stream.write(f"{key}.{part}", tensor)
            modules[key] = {"name": name, "base": refined.fmt.name, "shape": list(weight.shape),
                            "dtype": str(weight.dtype).removeprefix("torch."), "exact": depth is None}
        stream.metadata = {"format": FORMAT, "version": str(VERSION), "source": source_id, "modules": json.dumps(modules)}
    for file in source.iterdir():
        if file.is_file() and not file.match(SOURCE_WEIGHTS) and file.name not in NOT_COPIED:
            shutil.copy2(file, out / file.name)
    return path


def _source_readers(source: Path) -> dict[str, SafetensorsReader]:
    """The reader of every tensor of the source checkpoint, by key."""
    readers = [SafetensorsReader(file) for file in sorted(source.glob(SOURCE_WEIGHTS))]
    return {key: reader for reader in readers for key in reader.keys()}


def _source_keys(source: Path) -> dict[str, torch.dtype]:
    return {key: reader.dtype(key) for key, reader in _source_readers(source).items()}


def _source_tensors(source: Path, keys: list[str], device: str) -> Iterator[tuple[str, torch.Tensor]]:
    """The tensors `keys` of the source checkpoint, one at a time, in the order of `keys`."""
    readers = _source_readers(source)
    for key in keys:
        yield key, readers[key].read(key, device)


@dataclass
class ModelFile:
    """An open .refocustensors file: the copies of its controlled weights and every weight of the source."""

    path: Path
    device: str = "cuda"

    def __post_init__(self) -> None:
        self._file = SafetensorsReader(self.path)
        metadata = self._file.metadata
        self._keys = self._file.keys()
        if metadata.get("format") != FORMAT or int(metadata.get("version", 0)) != VERSION:
            raise ValueError(f"{self.path} is not a {FORMAT} file of version {VERSION}")
        self.source = metadata["source"]
        self.modules: dict[str, dict] = json.loads(metadata["modules"])
        self._key_of = {m["name"]: key for key, m in self.modules.items()}
        owners = {key: _owner(key, self.modules) for key in self._keys}
        self._passed = [key for key, owner in owners.items() if owner is None]
        self._parts_of: dict[str, list[str]] = {key: [] for key in self.modules}
        for key, owner in owners.items():
            if owner is not None:
                self._parts_of[owner].append(key)

    @property
    def directory(self) -> Path:
        return self.path.parent

    def copy(self, name: str) -> KRefinedWeight:
        """The bench's copy of the controlled weight `name` (the controller's name), on the device."""
        key = self._key_of[name]
        return self._refined(key, self._parts(key))

    def passed_state(self) -> dict[str, torch.Tensor]:
        """The weights the regulator does not read, as the source holds them, on the device."""
        return {key: self._read(key) for key in self._passed}

    @property
    def holds_source(self) -> bool:
        """Whether every controlled weight has its exact tail - a file cut to a depth has none."""
        return all(module["exact"] for module in self.modules.values())

    def source_state(self) -> dict[str, torch.Tensor]:
        """Every weight of the source model as the source holds it, bit for bit, on the device."""
        if not self.holds_source:
            raise ValueError(f"{self.path} is cut to a depth and holds no source weights: load it resident")
        state = self.passed_state()
        for key, module in self.modules.items():
            parts = self._parts(key)
            refined = self._refined(key, parts)
            tail = ExactTail.from_tensors(parts, DTYPES[module["dtype"]])
            state[key] = tail.decode(refined.prediction())
        return state

    def _parts(self, key: str) -> dict[str, torch.Tensor]:
        return {part[len(key) + 1:]: self._read(part) for part in self._parts_of[key]}

    def _read(self, key: str) -> torch.Tensor:
        return self._file.read(key, self.device)

    def _refined(self, key: str, parts: dict[str, torch.Tensor]) -> KRefinedWeight:
        module = self.modules[key]
        fmt: KFormat = BASE_FORMATS[module["base"]]
        return KRefinedWeight.from_tensors(fmt, parts, tuple(module["shape"]))


def _owner(key: str, modules: dict[str, dict]) -> str | None:
    """The controlled weight a stored tensor belongs to, if it is one of its parts."""
    head = key
    while "." in head:
        head = head.rsplit(".", 1)[0]
        if head in modules:
            return head
    return None


@dataclass(frozen=True)
class FileCopy:
    """precision.RefinedCopy read from a file: the module's copy is taken from it, never quantized again."""

    file: ModelFile

    def quantize(self, name: str, weight: torch.Tensor) -> KRefinedWeight:
        return self.file.copy(name)


def load(directory: Path, device: str = "cuda", attn_implementation: str | None = None, text_only: bool = True,
         gpu_share: float = 1.0) -> tuple[object, object, FileCopy]:
    """The model of a cut folder, as model.load gives the checkpoint's, and the copy its controller reads (install)."""
    file = ModelFile(directory / FILE, device=device)
    model, tokenizer = fm.load_state(directory, file.source_state(), device=device,
                                     attn_implementation=attn_implementation, text_only=text_only, gpu_share=gpu_share)
    return model, tokenizer, FileCopy(file)


def load_resident(directory: Path, device: str = "cuda", attn_implementation: str | None = None,
                  text_only: bool = True, gpu_share: float = 1.0) -> tuple[object, object, Controller]:
    """The model of a cut folder with its controlled weights read from their copies alone: the source weights of the
    controlled modules are never loaded, and the controller reads the depths of the file, D2 ... D8 - not bf16."""
    file = ModelFile(directory / FILE, device=device)
    controllers: list[Controller] = []
    model, tokenizer = fm.load_state(
        directory, file.passed_state(), device=device, attn_implementation=attn_implementation, text_only=text_only,
        gpu_share=gpu_share, fill=lambda m: controllers.append(install_resident(m, file.copy, device)))
    return model, tokenizer, controllers[0]
