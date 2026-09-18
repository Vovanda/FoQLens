"""E2B-it cut over bartowski's Q2_K (scripts/cut_model.py e2b-it --base bartowski-Q2_K): the file reads the published
file at its base depth byte for byte and the source checkpoint through its tail bit for bit."""

from collections import Counter

import pytest
import torch
from test_refocustensors_gpu import same_width_ints, source_tensors

from foqlens import model as fm
from foqlens.gguf_weights import GgufWeights, published_path
from foqlens.refocustensors import FILE, ModelFile, model_directory

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
BASE = "bartowski-Q2_K"


@pytest.fixture(scope="module")
def cut() -> ModelFile:
    path = model_directory(fm.E2B_IT, BASE) / FILE
    if not path.exists():
        pytest.skip(f"no model cut over {BASE} at {path}: run scripts/cut_model.py e2b-it --base {BASE}")
    return ModelFile(path, device=DEVICE)


def test_every_module_holds_the_published_files_blocks_as_its_base(cut):
    published = GgufWeights(published_path(BASE))
    formats = Counter()
    for module in cut.modules.values():
        copy = cut.copy(module["name"])
        fmt, blocks = published.base_blocks(module["name"], tuple(module["shape"]))
        assert copy.fmt is fmt and torch.equal(copy.blocks.cpu(), blocks), module["name"]
        formats[fmt.name] += 1
    print(f"bases: {dict(formats)}")
    assert sum(formats.values()) == len(cut.modules)


def test_the_source_weights_read_back_as_the_checkpoint_bit_for_bit(cut):
    state = cut.source_state()
    for key, weight in source_tensors():
        assert torch.equal(same_width_ints(state.pop(key)), same_width_ints(weight)), key
    assert not state
