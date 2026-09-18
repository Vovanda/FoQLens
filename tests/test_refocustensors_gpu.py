"""E2B-it cut into its .refocustensors folder is the model the corpus was measured on (scripts/cut_model.py).

The source weights read back are the Hugging Face checkpoint's bit for bit, so the model answers as the checkpoint
does; every module's copy is the one the bench builds from bf16, so every read depth answers as E002's ladder did -
held on the corpus by test_kquant_ladder_gpu, which runs on the model read from the file.
"""

from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

from foqlens import model as fm
from foqlens.quant import MAX_DEPTH
from foqlens.refinements import KQuantLadder
from foqlens.refocustensors import FILE, FileCopy, ModelFile, controlled_name, load, model_directory
from foqlens.safetensors_io import SafetensorsReader

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
INTS = {1: torch.uint8, 2: torch.int16, 4: torch.int32}


def same_width_ints(t: torch.Tensor) -> torch.Tensor:
    """The tensor's own bytes seen as integers: equal bit for bit, -0.0 apart from +0.0, with no copy - the per-layer
    embedding table alone is 2.35 billion weights."""
    return t.view(INTS[t.element_size()])


@pytest.fixture(scope="module")
def cut() -> ModelFile:
    path = model_directory(fm.E2B_IT) / FILE
    if not path.exists():
        pytest.skip(f"no cut model at {path}: run scripts/cut_model.py e2b-it")
    return ModelFile(path, device=DEVICE)


def source_tensors():
    source = Path(snapshot_download(fm.E2B_IT, revision=fm.REVISIONS[fm.E2B_IT], local_files_only=True))
    for file in sorted(source.glob("*.safetensors")):
        reader = SafetensorsReader(file)
        for key in reader.keys():
            yield key, reader.read(key, DEVICE)


def test_the_source_weights_read_back_as_the_checkpoint_bit_for_bit(cut):
    state = cut.source_state()
    seen = 0
    for key, weight in source_tensors():
        assert state[key].dtype == weight.dtype, key
        assert torch.equal(same_width_ints(state.pop(key)), same_width_ints(weight)), key
        seen += 1
    assert not state, f"tensors the checkpoint does not have: {sorted(state)[:5]}"
    print(f"{seen} tensors bit for bit")


def test_the_bench_loads_what_from_pretrained_builds_from_the_checkpoint_and_from_the_file():
    text = "The mitochondria produce most of the chemical energy a cell needs, stored as ATP."
    logits = {}
    for name, loader in (("checkpoint", lambda: fm.load(fm.E2B_IT)), ("file", lambda: load(model_directory(fm.E2B_IT))[:2])):
        model, tokenizer = loader()
        logits[name] = fm.logits(model, tokenizer, text)
        del model
        torch.cuda.empty_cache()
    # after the bench's loads: the precision flags they set (model._configure) are global and hold for this one too
    revision = fm.REVISIONS[fm.E2B_IT]
    reference = Gemma4ForConditionalGeneration.from_pretrained(fm.E2B_IT, revision=revision, dtype=torch.bfloat16,
                                                                device_map=DEVICE).eval()
    expected = fm.logits(reference, AutoTokenizer.from_pretrained(fm.E2B_IT, revision=revision), text)
    for name, got in logits.items():
        assert torch.equal(got, expected), name


def test_every_module_reads_the_copy_the_bench_builds_from_bf16(cut):
    copy, ladder = FileCopy(cut), KQuantLadder()
    modules = 0
    for key, weight in source_tensors():
        name = controlled_name(key)
        if name is None:
            continue
        read, built = copy.quantize(name, weight), ladder.quantize(name, weight)
        for depth in range(1, MAX_DEPTH + 1):
            assert torch.equal(read.dequantize(torch.float32, depth), built.dequantize(torch.float32, depth)), (name, depth)
        modules += 1
    assert modules == len(cut.modules)
    print(f"{modules} modules bit for bit at every depth")
