import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="module")
def e2b_eager():
    """Gemma 4 E2B with eager attention and the precision controller installed.

    Module scope: released at the end of each test file, so two 12 GB models never share the GPU.
    """
    from foqlens import model as fm
    from foqlens.precision import install

    model, tokenizer = fm.load(fm.E2B, attn_implementation="eager")
    ctl = install(model)
    yield model, tokenizer, ctl
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def e2b_sdpa():
    """Gemma 4 E2B with sdpa attention - the attention of the runs (Bench.load) - and the controller installed."""
    from foqlens import model as fm
    from foqlens.precision import install

    model, tokenizer = fm.load(fm.E2B, attn_implementation="sdpa")
    ctl = install(model)
    yield model, tokenizer, ctl
    del model
    torch.cuda.empty_cache()
