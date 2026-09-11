"""The naive per-block score on Gemma 4 E2B with eager attention: every mode gives a full, deterministic mask."""

import numpy as np
import pytest

from foqlens import model as fm
from foqlens.precision import install
from foqlens.scoring import MODES, BlockScorer

pytestmark = pytest.mark.gpu

TEXT = "Photosynthesis converts light energy into chemical energy stored in glucose."


@pytest.fixture(scope="module")
def scorer_and_model():
    model, tokenizer = fm.load(fm.E2B, attn_implementation="eager")
    ctl = install(model)
    return BlockScorer(ctl.modules), model, tokenizer


def test_every_mode_covers_every_block_and_is_deterministic(scorer_and_model):
    scorer, model, tokenizer = scorer_and_model
    first = scorer.score(model, tokenizer, TEXT)
    second = scorer.score(model, tokenizer, TEXT)
    assert set(first) == set(MODES)
    for mode in MODES:
        vec, positions = first[mode]
        assert vec.shape == (scorer.n_blocks,), mode
        assert np.all(vec >= 0) and vec.sum() > 0, mode
        assert np.array_equal(vec, second[mode][0]), mode
        assert 0 not in positions, mode
    assert len(first["norm"][1]) == len(first["attention"][1]) == scorer.top_k
    n_tokens = len(tokenizer(TEXT).input_ids)
    assert first["pooled"][1] == list(range(1, n_tokens))
    assert not np.array_equal(first["norm"][0], first["pooled"][0])
