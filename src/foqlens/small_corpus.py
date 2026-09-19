"""Infrastructure: the small corpus the strategies are tried on, and the masks of its questions kept on disk.

The draw (config.SmallCorpus) takes from every corpus's frozen kept questions a laid-out share and a calibration share
that never meet, and from its unknown questions a share of its own; every question is asked as its corpus freezes it.
The masks of a draw are a GPU pass of minutes that no knob changes, so they are computed once and kept as an .npz beside
what they are: the questions in their order and roles, the source, the precision read at, and the structure of the
blocks (layer, module kind, weights) - enough to read the address without the model.

Invariant: a draw gives the same questions in the same order for the same frozen files and SmallCorpus.
Invariant: stored masks read back bit for bit with their questions and the blocks' structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from foqlens import config, corpora
from foqlens.answering import Asking
from foqlens.corpora import Row
from foqlens.io import read_frozen
from foqlens.prompt_variants import TRAIN_POOL, examples_for, needs_train, setup_named
from foqlens.prompting import PromptFormat
from foqlens.quant import Level
from foqlens.selection import split_shares


@dataclass
class Draw:
    """The questions of a small corpus: laid out (known, then unknown), calibration, and how each corpus is asked."""

    laid: list[tuple[str, Row]] = field(default_factory=list)
    calibration: list[tuple[str, Row]] = field(default_factory=list)
    unknown: set[tuple[str, str]] = field(default_factory=set)
    askings: dict[str, Asking] = field(default_factory=dict)

    def prompts(self, fmt: PromptFormat, pairs: list[tuple[str, Row]]) -> list[str]:
        """The prompt each question is answered with - the one its mask reads."""
        return [self.askings[c].prompts(fmt, [r])[0] for c, r in pairs]


def draw(corpus_names: list[str], frozen_dir: Path, small: config.SmallCorpus, model: str) -> Draw:
    """The small corpus of `corpus_names` for `model` (checkpoint@revision), every corpus asked as it is frozen."""
    found = Draw()
    for corpus in corpus_names:
        rows, source = corpora.read(corpus)
        frozen = read_frozen(frozen_dir / f"{corpus}.json")
        frozen.check(model, source.revision, frozen.prompt)  # the frozen file names the setup it was frozen in
        known, background = split_shares(list(frozen.kept), small.seed, (small.share, small.calibration_share),
                                         small.floor)
        [strangers] = split_shares(list(frozen.unknown_share), small.seed, (small.unknown_share,))
        by_id = {r.id: r for r in rows}
        found.laid += [(corpus, by_id[i]) for i in known + strangers]
        found.unknown |= {(corpus, i) for i in strangers}
        found.calibration += [(corpus, by_id[i]) for i in background]
        setup = setup_named(corpus, frozen.prompt)
        train = corpora.read_train(corpus, TRAIN_POOL) if needs_train(setup) else []
        found.askings[corpus] = Asking(corpus, source.revision, model, Level.BF16, setup,
                                       examples_for(corpus, setup, train, small.seed))
    return found


@dataclass(frozen=True)
class StoredMasks:
    """Masks of a draw with what they are: [questions, n_blocks], laid-out questions first, then calibration."""

    masks: np.ndarray
    corpus: np.ndarray  # [questions] str
    ids: np.ndarray  # [questions] str
    laid: np.ndarray  # [questions] bool: laid out, else calibration
    unknown: np.ndarray  # [questions] bool
    block_layer: np.ndarray  # [n_blocks] int
    block_kind: np.ndarray  # [n_blocks] str: the module kind, as self_attn.q_proj
    block_weights: np.ndarray  # [n_blocks] weights a block holds
    meta: dict  # source, level, model, base, corpus config

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, masks=self.masks, corpus=self.corpus, ids=self.ids, laid=self.laid, unknown=self.unknown,
                 block_layer=self.block_layer, block_kind=self.block_kind, block_weights=self.block_weights,
                 meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: Path) -> StoredMasks:
        with np.load(path, allow_pickle=False) as z:
            return cls(z["masks"], z["corpus"], z["ids"], z["laid"], z["unknown"], z["block_layer"], z["block_kind"],
                       z["block_weights"], json.loads(str(z["meta"])))


def store(found: Draw, masks: np.ndarray, block_layer: np.ndarray, block_kind: np.ndarray, block_weights: np.ndarray,
          meta: dict) -> StoredMasks:
    """The masks of `found` - its laid-out questions, then its calibration - with what they are."""
    pairs = found.laid + found.calibration
    return StoredMasks(np.asarray(masks, dtype=np.float32), np.array([c for c, _ in pairs]),
                       np.array([r.id for _, r in pairs]),
                       np.array([i < len(found.laid) for i in range(len(pairs))]),
                       np.array([(c, r.id) in found.unknown for c, r in pairs]), np.asarray(block_layer),
                       np.asarray(block_kind), np.asarray(block_weights), dict(meta))
