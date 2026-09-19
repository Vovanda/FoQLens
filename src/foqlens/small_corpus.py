"""Infrastructure: the small corpus the strategies are tried on, and the masks of its questions kept on disk.

The draw (config.SmallCorpus) takes from every corpus's frozen kept questions a laid-out share and a calibration share
that never meet, and from its unknown questions a share of its own; every question is asked as its corpus freezes it.
The masks of a draw are a GPU pass of minutes that no knob changes, so they are computed once and kept as an .npz beside
what they are: the questions in their order and roles, the source, the precision read at, and the structure of the
blocks (layer, module kind, weights) - enough to read the address without the model.

Invariant: a draw gives the same questions in the same order for the same frozen files and SmallCorpus.
Invariant: stored masks read back bit for bit with their questions and the blocks' structure.
Invariant: the shards of a draw part its laid-out and its calibration questions - every question in exactly one shard,
every corpus's share in every shard to within one question.
"""

from __future__ import annotations

import glob
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

    def shard(self, index: int, count: int, seed: int) -> Draw:
        """Part `index` of `count` of the draw: of every corpus's laid-out and calibration questions an equal share,
        drawn by `seed`, in the draw's order - so the first k shards are a sample of every corpus as the whole is."""
        return self.shards([index], count, seed)

    def shards(self, indices: list[int], count: int, seed: int) -> Draw:
        """The parts `indices` of `count` together (Draw.shard), in the draw's order."""
        def part(pairs: list[tuple[str, Row]]) -> list[tuple[str, Row]]:
            parts = stratified_shards([c for c, _ in pairs], count, seed)
            return [pairs[i] for i in np.sort(np.concatenate([parts[k] for k in indices]))]

        return Draw(part(self.laid), part(self.calibration), set(self.unknown), dict(self.askings))


def pick_shards(found: Draw, small: config.SmallCorpus, shards: list[int] | None) -> tuple[Draw, str]:
    """Shards `shards` (1-based) of the draw together and the suffix their files carry; the whole draw and no suffix
    for None."""
    if not shards:
        return found, ""
    if not all(1 <= s <= small.shards for s in shards) or len(set(shards)) < len(shards):
        raise ValueError(f"shards {shards} of {small.shards}")
    ordered = sorted(shards)
    name = f"shard{ordered[0]}" if len(ordered) == 1 else "shards" + "-".join(map(str, ordered))
    return found.shards([s - 1 for s in ordered], small.shards, small.seed), f"-{name}of{small.shards}"


def pick_shard(found: Draw, small: config.SmallCorpus, shard: int | None) -> tuple[Draw, str]:
    """Shard `shard` (1-based) of the draw and the suffix its files carry; the whole draw and no suffix for None."""
    return pick_shards(found, small, None if shard is None else [shard])


def stratified_shards(labels: list[str], count: int, seed: int) -> list[np.ndarray]:
    """`count` shards of the positions of `labels`, every label dealt round the shards from a seeded shuffle and a
    seeded first shard, so every shard holds every label's share to within one: sorted positions per shard."""
    rng = np.random.default_rng(seed)
    owner = np.empty(len(labels), dtype=np.int64)
    names = np.array(labels)
    for label in dict.fromkeys(labels):
        positions = rng.permutation(np.flatnonzero(names == label))
        owner[positions] = (np.arange(len(positions)) + rng.integers(count)) % count
    return [np.flatnonzero(owner == k) for k in range(count)]


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
    def concat(cls, parts: list[StoredMasks]) -> StoredMasks:
        """The shards of one pass as one: their questions in shard order, the blocks and meta of the first."""
        first = parts[0]
        return cls(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("masks", "corpus", "ids", "laid", "unknown")),
                   first.block_layer, first.block_kind, first.block_weights, first.meta)

    @classmethod
    def read(cls, pattern: str) -> StoredMasks:
        """One kept file, or the shards a glob names joined in their order."""
        if not any(c in pattern for c in "*?["):
            return cls.load(Path(pattern))
        parts = sorted(Path(p) for p in glob.glob(pattern))  # Path.glob refuses an absolute pattern
        if not parts:
            raise FileNotFoundError(pattern)
        return cls.concat([cls.load(p) for p in parts])

    def rows_of(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """The masks of the questions `pairs` (corpus, id) in their order, NaN for a question this file does not hold:
        [len(pairs), n_blocks] - so a file of the whole draw serves any of its shards."""
        where = {k: i for i, k in enumerate(zip(self.corpus.tolist(), self.ids.tolist()))}
        out = np.full((len(pairs), self.masks.shape[1]), np.nan, dtype=np.float32)
        held = [(q, where[k]) for q, k in enumerate(pairs) if k in where]
        if held:
            rows, kept = map(list, zip(*held))
            out[rows] = self.masks[kept]
        return out

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
