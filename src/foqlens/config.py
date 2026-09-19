"""Run configurations: a TOML file per run under configs/, read into frozen dataclasses.

A run is its configuration: what a script does is decided by the file it is given, and a flag on the command line only
overrides a value for a smoke. A table maps onto a dataclass field by field - a key the dataclass does not have, or a
field without a default that the table leaves out, is an error when the file is read, not half an hour into a run.
Names that pick a part (a mechanism, a mask source, a reach) are checked against their registry by `choose`.

Invariant: read(path, cls) gives the same dataclass for the same file, and refuses any key cls does not name.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

T = TypeVar("T")


def build(cls: type[T], table: Mapping[str, Any]) -> T:
    """The dataclass `cls` from a table; nested dataclass fields from nested tables, lists into tuples."""
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(table) - set(known))
    if unknown:
        raise ValueError(f"{cls.__name__} has no {unknown}; it takes {sorted(known)}")
    missing = sorted(n for n, f in known.items()
                     if n not in table and f.default is MISSING and f.default_factory is MISSING)
    if missing:
        raise ValueError(f"{cls.__name__} needs {missing}")
    hints = get_type_hints(cls)
    values = {}
    for name, value in table.items():
        kind = hints[name]
        if is_dataclass(kind):
            value = build(kind, value)
        elif isinstance(value, list):
            value = tuple(value)
        values[name] = value
    return cls(**values)


def read(path: Path, cls: type[T]) -> T:
    """The configuration of a run: the TOML file at `path` as `cls`."""
    with open(path, "rb") as file:
        return build(cls, tomllib.load(file))


def choose(name: str, registry: Mapping[str, Any], what: str) -> str:
    """`name` if the registry of `what` knows it; otherwise the error lists what it knows."""
    if name not in registry:
        raise ValueError(f"unknown {what} {name!r}, expected one of {sorted(registry)}")
    return name


@dataclass(frozen=True)
class SmallCorpus:
    """The small corpus the strategies are tried on: shares of every corpus's kept questions and of its unknown share."""

    share: float
    calibration_share: float
    floor: int
    unknown_share: float
    seed: int = 0


@dataclass(frozen=True)
class AddressCheck:
    """Is a mask source an address (foqlens.address): the sources, the two-example wrapper of every corpus, and the
    working reading - its first layers at the base precision."""

    corpus: str  # the SmallCorpus file every run shares, so that every run draws the same questions
    sources: tuple[str, ...]
    two_shot: Mapping[str, str]
    working_layers: tuple[int, ...]  # every depth of the working reading tried: the fewest that holds is the price
    base_level: str
    base: str | None = None


@dataclass(frozen=True)
class DepthCheck:
    """How deep the working address must read (foqlens.address.deep_address): per corpus, a projection from the first
    N layers at the base onto the rest at bf16, fitted on the small corpus's calibration questions and tried on its
    laid-out ones, for every N of `depths`."""

    corpus: str  # the SmallCorpus file
    corpora: tuple[str, ...]
    sources: tuple[str, ...]
    depths: tuple[int, ...]
    ridges: tuple[float, ...]  # relative: alpha is ridge times the mean variance of a read block; every one tried
    base_level: str
    base: str | None = None
    # windows [start, end) of layers the address is read from, beside the depths' [0, N): the pass runs up to the end
    # either way, the window only leaves out layers close to the tokens
    windows: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class AdaptiveCheck:
    """Does a question tell at a shallow depth that it needs a deeper one (foqlens.address.adaptive_depth): one source,
    every depth of `depths` predicting the address from the deepest one on, the policy between `low` and `high`."""

    corpus: str  # the SmallCorpus file
    corpora: tuple[str, ...]
    source: str
    depths: tuple[int, ...]
    low: int
    high: int
    ridge: float  # relative, as in DepthCheck
    base_level: str
    base: str | None = None
    corpus_overrides: Mapping[str, str] = field(default_factory=dict)  # corpus -> its own SmallCorpus file


@dataclass(frozen=True)
class ParaphraseCheck:
    """Does a mask source find a question's meaning or its words: the question against its paraphrase, beside the
    same test on their bags of tokens."""

    corpus: str  # the corpus the paraphrased questions come from
    paraphrases: str  # a jsonl of {"id", "paraphrase"}
    sources: tuple[str, ...]
    base: str | None = None
