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
    shards: int = 1  # parts of the draw every oracle takes in turn (Draw.shard): results after every part


@dataclass(frozen=True)
class Sample:
    """The sample an experiment is measured on (scripts/draw_sample.py, foqlens.sample): the questions only the top
    rung answers and the ones the lower rungs answer too, drawn from a run the judge has already read.

    The ordinary side is drawn in the shares the hard side came out with, so that a difference between the two is
    about the rungs and not about which corpus each of them is made of.
    """

    answers: str  # the folder of the run the verdicts are read from
    corpora: tuple[str, ...]
    hard: int  # questions only `top` answers
    ordinary: int  # questions every rung in `lower` answers too
    out: str  # the folder the sets are written to
    judged: str | None = None  # the judge's own folder, where it wrote one
    view: str = "verdicts"  # where the verdicts lie: `verdicts` beside the answers, or inside `answers`
    top: str = "d8"
    lower: tuple[str, ...] = ("d4", "d6")
    seed: int = 0
    paraphrased: int = 0  # questions of every side a paraphrase is written for, drawn in that side's own shares


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
    silhouette: int  # k: the top blocks of the predicted excess - the future zones - whose staying put stops reading
    tolerance: float  # epsilon: a silhouette holds when its Jaccard with the depth before is at least 1 - tolerance
    patience: int  # p: depths a silhouette must hold running before reading stops and rolls back
    cap_shares: tuple[float, ...]  # the deepest reading, patience included, as a share of the network; each tried
    base: str | None = None
    corpus_overrides: Mapping[str, str] = field(default_factory=dict)  # corpus -> its own SmallCorpus file


@dataclass(frozen=True)
class ProbeCheck:
    """How deep sets of questions written for it read under the silhouette rule (foqlens.address.stop_summary): the
    projection is fitted on the calibration questions of every corpus together, then predicts each probe's address
    from every depth, and the rule picks its stop. A probe is asked in the frozen wrapper of `probe_corpus`, one of
    those corpora, so that the projection meets the form it was fitted on and only the question differs."""

    corpus: str  # the SmallCorpus file
    corpora: tuple[str, ...]  # whose calibration questions fit the projection, together
    probe_corpus: str  # whose frozen wrapper every probe is asked in; one of `corpora`
    probes: Mapping[str, str]  # set name -> a jsonl of {"id", "question"}; other keys are not shown to the model
    source: str
    depths: tuple[int, ...]  # every depth predicts the address from the deepest one on
    ridge: float  # relative, as in DepthCheck
    base_level: str
    silhouette: int  # k, as in AdaptiveCheck
    tolerance: float
    patience: int
    cap_share: float
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


@dataclass(frozen=True)
class OracleCheck:
    """Does the address stand for the sensitivity (foqlens.sensitivity): the address's stored masks against the
    oracle's, on the laid-out questions the models know."""

    estimate: str  # masks kept by small_corpus_masks.py: the address
    oracle: str  # the same draw's masks of the oracle
    shares: tuple[float, ...]  # the top shares of blocks the overlap and the tail correlation are read at


@dataclass(frozen=True)
class GroupOracleCheck:
    """The oracles by trying (foqlens.group_oracle): every group lifted from `low` to `high` and dropped from `high`."""

    corpus: str  # the SmallCorpus file: its laid-out questions are the ones tried
    base: str  # the published base the model is cut over
    low: str
    high: str
    batch_tokens: int  # padded tokens a batch of variants holds
    replies: str | None = None  # small_corpus.targets: the answers whose replies are the target; None - the reference


@dataclass(frozen=True)
class OracleAnswers:
    """The answers at the precision fields (scripts/oracle_answers.py): every field's layout and the reference's beside
    every block at the oracle by trying's `high` in the same batches - does the layout answer as the whole network does,
    and at what bytes."""

    group_oracle: str  # the file of group_oracle.py, of the same small corpus: its questions and levels
    precision: str  # the file of precision_fields.py: every field's layout and the reference's


@dataclass(frozen=True)
class RegulatorAnswers:
    """A run of the regulator over the small corpus (scripts/layerwise_answers.py): which approach reads the weights -
    `upfront`, the map decided before the pass (docs/precision-regulator.md), or `layerwise`, a layer decided while the
    pass runs (docs/precision-regulator.md) - and the controls of the one chosen."""

    approach: str  # upfront | layerwise
    corpus: str  # the SmallCorpus file: its laid-out questions are the ones answered
    base: str  # the published base the model is cut over
    floor: str  # the level everything outside the risen part is read at
    ceiling: str  # the highest level a block may reach
    prices: tuple[float, ...]  # the knapsack's price of memory; a run per price
    source: str = "activity"  # layerwise: what a layer is scored by (foqlens.layerwise.SOURCES)
    rim: float = 0.0  # layerwise: the share of a layer held one rung over the floor under the risen part
    precision: str | None = None  # upfront: the file of maps to read (scripts/precision_fields.py)
    uniform: tuple[str, ...] = ()  # the uniform rungs answered beside it, as the ladder to compare against


@dataclass(frozen=True)
class PrecisionFields:
    """The precision fields of the oracles (scripts/precision_fields.py, foqlens.precision_field): every oracle's field of
    importance read at the threshold of the question, and the reference measured group by group, rung by rung - on the
    oracle by trying's questions and target."""

    group_oracle: str  # the oracle by trying's file (a glob joins parts): its lift and drop are fields, its questions
    fields: Mapping[str, str]  # name -> a block oracle's kept masks (a glob joins shards), summed into groups
    energies: tuple[str, ...]  # the error energies of the rungs, coarsest first (D2, D4, D6): each rung's ratio
    tolerance: float  # nats of the answer's mean NLL above every block at the top
    probes: int  # layouts a batch of the threshold's search reads beside every block at the top
    batch_tokens: int  # padded tokens a batch of the reference's variants holds
    corpus: str
    base: str
    replies: str | None = None  # must be the oracle by trying's target
    reference: bool = True  # the measured reference, 1 + 3 x groups variants a question
    # Share of the top rung's own NLL the tolerance also allows (Volodya 20.09). Flat 0.02 nats is unreachable on a
    # question the top rung itself answers at 0.15 nats, and the oracle then raises the whole network: 66 of 301
    # questions of the lift oracle came out that way. The tolerance is the larger of the two.
    tolerance_share: float = 0.0


@dataclass(frozen=True)
class GradientOracle:
    """The gradient oracle (scripts/gradient_oracle.py): gradient x activation of every block for the loss of the
    reference answer, every block at `level`."""

    corpus: str  # the SmallCorpus file
    base: str  # the published base the model is cut over
    level: str  # every block's level while the gradient is taken
    batch_tokens: int  # padded tokens a batch of questions holds
    max_tokens: int  # prompt + answer longer than this get no mask (NaN)
    calibration_questions: int  # calibration questions given a mask, spread evenly over the corpora; the rest NaN
    gap_low: str | None = None  # with gap_high: also the "quant_gap" form, the loss's change read at low for high
    gap_high: str | None = None
    attention: str = "math"  # the backward pass's sdpa backend (scoring.GRADIENT_BACKENDS)
    replies: str | None = None  # small_corpus.targets: the answers whose replies are the target; None - the reference
