"""Run configurations: a TOML table maps onto a dataclass field by field, and the repository's own files read cleanly."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from foqlens import config
from foqlens.pipeline import ADDRESS_SOURCES
from foqlens.prompt_variants import SETUPS, setup_named

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@dataclass(frozen=True)
class Inner:
    rate: float


@dataclass(frozen=True)
class Outer:
    inner: Inner
    names: tuple[str, ...]
    label: str = "x"


def test_a_table_builds_nested_dataclasses_and_lists_become_tuples():
    built = config.build(Outer, {"inner": {"rate": 0.5}, "names": ["a", "b"]})
    assert built == Outer(Inner(0.5), ("a", "b"))


def test_an_unknown_key_and_a_missing_field_are_refused_when_read():
    with pytest.raises(ValueError, match="has no"):
        config.build(Outer, {"inner": {"rate": 0.5}, "names": [], "typo": 1})
    with pytest.raises(ValueError, match="needs"):
        config.build(Outer, {"names": []})


def test_a_name_is_chosen_only_from_its_registry():
    assert config.choose("pooled", ADDRESS_SOURCES, "mask source") == "pooled"
    with pytest.raises(ValueError, match="neuron_activity"):
        config.choose("entropy", ADDRESS_SOURCES, "mask source")


def test_the_repositorys_configurations_read_and_name_what_exists():
    for name in ("small-corpus.toml", "smoke-corpus.toml", "wide-corpus.toml"):
        corpus = config.read(CONFIGS / name, config.SmallCorpus)
        assert 0 < corpus.share and 0 < corpus.calibration_share and corpus.share + corpus.calibration_share < 1
    check = config.read(CONFIGS / "address.toml", config.AddressCheck)
    assert Path(check.corpus) == Path("configs/small-corpus.toml")
    assert all(s in ADDRESS_SOURCES for s in check.sources)
    for corpus_name, wrapper in check.two_shot.items():
        assert setup_named(corpus_name, wrapper).shots > 0 and corpus_name in SETUPS


def test_the_paraphrase_configuration_reads_and_its_file_names_every_id_once():
    check = config.read(CONFIGS / "address-paraphrase.toml", config.ParaphraseCheck)
    assert all(s in ADDRESS_SOURCES for s in check.sources)
    lines = (CONFIGS.parent / check.paraphrases).read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line)["id"] for line in lines if line]
    assert len(ids) == len(set(ids)) and ids


def test_the_depth_configuration_reads_and_names_what_exists():
    check = config.read(CONFIGS / "address-depth.toml", config.DepthCheck)
    assert all(s in ADDRESS_SOURCES and ADDRESS_SOURCES[s].reads_layers for s in check.sources)
    assert all(c in SETUPS for c in check.corpora) and list(check.depths) == sorted(check.depths)
