"""Zone strategies by name (foqlens.strategies): the registry picks the parts and adds nothing of its own."""

import numpy as np
import torch
import pytest

from foqlens import graph_zones as gz
from foqlens.layouts import GraphZoneLayout, QueryGraphZones
from foqlens.quant import Level
from foqlens.zones import MAX_ZONES
from foqlens.strategies import GRAPHS, MEDIA, STILL, Knobs, Space, per_block_layout, zone_layout

TOPICS, PER_TOPIC, BLOCKS = 3, 12, 90
DEPTHS = (Level.D2, Level.D4, Level.D6, Level.D8)


def calibration(seed: int = 0) -> tuple[np.ndarray, tuple[str, ...]]:
    """Three topics, each lighting a third of the blocks: scores [questions, blocks] and every question's topic."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(1.0, 0.1, size=(TOPICS * PER_TOPIC, BLOCKS))
    third = BLOCKS // TOPICS
    for t in range(TOPICS):
        scores[t * PER_TOPIC:(t + 1) * PER_TOPIC, t * third:(t + 1) * third] += rng.uniform(1, 2, size=(PER_TOPIC, third))
    return scores, tuple(f"t{t}" for t in range(TOPICS) for _ in range(PER_TOPIC))


def test_every_unknown_name_is_refused_with_the_names_known():
    scores, _ = calibration()
    with pytest.raises(ValueError, match="mutual-nicdm"):
        Space.build(scores, graph="knn")
    space = Space.build(scores, graph="union", k=6)
    with pytest.raises(ValueError, match="harmonic"):
        space.surface("wave")
    with pytest.raises(ValueError, match="activity"):
        space.surface("jump")
    with pytest.raises(ValueError, match="sum"):
        Knobs(Level.D2, 0.3, 1.0, combine="mean")
    with pytest.raises(ValueError, match="query"):
        zone_layout("z", "router", scores, space, space.surface(), np.ones(BLOCKS), Knobs(Level.D2, 0.3, 1.0))


@pytest.mark.parametrize("graph", sorted(GRAPHS))
def test_a_layout_by_name_is_the_layout_built_by_hand(graph):
    scores, _ = calibration()
    fields = scores - scores.mean(axis=0)
    space = Space.build(scores, graph=graph, k=6)
    surface, weights, knobs = space.surface(), np.ones(BLOCKS), Knobs(Level.D2, 0.3, 1.0)
    named = zone_layout("z", "query", fields, space, surface, weights, knobs, DEPTHS)
    by_hand = GraphZoneLayout("z", QueryGraphZones(fields, surface, weights), gz.EqualReach(0.3, space.width),
                              surface, Level.D2, 1.0, ladder=DEPTHS)
    questions = np.arange(len(fields))
    assert np.array_equal(named.levels(questions), by_hand.levels(questions))


@pytest.mark.parametrize("medium", [STILL, *sorted(MEDIA)])
def test_the_ends_of_f_hold_on_every_surface_and_topic_zones(medium):
    scores, domains = calibration()
    fields = scores - scores.mean(axis=0)
    space = Space.build(scores, k=6)
    surface = space.surface(medium, activity=scores, background=scores.mean(axis=0))
    questions = np.arange(len(fields))
    for zones in ("query", "topic"):
        centers = zone_layout("z", zones, fields, space, surface, np.ones(BLOCKS), Knobs(Level.D2, 0.0, 1.0), DEPTHS,
                              domains=domains).levels(questions)
        lifted = (centers > int(Level.D2)).sum(axis=1)  # f = 0: the centers alone, at the ceiling
        assert np.all((lifted >= 1) & (lifted <= MAX_ZONES)) and set(centers.ravel()) == {int(Level.D2), int(Level.D8)}
        everything = zone_layout("z", zones, fields, space, surface, np.ones(BLOCKS), Knobs(Level.D2, 1.0, 1.0), DEPTHS,
                                 domains=domains).levels(questions)
        assert np.all(everything > int(Level.D2)) and everything.max() == int(Level.D8)


def test_topic_zones_land_on_the_topic_blocks():
    scores, domains = calibration()
    fields = scores - scores.mean(axis=0)
    space = Space.build(scores, k=6)
    codes = zone_layout("z", "topic", fields, space, space.surface(), np.ones(BLOCKS), Knobs(Level.D2, 0.1, 1.0),
                        DEPTHS, domains=domains).levels(np.array([0, PER_TOPIC, 2 * PER_TOPIC]))
    third = BLOCKS // TOPICS
    for t in range(TOPICS):
        lifted = np.flatnonzero(codes[t] > int(Level.D2))
        assert len(lifted) and np.all((lifted >= t * third) & (lifted < (t + 1) * third)), t


def test_the_per_block_control_uses_the_same_knobs():
    scores, _ = calibration()
    codes = per_block_layout("control", scores, Knobs(Level.D4, 0.2, 1.0), DEPTHS).levels(np.arange(4))
    assert codes.min() == int(Level.D4) and codes.max() == int(Level.D8)
    assert np.allclose((codes > int(Level.D4)).mean(axis=1), 0.2, atol=0.02)


def test_every_mechanism_is_made_by_its_name_and_keeps_the_ends_of_f():
    from foqlens.strategies import MECHANISMS, Inputs, mechanism_layout

    scores, domains = calibration()
    rng = np.random.default_rng(5)
    # a toy attention of three heads: three q blocks feeding one o block
    weights = {f"layers.0.{k}": torch.as_tensor(rng.standard_normal(s), dtype=torch.float32)
               for k, s in {"self_attn.q_proj": (192, 64), "self_attn.o_proj": (64, 192)}.items()}
    peaked = np.ones((len(scores), 4))
    peaked[np.arange(len(scores)), np.arange(len(scores)) % 3] = 5.0  # every question peaks on one head's block
    questions = np.arange(len(scores))
    for mechanism in MECHANISMS:
        own = peaked if mechanism == "signal-path" else scores
        n = own.shape[1]
        inputs = Inputs(scores=own, calibration=own, block_weights=np.ones(n), domains=domains,
                        model_weights=weights, n_heads=3)
        whole = mechanism_layout(mechanism, inputs, Knobs(Level.D2, 1.0, 1.0), DEPTHS, k=1).levels(questions)
        assert np.all(whole > int(Level.D2)), mechanism
        assert whole.max() == int(Level.D8), mechanism
    with pytest.raises(ValueError, match="per-block"):
        mechanism_layout("router", inputs, Knobs(Level.D2, 0.5, 1.0))


def test_the_reach_is_picked_by_name_and_the_proportional_one_spends_the_mean_of_the_network_share():
    scores, _ = calibration()
    fields = scores - scores.mean(axis=0)
    space = Space.build(scores, k=6)
    questions = np.arange(len(fields))
    for reach in ("equal", "proportional"):
        layout = zone_layout("z", "query", fields, space, space.surface(), np.ones(BLOCKS), Knobs(Level.D2, 0.2, 1.0),
                             DEPTHS, reach=reach)
        assert layout.levels(questions).max() == int(Level.D8), reach
    with pytest.raises(ValueError, match="proportional"):
        zone_layout("z", "query", fields, space, space.surface(), np.ones(BLOCKS), Knobs(Level.D2, 0.2, 1.0), reach="gas")
