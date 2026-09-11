"""Step 3 building blocks without a model: budget, layout policies, batching, summary, io, GPU monitor."""

import json

import numpy as np
import pytest

from foqlens import budget as bg
from foqlens.evaluate import Question, mc_prompt
from foqlens.gpu_monitor import GpuMonitor
from foqlens.io import read_questions, write_json
from foqlens.layouts import Directed, Random, Uniform
from foqlens.quality import batches, compute_masks, summarize
from foqlens.quant import Level


def test_select_by_budget_reaches_the_share_of_weights():
    weights = np.array([10, 10, 10, 10, 60])
    assert bg.select_by_budget(np.arange(5), weights, 0.25).tolist() == [True, True, True, False, False]


def test_directed_and_random_spend_the_same_budget():
    weights = np.full(1000, 64)
    top = bg.directed(np.arange(1000, dtype=float), weights, 0.1)
    assert top.sum() == 100 and top[-100:].all()
    rnd = bg.random_layout(weights, 0.1, np.random.default_rng(0))
    assert rnd.sum() == 100 and (rnd & top).sum() < 40


def test_aperture_zero_is_closed_and_one_is_fully_open():
    weights = np.full(10, 64)
    assert not bg.directed(np.arange(10.0), weights, 0.0).any()
    assert bg.directed(np.arange(10.0), weights, 1.0).all()


def test_to_levels_keeps_the_shape():
    assert bg.to_levels(np.array([[True, False]])).tolist() == [[Level.BF16, Level.NF4]]


def test_policies_give_one_layout_per_question():
    weights = np.full(300, 64)
    scores = np.random.default_rng(1).random((5, 300))
    idx = np.array([0, 3])
    uni = Uniform(Level.INT8, 300).levels(idx)
    assert uni.shape == (2, 300) and (uni == Level.INT8).all()
    d = Directed("pooled", 1 / 3, scores, weights).levels(idx)
    assert d.shape == (2, 300)
    assert (d[1] == Level.BF16).tolist() == bg.directed(scores[3], weights, 1 / 3).tolist()
    share = (d == Level.BF16).mean(axis=1)
    assert np.all(np.abs(4 + 12 * share - 8.0) < 0.1)  # budget 1/3 = 8 bits, as uniform int8


def test_random_policy_is_reproducible_and_differs_between_questions():
    weights = np.full(300, 64)
    policy = Random(0.1, weights, seed=0)
    a, b = policy.levels(np.array([0, 1])), policy.levels(np.array([0, 1]))
    assert np.array_equal(a, b) and not np.array_equal(a[0], a[1])
    assert policy.name == "random_0.100"


def test_topic_mask_leaves_the_question_out_of_its_own_topic():
    from foqlens.layouts import topic_masks

    scores = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0], [0.0, 7.0]])
    domains = ("a", "a", "b", "b")
    assert topic_masks(scores, domains, 0, "a").tolist() == [3.0, 0.0]  # only the other "a" question
    assert topic_masks(scores, domains, 0, "b").tolist() == [0.0, 6.0]  # the whole paired topic


def test_own_and_other_topic_policies_point_at_different_blocks():
    from foqlens.layouts import TopicMask, TopicMeans

    weights = np.full(4, 64)
    scores = np.array([[9.0, 8.0, 0.0, 0.0]] * 3 + [[0.0, 0.0, 9.0, 8.0]] * 3)
    means = TopicMeans(scores, ("a",) * 3 + ("b",) * 3)
    own = TopicMask("own", "pooled", 0.5, means, {"a": "b", "b": "a"}, weights, Level.ZERO)
    other = TopicMask("other", "pooled", 0.5, means, {"a": "b", "b": "a"}, weights, Level.ZERO)
    assert own.levels(np.array([0]))[0].tolist() == [Level.BF16, Level.BF16, Level.ZERO, Level.ZERO]
    assert other.levels(np.array([0]))[0].tolist() == [Level.ZERO, Level.ZERO, Level.BF16, Level.BF16]
    assert own.name == "own_topic_pooled_0.500"


def test_layered_keeps_the_backbone_and_fills_the_rest_by_the_fill_order():
    weights = np.full(10, 10)
    backbone = np.arange(10, 0, -1, dtype=float)  # blocks 0, 1, 2 ... most important
    fill = np.arange(10, dtype=float)  # blocks 9, 8, 7 ... first by the topic
    sharp = bg.layered(backbone, fill, weights, aperture=0.4, share=0.5)
    assert sharp.tolist() == [True, True] + [False] * 5 + [False, True, True]  # 2 backbone + 2 topic blocks
    assert bg.layered(backbone, fill, weights, 0.4, 1.0).tolist() == bg.directed(backbone, weights, 0.4).tolist()
    assert bg.layered(backbone, fill, weights, 0.4, 0.0).tolist() == bg.directed(fill, weights, 0.4).tolist()


def test_backbone_fill_policies_share_the_backbone_and_differ_in_the_fill():
    from foqlens.layouts import Backbone, BackboneFill, TopicMeans

    weights = np.full(6, 10)
    backbone = np.array([9.0, 8.0, 0.0, 0.0, 0.0, 0.0])
    scores = np.array([[0, 0, 5.0, 4.0, 0, 0]] * 2 + [[0, 0, 0, 0, 5.0, 4.0]] * 2)
    means = TopicMeans(scores, ("a", "a", "b", "b"))
    pairs = {"a": "b", "b": "a"}
    own = BackboneFill("own", "pooled", 4 / 6, 0.5, backbone, weights, means, pairs, coarse=Level.ZERO).levels(np.array([0]))[0]
    other = BackboneFill("other", "pooled", 4 / 6, 0.5, backbone, weights, means, pairs, coarse=Level.ZERO).levels(np.array([0]))[0]
    sharp = lambda row: [i for i, v in enumerate(row) if v == Level.BF16]  # noqa: E731
    assert sharp(own) == [0, 1, 2, 3] and sharp(other) == [0, 1, 4, 5]
    rnd = BackboneFill("random", "-", 4 / 6, 0.5, backbone, weights, seed=0, coarse=Level.ZERO)
    assert set(sharp(rnd.levels(np.array([0]))[0])) >= {0, 1} and rnd.name == "bb0.50_random_0.667"
    assert sharp(Backbone(4 / 6, backbone, weights, Level.ZERO).levels(np.array([0, 3]))[1])[:2] == [0, 1]


def test_paired_bootstrap_interval_and_reproducibility():
    from foqlens.stats import paired_bootstrap

    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 200)
    b = a - 0.5 + rng.normal(0, 0.05, 200)
    r = paired_bootstrap(a, b, resamples=2000)
    assert r["lo"] <= r["mean"] <= r["hi"] and r["above_zero"] and abs(r["mean"] - 0.5) < 0.02
    assert r == paired_bootstrap(a, b, resamples=2000)
    assert not paired_bootstrap(a, a + rng.normal(0, 0.1, 200), resamples=2000)["above_zero"]
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([]), np.array([]))


def test_batches_cover_everything_in_order():
    assert [b.tolist() for b in batches(5, 2)] == [[0, 1], [2, 3], [4]]


def test_compute_masks_stacks_batches_in_order():
    masks = compute_masks(lambda texts: [np.full(3, float(len(t))) for t in texts], ["a", "bb", "ccc"], batch_size=2)
    assert masks[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_evaluate_all_names_results_and_logs_every_policy(monkeypatch):
    from types import SimpleNamespace

    import foqlens.quality as quality

    monkeypatch.setattr(quality, "evaluate_policy", lambda *a: [{"policy": a[4].name}])
    lines = []
    policies = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    out = quality.evaluate_all(None, None, None, [], policies, [], 1, log=lines.append)
    assert out == {"a": [{"policy": "a"}], "b": [{"policy": "b"}]} and lines == ["[1/2] a", "[2/2] b"]


def test_summarize_overall_and_per_domain():
    questions = [Question("x", "p", 0), Question("x", "p", 0), Question("y", "p", 0)]
    rows = [{"correct": True, "logprob": -0.1, "mean_bits": 8.0}, {"correct": False, "logprob": -2.0, "mean_bits": 8.0},
            {"correct": True, "logprob": -0.3, "mean_bits": 8.0}]
    s = summarize({"p": rows}, questions)["p"]
    assert s["accuracy"] == pytest.approx(2 / 3) and s["by_domain"] == {"x": 0.5, "y": 1.0} and s["mean_bits"] == 8.0


def test_read_questions_builds_prompts_and_names_the_domain(tmp_path):
    (tmp_path / "heldout").mkdir()
    row = {"text": "Q?", "choices": ["a", "b", "c", "d"], "answer": 2}
    (tmp_path / "heldout" / "history.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    [q] = read_questions(tmp_path, "heldout/history")
    assert q == Question("history", mc_prompt("Q?", ["a", "b", "c", "d"]), 2)
    assert q.prompt.endswith("D. d\nAnswer:")


def test_write_json_creates_directories(tmp_path):
    write_json(tmp_path / "a" / "b.json", {"k": 1})
    assert json.loads((tmp_path / "a" / "b.json").read_text()) == {"k": 1}


def test_gpu_monitor_summary():
    mon = GpuMonitor(sampler=lambda: (0.0, 0.0))
    assert mon.summary() == {"samples": 0}
    mon.samples = [(50.0, 1000.0), (90.0, 3000.0), (100.0, 2000.0)]
    mon.torch_peak_mib = 1500.0
    s = mon.summary()
    assert s["utilization_mean"] == pytest.approx(80.0) and s["utilization_median"] == 90.0
    assert s["memory_reserved_peak_mib"] == 3000.0 and s["memory_allocated_peak_mib"] == 1500.0 and s["samples"] == 3


def test_gpu_monitor_samples_in_the_background_until_exit():
    with GpuMonitor(interval=0.001, sampler=lambda: (42.0, 7.0)) as mon:
        while len(mon.samples) < 3:
            pass
    taken = len(mon.samples)
    assert taken >= 3 and mon.samples[0] == (42.0, 7.0)
    assert len(mon.samples) == taken  # the thread has stopped
