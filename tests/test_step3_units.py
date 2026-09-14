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


def test_precision_share_zero_is_all_coarse_and_one_all_sharp():
    weights = np.full(10, 64)
    assert not bg.directed(np.arange(10.0), weights, 0.0).any()
    assert bg.directed(np.arange(10.0), weights, 1.0).all()


def test_a_precision_share_outside_zero_to_one_is_refused():
    import math

    weights = np.full(10, 64)
    for precision_share in (-0.1, 1.5, math.nan):
        with pytest.raises(ValueError):
            bg.check_precision_share(precision_share)
        with pytest.raises(ValueError):
            bg.directed(np.arange(10.0), weights, precision_share)
    assert bg.check_precision_share(0.0) == 0.0 and bg.check_precision_share(1.0) == 1.0


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
    from foqlens.layouts import OtherTopic, OwnTopic, TopicMask, TopicMeans

    weights = np.full(4, 64)
    scores = np.array([[9.0, 8.0, 0.0, 0.0]] * 3 + [[0.0, 0.0, 9.0, 8.0]] * 3)
    means = TopicMeans(scores, ("a",) * 3 + ("b",) * 3)
    own = TopicMask(OwnTopic(means), "pooled", 0.5, weights, Level.ZERO)
    other = TopicMask(OtherTopic(means, {"a": "b", "b": "a"}), "pooled", 0.5, weights, Level.ZERO)
    assert own.levels(np.array([0]))[0].tolist() == [Level.BF16, Level.BF16, Level.ZERO, Level.ZERO]
    assert other.levels(np.array([0]))[0].tolist() == [Level.ZERO, Level.ZERO, Level.BF16, Level.BF16]
    assert own.name == "own_topic_pooled_0.500"


def test_layered_keeps_the_backbone_and_fills_the_rest_by_the_fill_order():
    weights = np.full(10, 10)
    backbone = np.arange(10, 0, -1, dtype=float)  # blocks 0, 1, 2 ... most important
    fill = np.arange(10, dtype=float)  # blocks 9, 8, 7 ... first by the topic
    sharp = bg.layered(backbone, fill, weights, precision_share=0.4, share=0.5)
    assert sharp.tolist() == [True, True] + [False] * 5 + [False, True, True]  # 2 backbone + 2 topic blocks
    assert bg.layered(backbone, fill, weights, 0.4, 1.0).tolist() == bg.directed(backbone, weights, 0.4).tolist()
    assert bg.layered(backbone, fill, weights, 0.4, 0.0).tolist() == bg.directed(fill, weights, 0.4).tolist()


def test_backbone_fill_policies_share_the_backbone_and_differ_in_the_fill():
    from foqlens.layouts import Backbone, BackboneFill, OtherTopic, OwnTopic, RandomFill, TopicMeans

    weights = np.full(6, 10)
    backbone = np.array([9.0, 8.0, 0.0, 0.0, 0.0, 0.0])
    scores = np.array([[0, 0, 5.0, 4.0, 0, 0]] * 2 + [[0, 0, 0, 0, 5.0, 4.0]] * 2)
    means = TopicMeans(scores, ("a", "a", "b", "b"))
    pairs = {"a": "b", "b": "a"}
    own = BackboneFill(OwnTopic(means), "pooled", 4 / 6, 0.5, backbone, weights, coarse=Level.ZERO).levels(np.array([0]))[0]
    other = BackboneFill(OtherTopic(means, pairs), "pooled", 4 / 6, 0.5, backbone, weights, coarse=Level.ZERO).levels(np.array([0]))[0]
    sharp = lambda row: [i for i, v in enumerate(row) if v == Level.BF16]  # noqa: E731
    assert sharp(own) == [0, 1, 2, 3] and sharp(other) == [0, 1, 4, 5]
    rnd = BackboneFill(RandomFill(4 / 6, seed=0), "-", 4 / 6, 0.5, backbone, weights, coarse=Level.ZERO)
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


def test_token_batches_keep_the_budget_and_take_every_index_once():
    from foqlens.quality import token_batches

    lengths = np.random.default_rng(1).integers(5, 200, size=97).tolist()
    groups = token_batches(lengths, max_tokens=8 * max(lengths), max_batch=32)
    assert sorted(np.concatenate(groups).tolist()) == list(range(97))
    for g in groups:
        assert len(g) <= 32 and len(g) * max(lengths[i] for i in g) <= 8 * max(lengths)
    assert lengths[groups[0][0]] == max(lengths) and len(groups[0]) == 8  # the longest go first, 8 of them
    assert token_batches([500, 10], max_tokens=100, max_batch=4)[0].tolist() == [0]  # too long alone still goes


def test_compute_masks_in_token_batches_come_back_in_the_prompts_order():
    prompts = ["aaaa", "b", "ccc", "dd"]
    masks = compute_masks(lambda texts: [np.full(2, float(len(t))) for t in texts], prompts, 2,
                          groups=[np.array([0, 2]), np.array([3, 1])])
    assert masks[:, 0].tolist() == [4.0, 1.0, 3.0, 2.0]


def test_evaluate_all_names_results_and_logs_every_policy(monkeypatch):
    from types import SimpleNamespace

    import foqlens.quality as quality

    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(quality, "evaluate_policy", lambda *a, **k: [{"policy": a[4].name}])
    lines = []
    policies = [SimpleNamespace(name="a", levels=lambda idx: idx), SimpleNamespace(name="b", levels=lambda idx: idx)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        out = quality.evaluate_all(None, None, None, [], policies, [], 1, log=lines.append, pool=pool)
    assert out == {"a": [{"policy": "a"}], "b": [{"policy": "b"}]}
    assert lines[0].startswith("[1/2] a (50%), ETA ") and lines[1].startswith("[2/2] b (100%), ETA ")


def test_layouts_built_in_the_worker_process_equal_those_built_here():
    from foqlens.layouts import Random
    from foqlens.quality import batches, layout_pool, policy_layouts

    policy = Random(0.3, np.arange(1, 41, dtype=np.int64) * 64, seed=4)
    groups = list(batches(7, 3))
    with layout_pool() as pool:
        remote = pool.submit(policy_layouts, policy, groups).result()
    local = policy_layouts(policy, groups)
    assert len(remote) == len(local) and all(np.array_equal(r, l) for r, l in zip(remote, local))


def test_evaluate_all_takes_queued_layouts_in_order_and_refuses_a_short_queue(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    import foqlens.quality as quality

    seen = []
    monkeypatch.setattr(quality, "evaluate_policy", lambda *a, layouts, **k: seen.append((a[4].name, layouts)) or [{}])
    policies = [SimpleNamespace(name=n, levels=lambda idx, n=n: np.full(len(idx), ord(n))) for n in "abc"]
    groups = list(quality.batches(5, 2))
    with ThreadPoolExecutor(max_workers=1) as pool:
        queued = quality.submit_layouts(pool, policies, groups)
        quality.evaluate_all(None, None, None, [None] * 5, policies, [], 2, log=lambda _: None, pool=pool, layouts=queued)
        assert [name for name, _ in seen] == ["a", "b", "c"]
        assert all(len(lay) == len(groups) and lay[0][0] == ord(name) for name, lay in seen)  # each policy got its own
        with pytest.raises(ValueError):  # a queue shorter than the policies is an error, not a silent skip
            quality.evaluate_all(None, None, None, [None] * 5, policies, [], 2, log=lambda _: None, pool=pool,
                                 layouts=queued[:2])


def test_a_new_fill_is_a_new_class_and_needs_no_change_to_the_policies():
    from typing import ClassVar

    from foqlens.layouts import BackboneFill, TopicMask

    class Last:  # ranks the last blocks first
        kind: ClassVar[str] = "last"

        def label(self, source: str) -> str:
            return f"last_{source}"

        def vector(self, index: int, n_blocks: int) -> np.ndarray:
            return np.arange(n_blocks, dtype=float)

    weights = np.full(6, 10)
    backbone = np.array([9.0, 8.0, 0.0, 0.0, 0.0, 0.0])
    fill = BackboneFill(Last(), "pooled", 4 / 6, 0.5, backbone, weights, coarse=Level.ZERO)
    sharp = np.flatnonzero(fill.levels(np.array([0]))[0] == Level.BF16).tolist()
    assert sharp == [0, 1, 4, 5] and fill.name == "bb0.50_last_pooled_0.667"
    mask = TopicMask(Last(), "pooled", 2 / 6, weights, Level.ZERO)
    assert np.flatnonzero(mask.levels(np.array([0]))[0] == Level.BF16).tolist() == [4, 5]
    assert mask.name == "last_topic_pooled_0.333"


def test_flat_policies_are_exactly_their_budget_rules():
    from foqlens.layouts import BackboneFill, OtherTopic, OwnTopic, RandomFill, TopicMask, TopicMeans, _rng

    rng = np.random.default_rng(3)
    domains = ("a",) * 3 + ("b",) * 3
    partner = {"a": "b", "b": "a"}
    n, share, cut = 40, 0.4, 0.5
    weights = rng.integers(64, 640, size=n)
    backbone = rng.normal(size=n)
    means = TopicMeans(rng.normal(size=(len(domains), n)), domains)
    idx = np.arange(len(domains))
    direct_topic = lambda topic_of: np.stack(  # noqa: E731
        [bg.to_levels(bg.directed(means.mean(i, topic_of(i)), weights, share), lo=Level.ZERO) for i in idx])
    assert np.array_equal(TopicMask(OwnTopic(means), "s", share, weights, Level.ZERO).levels(idx), direct_topic(lambda i: domains[i]))
    assert np.array_equal(TopicMask(OtherTopic(means, partner), "s", share, weights, Level.ZERO).levels(idx),
                          direct_topic(lambda i: partner[domains[i]]))
    random_fill = BackboneFill(RandomFill(share, seed=7), "-", share, cut, backbone, weights, coarse=Level.ZERO)
    direct_random = np.stack([bg.to_levels(bg.layered(backbone, _rng(7, int(i), share, salt=1).random(n), weights, share, cut),
                                           lo=Level.ZERO) for i in idx])
    assert np.array_equal(random_fill.levels(idx), direct_random)


def test_summarize_averages_whatever_the_metric_names_overall_and_per_domain():
    questions = [Question("x", "p", 0), Question("x", "p", 0), Question("y", "p", 0)]
    rows = [{"accuracy": 1.0, "logprob": -0.1, "mean_bits": 8.0}, {"accuracy": 0.0, "logprob": -2.0, "mean_bits": 8.0},
            {"accuracy": 1.0, "logprob": -0.3, "mean_bits": 8.0}]
    s = summarize({"p": rows}, questions)["p"]
    assert s["accuracy"] == pytest.approx(2 / 3) and s["mean_bits"] == 8.0
    assert s["by_domain"]["accuracy"] == {"x": 0.5, "y": 1.0} and s["by_domain"]["logprob"]["y"] == pytest.approx(-0.3)


def test_letter_choice_scores_the_right_letter(monkeypatch):
    import foqlens.evaluate as ev

    fake = np.log(np.array([[0.1, 0.6, 0.2, 0.1], [0.7, 0.1, 0.1, 0.1]]))
    monkeypatch.setattr(ev, "letter_logprobs_batch", lambda model, tok, prompts, ids: fake[: len(prompts)])
    rows = ev.LetterChoice((1, 2, 3, 4)).score(None, None, [Question("x", "p", 1), Question("y", "q", 2)])
    assert rows[0] == {"accuracy": 1.0, "logprob": pytest.approx(np.log(0.6)), "picked": 1.0}
    assert rows[1] == {"accuracy": 0.0, "logprob": pytest.approx(np.log(0.1)), "picked": 0.0}
    assert ev.LetterChoice.primary == "logprob"


def test_the_picked_letter_is_the_top_one_whether_or_not_it_is_right(monkeypatch):
    """A model that leans on a letter shows as a skew in `picked`; accuracy alone cannot tell."""
    import foqlens.evaluate as ev

    fake = np.log(np.array([[0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.6, 0.2]]))
    monkeypatch.setattr(ev, "letter_logprobs_batch", lambda model, tok, prompts, ids: fake[: len(prompts)])
    rows = ev.LetterChoice((1, 2, 3, 4)).score(None, None, [Question("x", "p", 0), Question("y", "q", 2)])
    assert [r["picked"] for r in rows] == [2.0, 2.0]        # both leaned to C
    assert [r["accuracy"] for r in rows] == [0.0, 1.0]      # one of them was right anyway


def test_a_new_metric_plugs_into_the_evaluation_without_touching_it():
    from types import SimpleNamespace

    from foqlens.quality import evaluate_policy

    class AnswerLength:
        name, primary = "answer_length", "length"

        def score(self, model, tokenizer, questions):
            return [{"length": float(len(q.prompt))} for q in questions]

    ctl = SimpleNamespace(set_layout=lambda layout: None, mean_bits=lambda: 6.0)
    policy = SimpleNamespace(name="any", levels=lambda idx: np.zeros((len(idx), 3), dtype=np.uint8))
    questions = [Question("x", "ab", 0), Question("y", "abcd", 0), Question("y", "a", 0)]
    rows = evaluate_policy(None, None, ctl, questions, policy, AnswerLength(), batch_size=2)
    assert rows == [{"length": 2.0, "mean_bits": 6.0}, {"length": 4.0, "mean_bits": 6.0}, {"length": 1.0, "mean_bits": 6.0}]


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
    mon = GpuMonitor(sampler=lambda: (0.0, 0.0, 0.0, 0.0))
    assert mon.summary() == {"samples": 0}
    mon.samples = [(50.0, 1000.0, 60.0, 200.0), (90.0, 3000.0, 78.0, 330.0), (100.0, 2000.0, 72.0, 310.0)]
    mon.torch_peak_mib = 1500.0
    s = mon.summary()
    assert s["utilization_mean"] == pytest.approx(80.0) and s["utilization_median"] == 90.0
    assert s["memory_reserved_peak_mib"] == 3000.0 and s["memory_allocated_peak_mib"] == 1500.0 and s["samples"] == 3
    assert s["temperature_peak_c"] == 78.0 and s["temperature_mean_c"] == pytest.approx(70.0)
    assert s["power_peak_w"] == 330.0 and s["power_mean_w"] == pytest.approx(280.0)


def test_gpu_monitor_samples_in_the_background_until_exit():
    with GpuMonitor(interval=0.001, sampler=lambda: (42.0, 7.0, 55.0, 120.0)) as mon:
        while len(mon.samples) < 3:
            pass
    taken = len(mon.samples)
    assert taken >= 3 and mon.samples[0] == (42.0, 7.0, 55.0, 120.0)
    assert len(mon.samples) == taken  # the thread has stopped


def test_a_new_mask_source_plugs_into_the_bench_without_touching_it():
    from types import SimpleNamespace

    from foqlens.pipeline import Bench

    class PromptLength:
        name, batch_size = "prompt_length", 2

        def score_batch(self, model, tokenizer, texts):
            return [np.full(3, float(len(t))) for t in texts]

    def one_token_per_char(texts):
        return {"input_ids": [list(t) for t in texts]}

    levels = []
    bench = Bench(model=None, tokenizer=one_token_per_char, ctl=SimpleNamespace(set_all=levels.append))
    masks = bench.masks(["a", "bbb", "cc"], [PromptLength()])
    assert list(masks) == ["prompt_length"] and masks["prompt_length"][:, 0].tolist() == [1.0, 3.0, 2.0]
    assert levels == [Level.BF16]  # masks are computed with every block at bf16
