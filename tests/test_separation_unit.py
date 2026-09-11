"""Step 0 metrics on synthetic vectors: no model, CPU."""

import numpy as np
import torch

from foqlens.separation import mean_pool, per_layer, separation


def two_clusters(n: int = 10, dim: int = 64, spread: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=dim), rng.normal(size=dim)
    x = np.vstack([a + spread * rng.normal(size=(n, dim)), b + spread * rng.normal(size=(n, dim))])
    return x, ["a"] * n + ["b"] * n


def test_separated_clusters_are_detected():
    x, labels = two_clusters()
    s = separation(x, labels)
    assert s.cos_in["a"] > s.cos_between and s.cos_in["b"] > s.cos_between
    assert s.silhouette > 0.5
    assert s.ari == 1.0
    assert s.separates


def test_cosine_direction_and_clustering_are_separate_fields():
    x, labels = two_clusters()
    d = separation(x, labels).as_dict()
    assert d["cosine_direction"] and d["clusters"] and d["separates"]
    # the cosine direction can hold while clustering fails: the two must not be conflated
    from foqlens.separation import Separation

    weak = Separation(cos_in={"a": 0.10, "b": 0.12}, cos_between=0.05, silhouette=-0.04, ari=-0.01)
    assert weak.cosine_direction and not weak.clusters and not weak.separates


def test_permutation_test_tells_clusters_from_noise():
    from foqlens.separation import permutation_test

    x, labels = two_clusters(n=10, spread=0.5)
    clear = permutation_test(x, labels, n_perm=200, seed=0)
    assert clear["margin"] > 0 and clear["p_value"] < 0.01
    rng = np.random.default_rng(3)
    noise = permutation_test(rng.normal(size=(20, 64)), ["a"] * 10 + ["b"] * 10, n_perm=200, seed=0)
    assert noise["p_value"] > 0.05


def test_one_blob_with_random_labels_does_not_separate():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 64))
    labels = ["a", "b"] * 10
    s = separation(x, labels)
    assert s.ari < 0.5
    assert s.silhouette < 0.1


def test_centering_removes_a_shared_direction():
    x, labels = two_clusters(spread=0.1)
    shared = 50.0 * np.ones((1, x.shape[1]))
    raw = separation(x + shared, labels)
    centered = separation(x + shared - (x + shared).mean(axis=0, keepdims=True), labels)
    # a big common offset pushes every cosine toward 1 and hides the gap; centering restores it
    assert raw.cos_between > 0.9
    assert centered.cos_in["a"] - centered.cos_between > raw.cos_in["a"] - raw.cos_between


def test_mean_pool_and_per_layer_shapes():
    hidden = tuple(torch.randn(7, 16, dtype=torch.bfloat16) for _ in range(4))
    pooled = mean_pool(hidden)
    assert pooled.shape == (4, 16) and pooled.dtype == np.float32
    x, labels = two_clusters(n=5, dim=16)
    stacked = np.stack([x, x, x], axis=1)  # [n, layers, dim]
    layers = per_layer(stacked, labels)
    assert [entry["layer"] for entry in layers] == [0, 1, 2]
    assert set(layers[0]) == {"layer", "raw", "centered"}
    assert layers[0]["raw"]["separates"] is True
