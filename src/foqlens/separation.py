"""Step 0 metrics: do topics separate in the model's representations.

One vector per prompt and per layer (hidden states mean-pooled over tokens), then per layer:

- cos_in per domain and cos_between over all cross-domain pairs (the preregistered direction:
  cos_in(A) > cos(A, B));
- silhouette with cosine distance;
- ARI of k-means with k = number of domains against the domain labels (0 is chance).

Every metric is reported twice: on raw vectors and on vectors centered by the mean over all
prompts. Mean-pooled hidden states share a large common direction, which lifts every cosine
toward 1; centering removes it. Both variants are fixed here before any run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def mean_pool(hidden: tuple[torch.Tensor, ...]) -> np.ndarray:
    """Hidden states of one prompt ([seq, dim] per layer) -> [layers, dim], mean over tokens, float32."""
    return torch.stack([h.float().mean(dim=0) for h in hidden]).cpu().numpy()


@dataclass
class Separation:
    cos_in: dict[str, float]
    cos_between: float
    silhouette: float
    ari: float

    @property
    def cosine_direction(self) -> bool:
        """The preregistered step 1 criterion: every cos_in above cos_between."""
        return all(v > self.cos_between for v in self.cos_in.values())

    @property
    def clusters(self) -> bool:
        """Unsupervised clustering recovers the labels better than chance (ARI > 0)."""
        return self.ari > 0

    @property
    def separates(self) -> bool:
        """The preregistered step 0 check: the cosine direction and clustering above chance, both."""
        return self.cosine_direction and self.clusters

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "cosine_direction": self.cosine_direction,
            "clusters": self.clusters,
            "separates": self.separates,
        }


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def separation(vectors: np.ndarray, labels: list[str], seed: int = 0) -> Separation:
    """Separation of labelled vectors [n, dim]; needs at least two domains with two vectors each."""
    labels_arr = np.asarray(labels)
    domains = sorted(set(labels))
    unit = _unit(vectors.astype(np.float64))
    cos = unit @ unit.T
    upper = np.triu(np.ones_like(cos, dtype=bool), k=1)
    same = labels_arr[:, None] == labels_arr[None, :]

    cos_in = {}
    for d in domains:
        mask = upper & same & (labels_arr[:, None] == d)
        cos_in[d] = float(cos[mask].mean())
    cos_between = float(cos[upper & ~same].mean())

    silhouette = float(silhouette_score(unit, labels_arr, metric="cosine"))
    clusters = KMeans(n_clusters=len(domains), n_init=10, random_state=seed).fit_predict(unit)
    ari = float(adjusted_rand_score(labels_arr, clusters))
    return Separation(cos_in=cos_in, cos_between=cos_between, silhouette=silhouette, ari=ari)


def permutation_test(vectors: np.ndarray, labels: list[str], n_perm: int = 1000, seed: int = 0) -> dict:
    """Is the margin min(cos_in) - cos_between larger than with shuffled labels? p = (1 + #perm >= obs) / (1 + n_perm)."""
    unit = _unit(vectors.astype(np.float64))
    cos = unit @ unit.T
    upper = np.triu(np.ones_like(cos, dtype=bool), k=1)
    domains = sorted(set(labels))

    def margin(lab: np.ndarray) -> float:
        same = lab[:, None] == lab[None, :]
        cos_in = [cos[upper & same & (lab[:, None] == d)].mean() for d in domains]
        return float(min(cos_in) - cos[upper & ~same].mean())

    observed = margin(np.asarray(labels))
    rng = np.random.default_rng(seed)
    permuted = np.array([margin(rng.permutation(labels)) for _ in range(n_perm)])
    p_value = float((1 + (permuted >= observed).sum()) / (1 + n_perm))
    return {"margin": observed, "p_value": p_value, "n_perm": n_perm, "seed": seed}


def per_layer(pooled: np.ndarray, labels: list[str]) -> list[dict]:
    """pooled [n_prompts, layers, dim] -> for every layer: raw and centered separation."""
    out = []
    for layer in range(pooled.shape[1]):
        x = pooled[:, layer, :]
        out.append(
            {
                "layer": layer,
                "raw": separation(x, labels).as_dict(),
                "centered": separation(x - x.mean(axis=0, keepdims=True), labels).as_dict(),
            }
        )
    return out
