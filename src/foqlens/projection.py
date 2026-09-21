"""The projection of issue #18: carry a signal read on a few units onto every block (open question 2 of the problem statement).

A working source reads the first layers only, and on its own units - heads, groups of neurons; the mask is
needed on every block. The projection is fitted on calibration questions to what a comparison score says
of every block (ShadowLLM, arXiv 2406.16635: one predictor at layer 1 recovers the Taylor score of every
head and neuron of all layers at Spearman about 0.85): a ridge regression from the units' signals to the
blocks' scores, centred on the calibration means,

    sigma = mean_tau + (A - mean_A) P,    P = (Ac^T Ac + alpha I)^-1 Ac^T tau_c.

A co-activation matrix maps activity onto activity, and activity is not what the comparison score ranks;
the regression learns how a unit's signal predicts it.

Invariants:
- Invariant: with alpha = 0 and signals of full column rank, an exact linear relation is recovered.
- Invariant: alpha shrinks the map towards the calibration mean - a larger alpha never gives a larger P.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Projection:
    weights: torch.Tensor  # P [units, n_blocks]
    unit_mean: torch.Tensor  # [units]
    block_mean: torch.Tensor  # [n_blocks]

    @classmethod
    def fit(cls, signals: np.ndarray | torch.Tensor, scores: np.ndarray | torch.Tensor, alpha: float,
            device: str | torch.device = "cpu") -> Projection:
        """Ridge regression of block scores [questions, n_blocks] on unit signals [questions, units]."""
        if alpha < 0:
            raise ValueError(f"ridge alpha {alpha} below zero")
        a = torch.as_tensor(np.asarray(signals), dtype=torch.float64, device=device)
        t = torch.as_tensor(np.asarray(scores), dtype=torch.float64, device=device)
        unit_mean, block_mean = a.mean(dim=0), t.mean(dim=0)
        ac, tc = a - unit_mean, t - block_mean
        gram = ac.T @ ac + alpha * torch.eye(a.shape[1], dtype=a.dtype, device=device)
        weights = torch.linalg.solve(gram, ac.T @ tc)
        return cls(weights, unit_mean, block_mean)

    def apply(self, signals: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Block scores [questions, n_blocks] from unit signals [questions, units]."""
        a = torch.as_tensor(np.asarray(signals), dtype=self.weights.dtype, device=self.weights.device)
        return self.block_mean + (a - self.unit_mean) @ self.weights
