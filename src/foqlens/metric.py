"""Distances between weight blocks: the space the zones live in (issue #4).

A zone is a ball in a metric on blocks. The rules of the zones (zones.py, issue #19) never look at
coordinates - only at how far every block is from a zone's center - so a metric is anything that
answers that, and a new space is a new class behind BlockMetric.

- CoactivationMetric (M1): d(a, b) = sqrt(2 (1 - corr(a, b))) between the blocks' mask profiles over
  calibration questions - the distance weight_map.coactivation_map cuts down to two principal axes,
  here in full. Profiles are standardized per block and scaled to unit length, so the distance is the
  Euclidean distance between them.
- neighbour_table: the k nearest blocks of every block in a metric, made symmetric - the graph the
  peaks and hills of a mask are found on (zones.find_ball_zones) and the paths of M4 run along.
- ResistiveMetric (M4): shortest paths along a neighbour table, every edge as long as its base
  distance divided by the harmonic mean of the query's activity at its two ends - two half-edges in
  series (Kirchhoff), so one quiet end closes the edge and one loud end cannot open it more than twice.

Everything is torch on the metric's device, computed in chunks of rows; nothing reads a tensor value on
the host except the loop of ResistiveMetric.distances, which runs outside any forward pass.

Invariants:
- Invariant: CoactivationMetric distances are Euclidean: symmetric, zero on the diagonal, the triangle
  inequality holds (tests), and sqrt(2 (1 - corr)) for blocks whose profile varies.
- Invariant: scaling or shifting one block's profile does not move it, as on the weight map.
- Invariant: a neighbour table is symmetric and a block is never its own neighbour.
- Invariant: with activity 1 everywhere a ResistiveMetric is the shortest path of its base table.
- Invariant: an edge's conductance lies between the smaller activity of its ends and twice that.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

EPS = 1e-12
PAD = -1  # a missing neighbour in a padded table
ROW_CHUNK = 1024  # rows of distances built at once: 1024 x 14 708 in float32 is 60 MB


class BlockMetric(Protocol):
    n_blocks: int

    def distances(self, sources: torch.Tensor) -> torch.Tensor:
        """Distances from the blocks `sources` [k] to every block: [k, n_blocks]."""
        ...


def unit_profiles(masks: np.ndarray | torch.Tensor, device: str | torch.device = "cpu") -> torch.Tensor:
    """Raw masks [questions, n_blocks] -> every block's profile standardized and of unit length: [n_blocks, questions].

    A block whose mask never changes has no profile and becomes the zero vector: distance 1 to every
    block that varies, 0 to another constant block.
    """
    # float64: the distance of two near-identical profiles is the root of 1 + 1 - 2 corr, and in float32
    # that difference loses the digits the root then blows up (5e-4 on the diagonal, measured)
    x = torch.as_tensor(np.asarray(masks), dtype=torch.float64, device=device).T
    x = x - x.mean(dim=1, keepdim=True)
    norm = x.norm(dim=1, keepdim=True)
    return torch.where(norm > EPS, x / norm.clamp_min(EPS), torch.zeros_like(x))


class CoactivationMetric:
    """M1: the co-activation distance over the full profile of calibration masks."""

    def __init__(self, masks: np.ndarray | torch.Tensor, device: str | torch.device = "cpu"):
        self.profiles = unit_profiles(masks, device)
        self.n_blocks = self.profiles.shape[0]
        self._sq = (self.profiles * self.profiles).sum(dim=1)  # 1, or 0 for a constant block

    def distances(self, sources: torch.Tensor) -> torch.Tensor:
        sources = torch.as_tensor(sources, device=self.profiles.device)
        gram = self.profiles[sources] @ self.profiles.T
        d = (self._sq[sources, None] + self._sq[None] - 2 * gram).clamp_min(0).sqrt()
        d[torch.arange(len(sources), device=d.device), sources] = 0.0  # a block is at no distance from itself, exactly
        return d.float()

    def diameter(self) -> float:
        """The largest distance between two blocks, over all pairs in chunks of rows."""
        every = torch.arange(self.n_blocks, device=self.profiles.device)
        return max(float(self.distances(rows).max()) for rows in every.split(ROW_CHUNK))


def neighbour_table(metric: BlockMetric, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The k nearest blocks of every block, made symmetric: indices [n_blocks, width] padded with PAD, and their distances.

    A block's own entry is never among them. The union of "a is among b's k nearest" and the reverse
    keeps the graph undirected, so a path found along it runs both ways.
    """
    n = metric.n_blocks
    device = metric.distances(torch.tensor([0])).device
    heads, tails, lengths = [], [], []
    for rows in torch.arange(n, device=device).split(ROW_CHUNK):
        d = metric.distances(rows)
        d[torch.arange(len(rows), device=device), rows] = torch.inf  # never one's own neighbour
        near = d.topk(min(k, n - 1), dim=1, largest=False)
        heads.append(rows.repeat_interleave(near.indices.shape[1]))
        tails.append(near.indices.flatten())
        lengths.append(near.values.flatten())
    a, b, w = torch.cat(heads), torch.cat(tails), torch.cat(lengths)
    return _padded(torch.cat([a, b]), torch.cat([b, a]), torch.cat([w, w]), n)


def _padded(heads: torch.Tensor, tails: torch.Tensor, lengths: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Edges (head, tail, length), duplicates dropped -> a table [n, width] of tails per head, padded with PAD."""
    key = heads * n + tails
    key, first = np.unique(key.cpu().numpy(), return_index=True)
    heads, tails, lengths = heads[first], tails[first], lengths[first]
    counts = torch.bincount(heads, minlength=n)
    width = int(counts.max())
    slot = torch.arange(len(heads), device=heads.device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    table = torch.full((n, width), PAD, dtype=torch.long, device=heads.device)
    table_lengths = torch.full((n, width), torch.inf, device=heads.device)
    table[heads, slot] = tails
    table_lengths[heads, slot] = lengths
    return table, table_lengths


def harmonic_conductance(activity: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Conductance of every edge of a table [n, width]: the harmonic mean of the activities at its ends, 0 on padding.

    Two half-edges of resistance 1 / (2 a) in series: 2 a_i a_j / (a_i + a_j).
    """
    a_i = activity[:, None].expand_as(table)
    a_j = activity[table.clamp_min(0)]
    c = 2 * a_i * a_j / (a_i + a_j).clamp_min(EPS)
    return torch.where(table == PAD, torch.zeros_like(c), c)


def geodesic(table: torch.Tensor, lengths: torch.Tensor) -> ResistiveMetric:
    """Shortest paths along the block graph with no medium: every edge as long as its base distance.

    The surface the zones live on is this graph (owner, 2026-09-16): a zone reaching R along it covers
    whatever blocks lie within R by its paths - an arbitrary figure, not a ball cut across the space.
    """
    return ResistiveMetric(table, lengths, torch.ones(table.shape[0], device=lengths.device))


def sweep_width(metric: BlockMetric, start: int = 0, sweeps: int = 4) -> float:
    """The width of the network along a metric by repeated farthest-block sweeps.

    Each sweep measures from the block the last one ended farthest at; the largest finite distance
    seen is a lower bound of the diameter, exact on a tree after two sweeps. All-pairs paths on
    14 708 blocks would cost 14 708 sweeps.
    """
    far, width = start, 0.0
    for _ in range(sweeps):
        d = metric.distances(torch.tensor([far]))[0]
        finite = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
        width, far = max(width, float(finite.max())), int(finite.argmax())
    return width


class ResistiveMetric:
    """M4: shortest paths along a neighbour table through a medium whose conductance is the query's activity."""

    def __init__(self, table: torch.Tensor, lengths: torch.Tensor, activity: torch.Tensor):
        self.table = table
        self.n_blocks = table.shape[0]
        c = harmonic_conductance(activity.to(lengths.device, torch.float32), table)
        self.edges = torch.where(c > 0, lengths / c.clamp_min(EPS), torch.full_like(lengths, torch.inf))

    def distances(self, sources: torch.Tensor) -> torch.Tensor:
        """Bellman-Ford by rows: relax every block from its neighbours until nothing shortens."""
        sources = torch.as_tensor(sources, device=self.table.device)
        d = torch.full((len(sources), self.n_blocks), torch.inf, device=self.table.device)
        d[torch.arange(len(sources), device=d.device), sources] = 0.0
        via = self.table.clamp_min(0)
        for _ in range(self.n_blocks):
            relaxed = torch.minimum(d, (d[:, via] + self.edges[None]).amin(dim=2))
            if torch.equal(relaxed, d):
                break
            d = relaxed
        return d
