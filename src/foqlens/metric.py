"""Distances between weight blocks: the space the zones live in (issue #4).

The surface the zones live on is a graph of blocks, and a zone is every block within a radius of its
center along it (issue #4). The rules of the zones (zones.py, issue #19) never look at coordinates - only
at how far every block is from a zone's center - so a metric is anything that answers that, and a new
space is a new class behind BlockMetric.

- CoactivationMetric (M1): d(a, b) = sqrt(2 (1 - corr(a, b))) between the blocks' mask profiles over
  calibration questions - the distance weight_map.coactivation_map cuts down to two principal axes,
  here in full. Profiles are standardized per block and scaled to unit length, so the distance is the
  Euclidean distance between them.
- neighbour_table: the k nearest blocks of every block in a metric, made symmetric by union - the graph
  the peaks and hills of a mask are found on (graph_zones.find_graph_zones) and the paths run along.
  The union grows hubs; mutual_nicdm_table is the construction with fewer of them (hubness, #4), and
  graph_report measures a graph for it.
- ResistiveMetric: shortest paths along a neighbour table, every edge as long as its base distance
  divided by its conductance. The conductance is replaceable (Conductance): HarmonicConductance (M4) is
  the harmonic mean of the query's activity at the two ends - two half-edges in series (Kirchhoff), so
  one quiet end closes the edge and one loud end cannot open it more than twice; JumpConductance (M4b,
  Perona-Malik) falls with the jump of activity between the ends, so a zone stops at the border of the
  active and the quiet.

Everything is torch on the metric's device, computed in chunks of rows; nothing reads a tensor value on
the host except the loop of ResistiveMetric.distances, which runs outside any forward pass.

Invariants:
- Invariant: CoactivationMetric distances are Euclidean: symmetric, zero on the diagonal, the triangle
  inequality holds (tests), and sqrt(2 (1 - corr)) for blocks whose profile varies.
- Invariant: scaling or shifting one block's profile does not move it, as on the weight map.
- Invariant: a neighbour table is symmetric and a block is never its own neighbour, for either construction.
- Invariant: mutual_nicdm_table is connected: whatever the k-nearest lists leave apart is joined by nearest pairs.
- Invariant: with activity 1 everywhere a ResistiveMetric is the shortest path of its base table, for either conductance.
- Invariant: a harmonic edge conducts between the smaller activity of its ends and twice that.
- Invariant: a jump edge conducts 1 where its ends are equally active and less the larger the jump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.stats import skew

EPS = 1e-12
PAD = -1  # a missing neighbour in a padded table
MAD_TO_SIGMA = 1.4826  # a normal distribution's standard deviation per median absolute deviation (Black et al. 1998)
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


def nearest(metric: BlockMetric, k: int, scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Every block's k nearest blocks and their distances, [n_blocks, k] each, never the block itself.

    With `scale` [n_blocks] the distance is rescaled first, d(a, b) / sqrt(scale_a scale_b) (NICDM).
    """
    n = metric.n_blocks
    device = metric.distances(torch.tensor([0])).device
    indices, distances = [], []
    for rows in torch.arange(n, device=device).split(ROW_CHUNK):
        d = metric.distances(rows)
        if scale is not None:
            d = d / torch.sqrt(scale[rows, None] * scale[None]).clamp_min(EPS)
        d[torch.arange(len(rows), device=device), rows] = torch.inf  # never one's own neighbour
        near = d.topk(min(k, n - 1), dim=1, largest=False)
        indices.append(near.indices)
        distances.append(near.values)
    return torch.cat(indices), torch.cat(distances)


def _directed(indices: torch.Tensor, distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """k-nearest lists as directed edges (head, tail, length)."""
    heads = torch.arange(len(indices), device=indices.device).repeat_interleave(indices.shape[1])
    return heads, indices.flatten(), distances.flatten()


def neighbour_table(metric: BlockMetric, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The k nearest blocks of every block, made symmetric: indices [n_blocks, width] padded with PAD, and their distances.

    A block's own entry is never among them. The union of "a is among b's k nearest" and the reverse
    keeps the graph undirected, so a path found along it runs both ways.
    """
    a, b, w = _directed(*nearest(metric, k))
    return edge_table(torch.cat([a, b]), torch.cat([b, a]), torch.cat([w, w]), metric.n_blocks)


def nicdm_scale(metric: BlockMetric, k: int) -> torch.Tensor:
    """Every block's mean distance to its k nearest: the scale of NICDM, d' = d / sqrt(mu_a mu_b) (Schnitzer et al. 2012)."""
    return nearest(metric, k)[1].mean(dim=1)


def mutual_nicdm_table(metric: BlockMetric, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A graph with fewer hubs (#4): mutual nearest neighbours on the NICDM distance, its components joined along a spanning tree.

    NICDM (Schnitzer et al. 2012) rescales d by the mean distance of either end to its k nearest,
    d' = d / sqrt(mu_a mu_b): a hub sits near the mean profile, has a small mu, and every distance to it
    grows. On d' only mutual pairs are linked - a hub can no longer collect the lists of hundreds. The
    components that leaves are joined along the minimum spanning tree of the union graph on d', one edge
    per join, shortest first (Dalmia & Sia 2021) - repairing on the raw d would hang them on the hubs
    again (Flexer & Stevens 2018). What even the union on d' leaves apart (E001's masks: 32 pieces) is
    joined by Boruvka rounds - every component takes its nearest pair outside it on d' - until one
    component is left. Edge lengths are d'.
    """
    n = metric.n_blocks
    scale = nicdm_scale(metric, k)
    heads, tails, lengths = _directed(*nearest(metric, k, scale=scale))
    h, t, w = heads.cpu().numpy(), tails.cpu().numpy(), lengths.cpu().numpy()
    mutual = np.isin(h * n + t, t * n + h)
    _, parts = connected_components(coo_matrix((np.ones(mutual.sum()), (h[mutual], t[mutual])), shape=(n, n)), directed=False)
    # explicit zeros are no edge to scipy: identical profiles lie at d' = 0, so every weight is lifted by EPS
    tree = minimum_spanning_tree(coo_matrix((w + EPS, (h, t)), shape=(n, n)).tocsr().maximum(coo_matrix((w + EPS, (t, h)), shape=(n, n)).tocsr())).tocoo()
    order = np.argsort(tree.data, kind="stable")
    root = np.arange(parts.max() + 1)

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    joins = []
    for e in order:
        a, b = find(parts[tree.row[e]]), find(parts[tree.col[e]])
        if a != b:
            root[a] = b
            joins.append(e)
    ja, jb = tree.row[joins], tree.col[joins]
    jw = tree.data[joins] - EPS
    ba, bb, bw = _join_apart(metric, scale, np.concatenate([h[mutual], ja]), np.concatenate([t[mutual], jb]))
    ea = np.concatenate([h[mutual], ja, jb, ba, bb])
    eb = np.concatenate([t[mutual], jb, ja, bb, ba])
    ew = np.concatenate([w[mutual], jw, jw, bw, bw])
    device = lengths.device
    as_tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=device)  # noqa: E731
    return edge_table(as_tensor(ea, torch.long), as_tensor(eb, torch.long), as_tensor(ew, torch.float32), n)


def _join_apart(metric: BlockMetric, scale: torch.Tensor, heads: np.ndarray, tails: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edges that join a graph's components (Boruvka): every component but the largest takes its nearest outside pair on d'."""
    n = metric.n_blocks
    device = scale.device
    ea, eb, ew = [], [], []
    while True:
        graph = coo_matrix((np.ones(len(heads) + len(ea)), (np.concatenate([heads, ea]).astype(np.int64),
                                                             np.concatenate([tails, eb]).astype(np.int64))), shape=(n, n))
        n_parts, parts = connected_components(graph, directed=False)
        if n_parts == 1:
            return np.asarray(ea, dtype=np.int64), np.asarray(eb, dtype=np.int64), np.asarray(ew, dtype=np.float64)
        largest = np.bincount(parts).argmax()
        labels = torch.as_tensor(parts, device=device)
        best: dict[int, tuple[float, int, int]] = {}
        for rows in torch.as_tensor(np.flatnonzero(parts != largest), device=device).split(ROW_CHUNK):
            d = metric.distances(rows) / torch.sqrt(scale[rows, None] * scale[None]).clamp_min(EPS)
            d[labels[rows][:, None] == labels[None]] = torch.inf  # only pairs outside a block's own component
            value, target = d.min(dim=1)
            for r, v, t_ in zip(rows.tolist(), value.tolist(), target.tolist()):
                part = int(parts[r])
                if v < best.get(part, (np.inf, -1, -1))[0]:
                    best[part] = (v, r, t_)
        for v, r, t_ in best.values():
            ea.append(r)
            eb.append(t_)
            ew.append(v)


def graph_report(table: torch.Tensor | np.ndarray, indices: torch.Tensor | np.ndarray | None = None) -> dict:
    """How a block graph measures up for hubness (#4): degrees, components, and - given the k-nearest lists - the k-occurrence."""
    t = np.asarray(table.cpu() if isinstance(table, torch.Tensor) else table)
    real = t != PAD
    degree = real.sum(axis=1)
    heads = np.repeat(np.arange(len(t)), t.shape[1])[real.ravel()]
    n_parts, parts = connected_components(coo_matrix((np.ones(len(heads)), (heads, t.ravel()[real.ravel()])), shape=(len(t),) * 2), directed=False)
    report = {"degree_median": float(np.median(degree)), "degree_p99": float(np.percentile(degree, 99)),
              "degree_max": int(degree.max()), "components": int(n_parts),
              "largest_share": float(np.bincount(parts).max() / len(t)), "isolated_share": float((degree == 0).mean())}
    if indices is not None:
        lists = np.asarray(indices.cpu() if isinstance(indices, torch.Tensor) else indices)
        occurrence = np.bincount(lists.ravel(), minlength=len(lists))
        rows = np.repeat(np.arange(len(lists)), lists.shape[1])
        pairs = set(zip(rows.tolist(), lists.ravel().tolist()))
        report |= {"k_occurrence_skewness": float(skew(occurrence)), "never_neighbour_share": float((occurrence == 0).mean()),
                   "symmetric_share": float(np.mean([(b, a) in pairs for a, b in pairs]))}
    return report


def edge_table(heads: torch.Tensor, tails: torch.Tensor, lengths: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
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


class Conductance(Protocol):
    def edges(self, table: torch.Tensor) -> torch.Tensor:
        """The conductance of every edge of a table [n, width]; 0 on padding."""
        ...


def harmonic_conductance(activity: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Conductance of every edge of a table [n, width]: the harmonic mean of the activities at its ends, 0 on padding.

    Two half-edges of resistance 1 / (2 a) in series: 2 a_i a_j / (a_i + a_j).
    """
    a_i = activity[:, None].expand_as(table)
    a_j = activity[table.clamp_min(0)]
    c = 2 * a_i * a_j / (a_i + a_j).clamp_min(EPS)
    return torch.where(table == PAD, torch.zeros_like(c), c)


@dataclass(frozen=True)
class HarmonicConductance:
    """M4: an edge conducts the harmonic mean of the activity at its ends."""

    activity: torch.Tensor

    def edges(self, table: torch.Tensor) -> torch.Tensor:
        return harmonic_conductance(self.activity, table)


@dataclass(frozen=True)
class JumpConductance:
    """M4b (Perona-Malik, read through Black et al. 1998): exp(-(a_i - a_j)^2 / K^2) - an edge closes with the jump.

    K is the robust scale of the jumps over all edges, MAD_TO_SIGMA times their median absolute
    deviation, unless given.
    """

    activity: torch.Tensor
    scale: float | None = None

    def edges(self, table: torch.Tensor) -> torch.Tensor:
        real = table != PAD
        jump = self.activity[:, None].expand_as(table) - self.activity[table.clamp_min(0)]
        if self.scale is None:
            real_jumps = jump[real]
            k = MAD_TO_SIGMA * (real_jumps - real_jumps.median()).abs().median()
        else:
            k = torch.tensor(float(self.scale))
        c = torch.exp(-((jump / k.clamp_min(EPS)) ** 2))
        return torch.where(real, c, torch.zeros_like(c))


class Surface(Protocol):
    """The graph the zones live on and the distances a question's zones reach by along it."""

    table: np.ndarray  # [n_blocks, width] neighbours, padded with PAD: where peaks and hills are found

    def metric(self, index: int) -> BlockMetric:
        """The distances the zones of question `index` reach by."""
        ...


@dataclass(frozen=True)
class StillSurface:
    """One set of distances for every question: the graph at rest (geodesic) or any fixed metric."""

    table: np.ndarray
    fixed: BlockMetric

    def metric(self, index: int) -> BlockMetric:
        return self.fixed


class MediumSurface:
    """M4 (M4b by `medium`) per question: the graph's paths through a medium of the question's own activity.

    activity [questions, n_blocks] is how loud every block is on a question (an output energy),
    background [n_blocks] the same on average; the medium is their ratio, so a block that is loud on
    every question - a massive activation - conducts no better than a quiet one that is loud here.
    """

    def __init__(self, table: torch.Tensor, lengths: torch.Tensor, activity: np.ndarray, background: np.ndarray,
                 medium: type = HarmonicConductance):
        self.table = table.cpu().numpy()
        self._table, self._lengths, self._medium = table, lengths, medium
        ratio = np.asarray(activity, dtype=np.float64) / np.maximum(np.asarray(background, dtype=np.float64), EPS)
        self._ratio = torch.as_tensor(ratio, dtype=torch.float32, device=lengths.device)

    def metric(self, index: int) -> ResistiveMetric:
        return ResistiveMetric(self._table, self._lengths, self._medium(self._ratio[index]))


def geodesic(table: torch.Tensor, lengths: torch.Tensor) -> ResistiveMetric:
    """Shortest paths along the block graph with no medium: every edge as long as its base distance.

    The surface the zones live on is this graph (owner, 2026-09-16): a zone reaching R along it covers
    whatever blocks lie within R by its paths - an arbitrary figure, not a ball cut across the space.
    """
    return ResistiveMetric(table, lengths, HarmonicConductance(torch.ones(table.shape[0], device=lengths.device)))


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
    """Shortest paths along a neighbour table through a medium: every edge as long as its base distance over its conductance."""

    def __init__(self, table: torch.Tensor, lengths: torch.Tensor, conductance: Conductance):
        self.table = table
        self.n_blocks = table.shape[0]
        c = conductance.edges(table).to(lengths.device, torch.float32)
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
