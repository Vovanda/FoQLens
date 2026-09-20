"""The bridge from the address to a precision field: the address names the question (foqlens.activity), the oracles say
what map its answer needs (foqlens.precision_field), and this is what carries one to the other. Fitted on one part of
the corpus, read on another, never on the part it was fitted on.

Every bridge takes the questions' addresses over the groups [questions, groups] and their maps' values [questions,
groups], and gives a value a group for a question it has not seen:

- Static: the mean map of the fitted questions, the same for every question - what the address adds is measured
  against it, and where it adds nothing the map belongs in the model, not in the regulator.
- Topic: the mean map of the questions of the same corpus - zones of a topic, a mixture of experts on a domain.
- Nearest: the mean map of the `k` fitted questions whose address is closest by cosine - the address looks its
  question up among those whose map is known.
- Ridge: a linear map from the address's excess to the map's, one matrix for every group (foqlens.projection's rule).

What a bridge gives is the field itself, a value in 0 ... 1 a group; the mapper over it (`to_levels`) takes every value
to the nearest rung of the ladder. At inference there is no answer to choose a threshold by, so one shift of the whole
field is fixed on the fitted questions - the shift whose maps spend the bits asked of them (`shift_for_bits`) - and
every question is read at that one price.

Invariants:
- Invariant: a bridge fitted on questions gives their own maps back no worse than the static one does on them.
- Invariant: Nearest with k = 1 on a question of the fitted set gives that question's own map.
- Invariant: to_levels never leaves the ladder, and a larger shift never reads a group finer... coarser: the bits it
  spends do not fall when the shift grows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foqlens.precision_field import FIELD_VALUE, RUNGS, TOP


def excess(address: np.ndarray) -> np.ndarray:
    """The address over the questions' mean - what this question asks beyond what every question does."""
    return address - address.mean(axis=0, keepdims=True)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine of every row of a against every row of b: [len(a), len(b)]."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-12)
    return a @ b.T


@dataclass
class Static:
    """The mean map of the fitted questions, whatever the address."""

    name: str = "static"

    def fit(self, address: np.ndarray, maps: np.ndarray, topics: np.ndarray) -> Static:
        self.mean = maps.mean(axis=0)
        return self

    def read(self, address: np.ndarray, topics: np.ndarray) -> np.ndarray:
        return np.repeat(self.mean[None], len(address), axis=0)


@dataclass
class Topic:
    """The mean map of the fitted questions of the same corpus; a corpus it has not seen gets the mean of all."""

    name: str = "topic"

    def fit(self, address: np.ndarray, maps: np.ndarray, topics: np.ndarray) -> Topic:
        self.mean = maps.mean(axis=0)
        self.by_topic = {t: maps[topics == t].mean(axis=0) for t in np.unique(topics)}
        return self

    def read(self, address: np.ndarray, topics: np.ndarray) -> np.ndarray:
        return np.stack([self.by_topic.get(t, self.mean) for t in topics])


@dataclass
class Nearest:
    """The mean map of the `k` fitted questions whose address is closest by cosine."""

    k: int
    name: str = "nearest"

    def fit(self, address: np.ndarray, maps: np.ndarray, topics: np.ndarray) -> Nearest:
        self.address, self.maps = excess(address), maps
        self.mean = address.mean(axis=0)
        return self

    def read(self, address: np.ndarray, topics: np.ndarray) -> np.ndarray:
        near = cosine(address - self.mean, self.address)
        pick = np.argsort(-near, axis=1)[:, :self.k]
        return self.maps[pick].mean(axis=1)


@dataclass
class Ridge:
    """A linear map from the address's excess onto the map's, with a ridge relative to the addresses' own variance."""

    relative_ridge: float
    name: str = "ridge"

    def fit(self, address: np.ndarray, maps: np.ndarray, topics: np.ndarray) -> Ridge:
        self.mean_address, self.mean_map = address.mean(axis=0), maps.mean(axis=0)
        x = address - self.mean_address
        alpha = self.relative_ridge * float((x ** 2).mean()) * len(x)
        self.matrix = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (maps - self.mean_map))
        return self

    def read(self, address: np.ndarray, topics: np.ndarray) -> np.ndarray:
        return self.mean_map + (address - self.mean_address) @ self.matrix


@dataclass
class Product:
    """The sensitivity of section 3 as the address can have it: the forward half is the address itself (the signal
    through the block), the backward half - how much the loss listens to that output - is predicted from the same
    address by a ridge fitted where a gradient oracle has run. Their product is the lift, scaled by the question's own
    top value.

    Measured 20.09: the address alone carries none of the backward half (rank 0.04 on the excess), the projection
    carries it at rank 0.39.
    """

    backward: object  # a fitted foqlens.projection.Projection over the same groups
    early: np.ndarray  # [groups] bool: the groups the projection reads
    name: str = "product"

    def fit(self, address: np.ndarray, maps: np.ndarray, topics: np.ndarray) -> Product:
        self.mean = address.mean(axis=0)
        return self

    def read(self, address: np.ndarray, topics: np.ndarray) -> np.ndarray:
        forward = np.maximum(address - self.mean, 0.0)
        backward = np.maximum(np.asarray(self.backward.apply(address[:, self.early])), 0.0)
        both = forward * backward
        return both / np.maximum(both.max(axis=1, keepdims=True), 1e-12)


@dataclass
class Direct:
    """No geometry at all: the address's excess is the lift itself, scaled by the question's own top value. The
    simplest rule from the address to a map - what a zone has to beat to be worth its parts."""

    name: str = "direct"

    def fit(self, address: np.ndarray, maps: np.ndarray, topics: np.ndarray) -> Direct:
        self.mean = address.mean(axis=0)
        return self

    def read(self, address: np.ndarray, topics: np.ndarray) -> np.ndarray:
        over = np.maximum(address - self.mean, 0.0)
        return over / np.maximum(over.max(axis=1, keepdims=True), 1e-12)


def neighbours_by_depth(layers: np.ndarray) -> np.ndarray:
    """Every group's neighbours - the groups of its layer and of the layers next to it: [groups, k] of indices, -1
    where a group has fewer."""
    lists = [np.flatnonzero((np.abs(layers - la) <= 1) & (np.arange(len(layers)) != g))
             for g, la in enumerate(layers)]
    width = max(len(x) for x in lists)
    out = np.full((len(layers), width), -1)
    for g, x in enumerate(lists):
        out[g, :len(x)] = x
    return out


def neighbours_of_blocks(block_layer: np.ndarray, block_kind: np.ndarray) -> np.ndarray:
    """Every block's neighbours: the blocks next to it in its own module (its neighbouring output rows - the
    neighbouring neurons or slices of a head) and the block at its place in the same module of the layers next to it.
    This is what gives a zone a width inside a layer, which groups do not have: [blocks, 4] of indices, -1 where a
    block has fewer."""
    order = np.arange(len(block_layer))
    place = np.zeros(len(block_layer), dtype=int)
    at = {}
    for b, (la, kind) in enumerate(zip(block_layer.tolist(), block_kind.tolist())):
        place[b] = at.get((la, kind), 0)
        at[(la, kind)] = place[b] + 1
    where = {(la, kind, p): b for b, (la, kind, p) in enumerate(zip(block_layer.tolist(), block_kind.tolist(),
                                                                   place.tolist()))}
    out = np.full((len(block_layer), 4), -1)
    for b in order:
        la, kind, p = int(block_layer[b]), str(block_kind[b]), int(place[b])
        for j, key in enumerate(((la, kind, p - 1), (la, kind, p + 1), (la - 1, kind, p), (la + 1, kind, p))):
            out[b, j] = where.get(key, -1)
    return out


@dataclass
class Zones:
    """The mechanism itself (docs/quantization-filter.md, docs/zone-strategies.md): a zone is the hill around a peak of
    the question's address - the connected region where the address stays above a share of that peak - and never a
    ball of some radius. A ball along the depth takes every group near the centre, engaged or not, and at any useful
    size covers the network (Volodya 20.09: "шар накроет всю сеть, пользы 0"). Inside the hill the lift is the address
    itself over the peak, outside every hill the group reads the base.

    Nothing is fitted. The knobs are the quantile a peak must pass and `slope`, the share of the peak the hill breaks
    off at; both carry to a network of any depth.
    """

    layers: np.ndarray
    quantile: float  # a peak must be above this quantile of the question's address
    # who is next to whom: [units, k] of indices, -1 where a unit has fewer. Over the groups it is the layers next to
    # it (neighbours_by_depth), over the blocks their own module and the same place in the layers next to it
    # (neighbours_of_blocks), and along the signal's path it is that graph's table (foqlens.coupling) - the mechanism
    # does not change, only what counts as next to what
    near: np.ndarray | None = None
    slope: float = 0.5  # the hill holds while the address stays above this share of its peak
    # the figure of one lens is finite: however far the signal holds, a zone spans no more layers than this
    # (Volodya 20.09: "её ширина максимальная пусть будет иметь диаметр"), as a share of the depth
    diameter: float = 0.2
    own_height: bool = True  # a zone rises to its own peak; else every zone rises to the top
    name: str = "zones"

    def fit(self, address: np.ndarray, maps: np.ndarray, topics: np.ndarray) -> Zones:
        self.mean = address.mean(axis=0)
        if self.near is None:
            self.near = neighbours_by_depth(self.layers)
        return self

    def around(self, unit: int) -> np.ndarray:
        """The units next to `unit`, by whatever table this mechanism was given."""
        row = self.near[unit]
        return row[row >= 0]

    def peaks(self, row: np.ndarray) -> np.ndarray:
        """The units of one question's address that none of their neighbours exceeds and that pass the quantile:
        [units] bool."""
        over = row >= np.quantile(row, self.quantile)
        out = np.zeros(len(row), dtype=bool)
        for g in np.flatnonzero(over):
            out[g] = row[g] >= row[self.around(g)].max(initial=-np.inf)
        return out

    def hill(self, row: np.ndarray, centre: int) -> np.ndarray:
        """The zone of a peak: the groups connected to it along the depth whose address holds above `slope` of the
        peak - any shape, and as small as the question's own signal is: [groups] bool."""
        span = max(self.diameter * (self.layers.max() - self.layers.min() + 1), 1.0) / 2
        over = (row >= row[centre] * self.slope) & (np.abs(self.layers - self.layers[centre]) <= span)
        found = np.zeros(len(row), dtype=bool)
        found[centre], front = True, [centre]
        while front:
            g = front.pop()
            near = self.around(g)
            near = near[over[near] & ~found[near]]
            found[near] = True
            front += near.tolist()
        return found

    def read(self, address: np.ndarray, topics: np.ndarray) -> np.ndarray:
        excess_of = address - self.mean
        out = np.zeros_like(excess_of)
        for q, row in enumerate(excess_of):
            found = np.flatnonzero(self.peaks(row))
            if not len(found):
                continue
            for centre in found:
                inside = self.hill(row, centre)
                top = max(row[centre], 1e-12)
                # inside the hill the lift is the address itself over the peak; a zone of its own height is scaled
                # by how high its peak stands among the question's peaks
                lift = np.clip(row / top, 0.0, 1.0) * inside
                if self.own_height:
                    lift = lift * min(row[centre] / max(row[found].max(), 1e-12), 1.0)
                out[q] = np.maximum(out[q], lift)
        return out


BRIDGES = {"static": Static, "topic": Topic, "nearest": Nearest, "ridge": Ridge, "zones": Zones, "direct": Direct,
           "product": Product}


LADDER = list(RUNGS) + [TOP]
SHIFT_STEPS = 40  # bisections of the shift: 2^-40 of its range, far below one group's value
# The lift a level stands for, so that a map read from the oracles can be fitted as a lift and read back as levels:
# the bench's measured scale (precision_field.FIELD_VALUE), the share of the base rung's error the rung removes.
# Evenly spaced until 20.09, which put D6 at 0.67 where the measurement puts it at 0.945, so a map fitted on the even
# scale spent its middle rungs as if they were half of what they are worth.
ABOVE = [lv for lv in LADDER if lv > RUNGS[0]]  # the rungs above the base, coarsest first
LIFT_VALUE = {lv: FIELD_VALUE[lv] for lv in LADDER}
# The outer edge of every rung's ring, the top rung first: a rung reaches as far out as its own worth leaves room for,
# and past the last of them - a lift under what the coarsest rung above the base is worth - lies the base itself
# (Volodya 20.09: everything under 0.8 reads the base).
EVEN_STOPS = tuple(1.0 - FIELD_VALUE[lv] for lv in ABOVE[::-1])


def lift_of(levels: np.ndarray) -> np.ndarray:
    """Every level code as the lift it stands for (LIFT_VALUE): the same shape, 0 at the base, 1 at the top."""
    table = np.zeros(int(TOP) + 1)
    for lv, value in LIFT_VALUE.items():
        table[int(lv)] = value
    return table[levels]


def to_levels(lift: np.ndarray, shift: float = 0.0, stops: tuple[float, ...] = EVEN_STOPS) -> np.ndarray:
    """The mapper of the filter (docs/quantization-filter.md, rule 6) over a field of lifts: with the stops of the
    profile (the outer edge of every rung's ring, as a share of the radius, the top rung first), a group reads the
    finest rung whose stop is at or past its place in the zone, rho = 1 - lift, and the base where no stop reaches it -
    a lift under what the coarsest rung above the base is worth: [questions, groups] lifts -> codes."""
    lift = np.asarray(lift, dtype=float) + shift
    rho = 1.0 - lift
    inside = np.array(stops) >= rho[..., None]
    # the finest rung first, as the stops are; a profile of fewer stops keeps the finest rungs and skips the rest
    codes = np.array([int(lv) for lv in ABOVE[::-1][:len(stops)]], dtype=np.uint8)
    return np.where(inside.any(axis=-1), codes[inside.argmax(axis=-1)], np.uint8(int(RUNGS[0]))).astype(np.uint8)


def bits_of(levels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The nominal bits a weight of every question's layout: [questions]."""
    table = np.zeros(int(TOP) + 1)
    for lv in LADDER:
        table[int(lv)] = lv.bits
    return (table[levels] * weights).sum(axis=1) / weights.sum()


def shift_for_bits(field: np.ndarray, weights: np.ndarray, bits: float) -> float:
    """The one shift of the whole field at which its maps spend `bits` a weight on average, by bisection: the price of
    memory read on the questions a bridge was fitted on, the same for every question after it."""
    lo, hi = -1.0, 1.0  # the field lives in 0 ... 1: a shift past either end is every group at one rung
    for _ in range(SHIFT_STEPS):
        mid = (lo + hi) / 2
        spent = float(bits_of(to_levels(field, mid), weights).mean())
        lo, hi = (lo, mid) if spent > bits else (mid, hi)  # a larger shift reads finer and spends more
    return (lo + hi) / 2
