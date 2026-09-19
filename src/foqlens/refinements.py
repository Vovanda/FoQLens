"""The bench's copy of a weight: a k-quant base with refinements over it, one copy read at its base and deeper.

The base is kquant.KBase - Q2_K or Q4_K after llama.cpp, or a published GGUF file's k-quant blocks read as they lie
(PublishedBaseLadder: Q2_K, Q3_K, Q4_K, Q6_K). Over it lie refinements as in quant.RefinedWeight:
refinement k quantizes what the base and the refinements before it left, 2-bit symmetric codes with step
block_step / 4**k, where block_step is the base's own quantized step d * scale of that block - a refinement stores
codes only, no scale.

A depth counts 2-bit steps: a base takes depth bits // 2 - Q2_K and Q3_K 1, Q4_K 2, Q6_K 3 - and every refinement
adds one.
A copy cannot be read shallower than its base.

Over the deepest refinement lies the exact tail (ExactTail): how many units in the last place of the source type
(bf16, fp16, fp32) every weight still is from the prediction rounded to that type. The base, the refinements and the
tail read back the source weight bit for bit.

Invariant: a weight whose base error is within half its block step stays within block_step / 2 / 4**k after
k refinements.
Invariant: a deeper refinement never changes what the shallower depths read.
Invariant: ulp_order is a bijection that keeps the order of the floats of a type, -0.0 below +0.0.
Invariant: the exact tail restores the source weight bit for bit from the prediction it was encoded against, for every
source type in ORDER_BITS.
Invariant: a copy taken apart into tensors and put back (tensors, KRefinedWeight.from_tensors) reads exactly as before
at every depth.
Invariant: stack_fit counts a weight past the stored depth's bound only if its base error is past half a step - the
refinements bring every other weight within step / 2 / 4**k.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import torch

from foqlens.kquant import Q2_K, Q4_K, KBase, KFormat, _blocks, from_gguf_blocks, gguf_blocks
from foqlens.quant import DEPTH_BITS, MAX_DEPTH, _DepthReader, _pack, _unpack

_REFINEMENT_LEVELS = 2**DEPTH_BITS
_REFINEMENT_CENTER = _REFINEMENT_LEVELS // 2  # codes 0..3 read as -1.5 .. 1.5 steps, as in quant.RefinedWeight


@dataclass
class KRefinedWeight(_DepthReader):
    """A k-quant base and refinements over it, up to MAX_DEPTH in all.

    A _DepthReader, so the precision controller reads it by blocks at several depths the way it reads quant.RefinedWeight.
    The base is held as ggml blocks, as the file holds it - 2.625 or 4.5 bits per weight -
    and unpacked when it is read, as the refinements are.
    """

    fmt: KFormat
    blocks: torch.Tensor  # uint8 [out, super-blocks, bytes]: the base as ggml blocks (kquant.gguf_blocks)
    refinements: torch.Tensor  # uint8 [refinements, out, in // 4] packed 2-bit codes
    shape: tuple[int, int]

    @property
    def base(self) -> KBase:
        """The base unpacked from its blocks."""
        return from_gguf_blocks(self.blocks, self.fmt)

    @classmethod
    def quantize(cls, weight: torch.Tensor, fmt: KFormat, depth: int = MAX_DEPTH) -> KRefinedWeight:
        return cls.over(gguf_blocks(KBase.quantize(weight, fmt)), fmt, weight, depth)

    @classmethod
    def over(cls, blocks: torch.Tensor, fmt: KFormat, weight: torch.Tensor, depth: int = MAX_DEPTH) -> KRefinedWeight:
        """Refinements of `weight` over a base given as ggml blocks - ours, or a published GGUF file's as it lies."""
        assert fmt.base_depth <= depth <= MAX_DEPTH, (fmt, depth)
        base = from_gguf_blocks(blocks, fmt)
        rest = _blocks(weight, fmt) - base.dequantize()
        step = base.steps() / _REFINEMENT_LEVELS
        live = step != 0
        safe_step = torch.where(live, step, torch.ones_like(step))
        codes = []
        for _ in range(depth - fmt.base_depth):
            code = torch.where(live, torch.floor(rest / safe_step + _REFINEMENT_CENTER).clamp_(0, _REFINEMENT_LEVELS - 1),
                               torch.full_like(rest, _REFINEMENT_CENTER))
            rest = rest - step * (code - _REFINEMENT_CENTER + 0.5)
            codes.append(_pack(code.to(torch.uint8).view(weight.shape)))
            step = step / _REFINEMENT_LEVELS
            safe_step = safe_step / _REFINEMENT_LEVELS
        refinements = torch.stack(codes) if codes else torch.empty((0, *weight.shape[:-1], weight.shape[1] // 4), dtype=torch.uint8, device=weight.device)
        return cls(fmt=fmt, blocks=blocks, refinements=refinements, shape=tuple(weight.shape))

    @property
    def depth(self) -> int:
        """The depth stored: the base's own and one per refinement."""
        return self.fmt.base_depth + self.refinements.shape[0]

    def read_depth(self, depth: int) -> int:
        """The depth a read asking for `depth` gets: never shallower than the base, never deeper than stored."""
        return min(max(depth, self.fmt.base_depth), self.depth)

    def bits_per_weight(self, depth: int) -> float:
        return self.fmt.bits_per_weight + DEPTH_BITS * (self.read_depth(depth) - self.fmt.base_depth)

    @property
    def nbytes(self) -> int:
        """Bytes held: the base's blocks and the refinements."""
        return sum(t.numel() * t.element_size() for t in (self.blocks, self.refinements))

    def _sums(self, depths: list[int]) -> Iterator[torch.Tensor]:
        """The float32 blocks read to each of the ascending `depths`: the base, then one refinement at a time."""
        base = self.base
        w = base.dequantize()
        step, done = base.steps() / _REFINEMENT_LEVELS, 0
        for depth in depths:
            for e in range(done, self.read_depth(depth) - self.fmt.base_depth):
                w += step * (_unpack(self.refinements[e]).view(w.shape).float() - _REFINEMENT_CENTER + 0.5)
                step = step / _REFINEMENT_LEVELS
                done = e + 1
            yield w

    def _weight(self, w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return w.reshape(self.shape).to(dtype)

    def prediction(self) -> torch.Tensor:
        """float32 [out, in]: the weight read to the stored depth - what the exact tail is counted from."""
        return self.dequantize(torch.float32, self.depth)

    def tensors(self) -> dict[str, torch.Tensor]:
        """The copy as named tensors: the base as ggml blocks, then every refinement."""
        return {"base": self.blocks} | {f"refinement.{k}": r for k, r in enumerate(self.refinements)}

    @classmethod
    def from_tensors(cls, fmt: KFormat, tensors: Mapping[str, torch.Tensor], shape: tuple[int, int]) -> KRefinedWeight:
        """The copy tensors() took apart; the refinements are read in order while they last."""
        blocks = tensors["base"]
        stack = []
        while f"refinement.{len(stack)}" in tensors:
            stack.append(tensors[f"refinement.{len(stack)}"])
        refinements = torch.stack(stack) if stack else torch.empty((0, shape[0], shape[1] // 4), dtype=torch.uint8, device=blocks.device)
        return cls(fmt=fmt, blocks=blocks, refinements=refinements, shape=tuple(shape))


# ==== Exact tail ====
# The source's own grid: a weight is an integer in the order of its type's floats, and the tail stores how far the
# source is from the prediction in that order - units in the last place of the source type.

# Weights of a row sharing one code width. Measured on E2B-it (scripts/exact_tail_cost.py): the base, refinements to D8
# and the tail cost 20.1 bits per weight at groups of 256, 15.9 at 32, 15.05 at 16 - below the 16 of bf16.
EXACT_GROUP = 16
ORDER_BITS = {torch.bfloat16: 16, torch.float16: 16, torch.float32: 32}
_ORDER_VIEW = {16: torch.int16, 32: torch.int32}
_MAX_WIDTH = 64  # a zigzagged int64 never needs more
# Weights packed or unpacked at once: their int64 bits at the widest fp32 distance (33 bits) stay under 0.6 GiB.
_CHUNK_GROUPS = 2**21 // EXACT_GROUP


def ulp_order(t: torch.Tensor) -> torch.Tensor:
    """int64: the float's rank among the floats of its type; -0.0 is -1 and +0.0 is 0, so no two floats share a rank."""
    bits = t.view(_ORDER_VIEW[ORDER_BITS[t.dtype]]).long()
    magnitude = bits & ((1 << (ORDER_BITS[t.dtype] - 1)) - 1)
    return torch.where(bits >= 0, magnitude, -1 - magnitude)


def from_ulp_order(order: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The floats of `dtype` ulp_order ranked: its inverse."""
    sign = 1 << (ORDER_BITS[dtype] - 1)
    bits = torch.where(order >= 0, order, (-1 - order) - sign)
    return bits.to(_ORDER_VIEW[ORDER_BITS[dtype]]).view(dtype)


def _zigzag(d: torch.Tensor) -> torch.Tensor:
    return (d << 1) ^ (d >> 63)


def _unzigzag(z: torch.Tensor) -> torch.Tensor:
    return (z >> 1) ^ -(z & 1)


def _pack_width(values: torch.Tensor, width: int) -> torch.Tensor:
    """uint8 [n, EXACT_GROUP * width / 8]: every value's `width` low bits, the first value's lowest bit first."""
    bits = (values[..., None] >> torch.arange(width, device=values.device)) & 1
    bytes_ = bits.reshape(values.shape[0], -1, 8) << torch.arange(8, device=values.device)
    return bytes_.sum(-1).to(torch.uint8)


def _unpack_width(packed: torch.Tensor, width: int) -> torch.Tensor:
    """int64 [n, EXACT_GROUP]: what _pack_width packed."""
    bits = (packed.long()[..., None] >> torch.arange(8, device=packed.device)) & 1
    return (bits.reshape(packed.shape[0], EXACT_GROUP, width) << torch.arange(width, device=packed.device)).sum(-1)


@dataclass
class ExactTail:
    """How far the source is from the prediction, in units in the last place of the source type.

    Every group of EXACT_GROUP weights of a row is written at its own width - the bits its largest zigzagged
    distance needs - and the groups follow row by row, so the bytes of a block of rows are one run of `data`.
    """

    widths: torch.Tensor  # uint8 [out, in // EXACT_GROUP]
    data: torch.Tensor  # uint8 [sum(widths) * EXACT_GROUP / 8]
    dtype: torch.dtype  # the source's

    @classmethod
    def encode(cls, source: torch.Tensor, prediction: torch.Tensor) -> ExactTail:
        rounded = prediction.to(source.dtype)
        z = _zigzag(ulp_order(source) - ulp_order(rounded)).view(-1, EXACT_GROUP)
        top = z.amax(-1)
        widths = (top[:, None] >= (1 << torch.arange(_MAX_WIDTH - 1, device=z.device))).sum(-1)
        offsets = _group_offsets(widths)
        data = torch.empty(int(offsets[-1]), dtype=torch.uint8, device=z.device)
        for width, groups in _groups_by_width(widths):
            data[_byte_index(offsets, groups, width)] = _pack_width(z[groups], width)
        return cls(widths=widths.to(torch.uint8).view(source.shape[0], -1), data=data, dtype=source.dtype)

    def decode(self, prediction: torch.Tensor) -> torch.Tensor:
        """The source weight: the prediction rounded to the source type, moved by the stored distances."""
        rounded = prediction.to(self.dtype)
        widths = self.widths.reshape(-1).long()
        offsets = _group_offsets(widths)
        z = torch.zeros((widths.numel(), EXACT_GROUP), dtype=torch.long, device=self.data.device)
        for width, groups in _groups_by_width(widths):
            z[groups] = _unpack_width(self.data[_byte_index(offsets, groups, width)], width)
        return from_ulp_order(ulp_order(rounded) + _unzigzag(z).view(rounded.shape), self.dtype)

    @property
    def bits_per_weight(self) -> float:
        return (self.data.numel() * 8 + self.widths.numel() * 8) / (self.widths.numel() * EXACT_GROUP)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {"exact.widths": self.widths, "exact": self.data}

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, torch.Tensor], dtype: torch.dtype) -> ExactTail:
        return cls(widths=tensors["exact.widths"], data=tensors["exact"], dtype=dtype)


def _groups_by_width(widths: torch.Tensor) -> Iterator[tuple[int, torch.Tensor]]:
    """(width, group indices) for every nonzero width, the groups in chunks of _CHUNK_GROUPS.

    Once per module at write and load, never in a forward pass: the host reads the widths present.
    """
    for width in widths.unique().tolist():
        if width:
            groups = torch.nonzero(widths == width).squeeze(1)
            yield from ((width, chunk) for chunk in groups.split(_CHUNK_GROUPS))


def _group_offsets(widths: torch.Tensor) -> torch.Tensor:
    """[groups + 1] int64: where every group's bytes start in `data`, and the end."""
    sizes = widths.long() * (EXACT_GROUP // 8)
    return torch.cat([sizes.new_zeros(1), sizes.cumsum(0)])


def _byte_index(offsets: torch.Tensor, groups: torch.Tensor, width: int) -> torch.Tensor:
    """[len(groups), EXACT_GROUP * width / 8]: the positions in `data` of the bytes of `groups`, all of one width."""
    return offsets[groups][:, None] + torch.arange(EXACT_GROUP * width // 8, device=offsets.device)


# The classes whose weights the floor keeps on a Q4_K base: raised one at a time over a 2-bit floor none brings the
# knowledge back, together they do (E002, exploration-module-classes: EM 0.048 -> 0.413, 2% of the weights in k and
# the per-layer modules); unsloth's UD-Q2_K_XL raises the same classes.
SENSITIVE_CLASSES = ("self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.down_proj",
                     "per_layer_input_gate", "per_layer_projection")


@dataclass(frozen=True)
class KQuantLadder:
    """The read depths of one k-quant copy per module: the sensitive classes on a Q4_K base, the rest on Q2_K.

    A WeightSource for Controller.bake_from. A level reads its depth, never shallower than the base: D2 of a
    sensitive module reads its Q4_K base.

    Invariant: read(name, weight, level) equals KRefinedWeight.quantize(weight, format_for(name)) read to the level's
    depth.
    """

    sensitive: tuple[str, ...] = SENSITIVE_CLASSES

    def format_for(self, name: str) -> KFormat:
        return Q4_K if name.endswith(self.sensitive) else Q2_K

    def quantize(self, name: str, weight: torch.Tensor, depth: int = MAX_DEPTH) -> KRefinedWeight:
        """The module's copy to `depth` - every depth by default, never shallower than its base: a precision.RefinedCopy,
        so the controller reads it by blocks."""
        fmt = self.format_for(name)
        return KRefinedWeight.quantize(weight, fmt, max(depth, fmt.base_depth))

    def read(self, name: str, weight: torch.Tensor, level) -> torch.Tensor:
        fmt = self.format_for(name)
        depth = max(level.depth, fmt.base_depth)
        return KRefinedWeight.quantize(weight, fmt, depth=depth).dequantize(weight.dtype, depth)


# A module's base as another quantizer left it: its format and ggml blocks [out, super-blocks, bytes], or None where it
# has none the stack can refine - a tensor kept in float, or an IQ type with no block step.
BaseBlocks = Callable[[str, tuple[int, int]], "tuple[KFormat, torch.Tensor] | None"]


@dataclass(frozen=True)
class PublishedBaseLadder:
    """The read depths of a copy over another quantizer's base: a published file's blocks read as they lie, our
    refinements and exact tail over them. A module the file gives no refinable base keeps the bench's own.

    Invariant: a module with a foreign base reads it at its base depth byte for byte as the file holds it.
    """

    bases: BaseBlocks
    own: KQuantLadder = KQuantLadder()

    def quantize(self, name: str, weight: torch.Tensor, depth: int = MAX_DEPTH) -> KRefinedWeight:
        found = self.bases(name, tuple(weight.shape))
        if found is None:
            return self.own.quantize(name, weight, depth)
        fmt, blocks = found
        return KRefinedWeight.over(blocks.to(weight.device), fmt, weight, max(depth, fmt.base_depth))

    def read(self, name: str, weight: torch.Tensor, level) -> torch.Tensor:
        copy = self.quantize(name, weight)
        return copy.dequantize(weight.dtype, copy.read_depth(level.depth))


# How far past half a step a base error must be to count as out of the refinements' reach: fp32 noise of the
# subtraction, relative to the step.
HALF_STEP_SLACK = 1e-5


@dataclass(frozen=True)
class StackFit:
    """How a copy holds the weight it was cut from, depth by depth.

    A refinement codes the rest within +-half its step, so a weight whose base error is past half the block's step is out
    of the refinements' reach: every depth leaves it where the first one clamped it. A base quantized to the nearest
    level has few such weights; a base fitted to a weighted error (an imatrix) may leave more.
    """

    past_half_step: float  # share of weights whose base error is past half their block's step
    past_bound_at_top: float  # share whose error at the stored depth is past half the step of that depth
    error_by_depth: dict[int, float]  # RMS error against the source over the source's RMS, at every depth held


def stack_fit(copy: KRefinedWeight, source: torch.Tensor) -> StackFit:
    """StackFit of `copy` against the `source` weight [out, in] it was cut from."""
    base = copy.base
    step = base.steps().abs()  # a symmetric base's scale is signed
    past = (_blocks(source, copy.fmt) - base.dequantize()).abs() > step / 2 * (1 + HALF_STEP_SLACK)
    top = copy.depth
    top_step = step / float(_REFINEMENT_LEVELS) ** (top - copy.fmt.base_depth)
    top_error = (copy.dequantize(torch.float32, top).view_as(past) - _blocks(source, copy.fmt)).abs()
    scale = source.float().pow(2).mean().sqrt()
    by_depth = {d: float((copy.dequantize(torch.float32, d) - source.float()).pow(2).mean().sqrt() / scale)
                for d in range(copy.fmt.base_depth, top + 1)}
    return StackFit(float(past.float().mean()),
                    float((top_error > top_step / 2 * (1 + HALF_STEP_SLACK)).float().mean()), by_depth)
