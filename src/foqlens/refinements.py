"""The bench's copy of a weight: a k-quant base with refinements over it, one copy read at its base and deeper.

The base is kquant.KBase - Q2_K or Q4_K after llama.cpp. Over it lie refinements as in quant.RefinedWeight:
refinement k quantizes what the base and the refinements before it left, 2-bit symmetric codes with step
block_step / 4**k, where block_step is the base's own quantized step d * scale of that block - a refinement stores
codes only, no scale.

A depth counts 2-bit steps: a base of Q2_K takes depth 1, a base of Q4_K depth 2, and every refinement adds one.
A copy cannot be read shallower than its base.

Invariant: a weight whose base error is within half its block step stays within block_step / 2 / 4**k after
k refinements.
Invariant: a deeper refinement never changes what the shallower depths read.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from foqlens.kquant import Q2_K, Q4_K, KBase, KFormat, _blocks
from foqlens.quant import DEPTH_BITS, MAX_DEPTH, _DepthReader, _pack, _unpack

_REFINEMENT_LEVELS = 2**DEPTH_BITS
_REFINEMENT_CENTER = _REFINEMENT_LEVELS // 2  # codes 0..3 read as -1.5 .. 1.5 steps, as in quant.RefinedWeight


@dataclass
class KRefinedWeight(_DepthReader):
    """A k-quant base and refinements over it, up to MAX_DEPTH in all.

    A _DepthReader, so the precision controller reads it by blocks at several depths the way it reads quant.RefinedWeight.
    """

    base: KBase
    refinements: torch.Tensor  # uint8 [refinements, out, in // 4] packed 2-bit codes
    shape: tuple[int, int]

    @classmethod
    def quantize(cls, weight: torch.Tensor, fmt: KFormat, depth: int = MAX_DEPTH) -> KRefinedWeight:
        assert fmt.base_depth <= depth <= MAX_DEPTH, (fmt, depth)
        base = KBase.quantize(weight, fmt)
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
        return cls(base=base, refinements=refinements, shape=tuple(weight.shape))

    @property
    def depth(self) -> int:
        """The depth stored: the base's own and one per refinement."""
        return self.base.fmt.base_depth + self.refinements.shape[0]

    def read_depth(self, depth: int) -> int:
        """The depth a read asking for `depth` gets: never shallower than the base, never deeper than stored."""
        return min(max(depth, self.base.fmt.base_depth), self.depth)

    def bits_per_weight(self, depth: int) -> float:
        return self.base.fmt.bits_per_weight + DEPTH_BITS * (self.read_depth(depth) - self.base.fmt.base_depth)

    @property
    def nbytes(self) -> int:
        """Bytes of the stored codes, block scales and super-block pairs."""
        parts = (self.base.codes, self.base.scales, self.base.mins, self.base.d, self.base.dmin, self.refinements)
        return sum(t.numel() * t.element_size() for t in parts)

    def _sums(self, depths: list[int]) -> Iterator[torch.Tensor]:
        """The float32 blocks read to each of the ascending `depths`: the base, then one refinement at a time."""
        w = self.base.dequantize()
        step, done = self.base.steps() / _REFINEMENT_LEVELS, 0
        for depth in depths:
            for e in range(done, self.read_depth(depth) - self.base.fmt.base_depth):
                w += step * (_unpack(self.refinements[e]).view(w.shape).float() - _REFINEMENT_CENTER + 0.5)
                step = step / _REFINEMENT_LEVELS
                done = e + 1
            yield w

    def _weight(self, w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return w.reshape(self.shape).to(dtype)


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

    def quantize(self, name: str, weight: torch.Tensor) -> KRefinedWeight:
        """The module's whole copy, every depth: a precision.RefinedCopy, so the controller reads it by blocks."""
        return KRefinedWeight.quantize(weight, self.format_for(name))

    def read(self, name: str, weight: torch.Tensor, level) -> torch.Tensor:
        fmt = self.format_for(name)
        depth = max(level.depth, fmt.base_depth)
        return KRefinedWeight.quantize(weight, fmt, depth=depth).dequantize(weight.dtype, depth)
