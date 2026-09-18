"""What the exact tail of a copy costs under different codings - measured before one is built into the format.

The tail is the distance of every source weight from its prediction in units in the last place of the source type
(refinements.ExactTail). Its cost here, in bits per weight:

- fixed width: every group of `group` weights of a row at the width of its largest zigzagged distance, one byte of
  width per group - the coding of ExactTail at group = EXACT_GROUP;
- entropy: the order-0 entropy of the distances - what an entropy coder over the whole module would reach;
- entropy given the exponent: the entropy of the distances given the exponent of the prediction, which the reader
  knows before the tail - what a coder with that context would reach;
- source entropy: the order-0 entropy of the source weights themselves, with no base under them - the size
  lossless compression of the source alone reaches (DFloat11 codes bf16 at about 11 bits).

Invariant: an entropy of n equal symbols is 0 and of 2**k equally frequent symbols is k bits.
Invariant: the fixed-width cost of a group is its width plus the width's own byte, spread over the group.
"""

from __future__ import annotations

import torch

WIDTH_BYTE_BITS = 8
_MAX_WIDTH = 64


def fixed_width_bits(z: torch.Tensor, group: int) -> float:
    """Bits per weight of zigzagged distances `z` [out, in] written group by group at each group's own width."""
    top = z.reshape(-1, group).amax(-1)
    widths = (top[:, None] >= (1 << torch.arange(_MAX_WIDTH - 1, device=z.device))).sum(-1)
    return float((widths.double().mean() * group + WIDTH_BYTE_BITS) / group)


def entropy_bits(symbols: torch.Tensor) -> float:
    """Order-0 entropy of the symbols, bits per symbol."""
    _, counts = torch.unique(symbols.reshape(-1), return_counts=True)
    p = counts.double() / counts.sum()
    return float(-(p * p.log2()).sum())


def conditional_entropy_bits(symbols: torch.Tensor, context: torch.Tensor) -> float:
    """H(symbols | context), bits per symbol: the entropy within each context, weighted by how often it occurs."""
    symbols, context = symbols.reshape(-1), context.reshape(-1)
    total = 0.0
    for value in torch.unique(context):
        inside = symbols[context == value]
        total += inside.numel() * entropy_bits(inside)
    return total / symbols.numel()


def exponent(t: torch.Tensor) -> torch.Tensor:
    """The biased exponent field of each float - the context of its distance."""
    bits = torch.finfo(t.dtype).bits
    mantissa = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}[t.dtype]
    raw = t.view({16: torch.int16, 32: torch.int32}[bits]).long()
    return (raw >> mantissa) & ((1 << (bits - 1 - mantissa)) - 1)
