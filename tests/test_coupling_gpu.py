"""The signal-path graph from weights on the card (foqlens.coupling): the same graph as from the same weights on the CPU."""

import pytest
import torch

from foqlens.coupling import signal_path_table

pytestmark = pytest.mark.gpu

N_HEADS, HEAD_DIM, HIDDEN = 4, 32, 128


def toy(device: str) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    shapes = {"self_attn.q_proj": (N_HEADS * HEAD_DIM, HIDDEN), "self_attn.v_proj": (2 * HEAD_DIM, HIDDEN),
              "self_attn.o_proj": (HIDDEN, N_HEADS * HEAD_DIM), "mlp.gate_proj": (192, HIDDEN),
              "mlp.down_proj": (HIDDEN, 192)}
    return {f"layers.{i}.{k}": torch.randn(*s, generator=g).to(device, torch.bfloat16)
            for i in range(2) for k, s in shapes.items()}


def test_the_graph_from_weights_on_the_card_is_the_graph_from_the_cpu():
    on_card, on_cpu = signal_path_table(toy("cuda"), N_HEADS, 3), signal_path_table(toy("cpu"), N_HEADS, 3)
    assert torch.equal(on_card[0], on_cpu[0])
    torch.testing.assert_close(on_card[1], on_cpu[1], rtol=1e-4, atol=0)
