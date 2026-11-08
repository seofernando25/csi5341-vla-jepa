"""The graph engine's building blocks must reproduce the stock computation exactly."""

import types

import pytest
import torch

from evaluation.fastpath import _sinusoid

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="NF4 kernels need CUDA")


@cuda
@pytest.mark.parametrize("shape", [(2048, 2048), (1024, 2048), (6144, 2048), (2048, 6144)])
def test_nf4_dequantization_equals_bitsandbytes_bit_for_bit(shape):
    from bitsandbytes import functional

    from evaluation import nf4

    weight = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    packed, state = functional.quantize_4bit(
        weight, blocksize=64, compress_statistics=True, quant_type="nf4"
    )
    expected = functional.dequantize_4bit(packed, state)
    out = nf4.dequantize(packed, nf4.tables(types.SimpleNamespace(quant_state=state)), shape)
    assert torch.equal(out, expected)


def test_capturable_sinusoid_matches_the_stock_encoding():
    from lerobot.policies.vla_jepa.action_head import SinusoidalPositionalEncoding

    module = SinusoidalPositionalEncoding(64)
    timesteps = torch.tensor([[0], [250], [999]])
    expected = module(timesteps)
    module.log_base = torch.log(torch.tensor(10000.0))
    assert torch.equal(_sinusoid(module, timesteps), expected)
