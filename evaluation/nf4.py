"""Bit-exact NF4 weight dequantization at memory bandwidth.

bitsandbytes dequantizes NF4 with a nested (double-quantized) absmax, which costs extra kernels per
weight. Here one Triton kernel decodes the nested absmax and unpacks two 4-bit codes per byte, so
the result equals `bitsandbytes.functional.dequantize_4bit` exactly and no fp32 absmax copy is kept.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _dequantize(
    packed, absmax_codes, code2, absmax2, offset, lut, out, n_bytes, BLOCK: tl.constexpr
):
    base = tl.program_id(0) * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n_bytes
    byte = tl.load(packed + offs, mask=mask, other=0).to(tl.int32)
    block = offs // 32  # 64 elements = 32 bytes per quantization block
    scale = tl.load(code2 + tl.load(absmax_codes + block, mask=mask, other=0).to(tl.int32))
    # Separately rounded multiply and add: a fused multiply-add differs from bitsandbytes in the last bit.
    scale = libdevice.add_rn(
        libdevice.mul_rn(scale, tl.load(absmax2 + block // 256, mask=mask, other=0.0)),
        tl.load(offset),
    )
    high = tl.load(lut + (byte >> 4)) * scale
    low = tl.load(lut + (byte & 15)) * scale
    values = tl.interleave(high, low).to(tl.bfloat16)
    position = tl.arange(0, 2 * BLOCK)
    tl.store(out + 2 * base + position, values, mask=(base + position // 2) < n_bytes)


def tables(weight):
    """Decoding tables of a double-quantized NF4 `Params4bit` weight, as plain tensors."""
    state = weight.quant_state
    if state.blocksize != 64 or state.quant_type != "nf4" or state.nested is not True:
        raise ValueError("Only nested NF4 with block size 64 is supported")
    inner = state.state2
    return {
        "absmax_codes": state.absmax,
        "code2": inner.code.float().contiguous(),
        "absmax2": inner.absmax,
        "offset": state.offset.float().reshape(1).contiguous(),
        "lut": state.code.float().contiguous(),
    }


@torch.library.custom_op("csi5341::nf4_dequantize", mutates_args=())
def _dequantize_op(
    packed: torch.Tensor,
    absmax_codes: torch.Tensor,
    code2: torch.Tensor,
    absmax2: torch.Tensor,
    offset: torch.Tensor,
    lut: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    out = torch.empty(rows, cols, dtype=torch.bfloat16, device=packed.device)
    n_bytes = packed.numel()
    block = 2048
    _dequantize[(triton.cdiv(n_bytes, block),)](
        packed.view(-1), absmax_codes, code2, absmax2, offset, lut, out, n_bytes, BLOCK=block
    )
    return out


@_dequantize_op.register_fake
def _(packed, absmax_codes, code2, absmax2, offset, lut, rows, cols):
    return torch.empty(rows, cols, dtype=torch.bfloat16, device=packed.device)


def dequantize(packed, t, shape):
    """New bf16 (N, K) tensor from packed NF4 bytes, matching bitsandbytes bit for bit.

    A custom op that returns a fresh tensor keeps `torch.compile` from copying aliased scratch
    buffers, and lets the allocator reuse each weight's memory once its GEMM has consumed it.
    """
    return _dequantize_op(
        packed,
        t["absmax_codes"],
        t["code2"],
        t["absmax2"],
        t["offset"],
        t["lut"],
        shape[0],
        shape[1],
    )
