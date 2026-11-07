"""Pinned HQQ proximal PTQ with real 2/3-bit packing and optional GemLite."""

import sys

import torch
from torch import nn
from torch.nn import functional as F

from evaluation.common import ROOT, file_hash, read_json
from evaluation.compression.transforms import replace, selected_linears


def package(method):
    folder = ROOT / "outputs/compression/vendor" / method
    manifest = read_json(ROOT / "studies/compression/methods.json")["upstream"][method]
    for name, digest in manifest["files"].items():
        if file_hash(folder / name) != digest:
            raise ValueError(f"Pinned low-bit source changed: {method}/{name}")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return manifest


class HQQPackedLinear(nn.Module):
    """Register packed weights and all metadata; dequantize one layer per call."""

    def __init__(self, source, quantizer, values, meta):
        super().__init__()
        self.in_features, self.out_features = source.in_features, source.out_features
        self.quantizer = quantizer
        self.register_buffer("packed_weight", values)
        self.register_buffer("scales", meta["scale"].to(torch.bfloat16))
        self.register_buffer("zeros", meta["zero"].to(torch.bfloat16))
        self.meta = {k: v for k, v in meta.items() if k not in {"scale", "zero"}}
        self.meta["compute_dtype"] = torch.bfloat16
        self.bias = (
            nn.Parameter(source.bias.detach().clone(), requires_grad=False)
            if source.bias is not None
            else None
        )
        self._compression_logical_parameter_elements = self.in_features * self.out_features + (
            self.out_features if self.bias is not None else 0
        )

    def dequantize(self):
        return self.quantizer.dequantize(
            self.packed_weight, {**self.meta, "scale": self.scales, "zero": self.zeros}
        )

    def forward(self, x):
        return F.linear(x.to(torch.bfloat16), self.dequantize(), self.bias)


class GemliteAdapter(nn.Module):
    def __init__(self, source, kernel):
        super().__init__()
        self.in_features, self.out_features = source.in_features, source.out_features
        self.kernel = kernel
        self._compression_logical_parameter_elements = self.in_features * self.out_features + (
            self.out_features if source.bias is not None else 0
        )

    def forward(self, x):
        return self.kernel(x.to(torch.bfloat16))


def hqq_quantize(policy, change):
    manifest = package("hqq")
    from hqq.core.quantize import Quantizer

    bits, axis = change["bits"], change.get("axis", 1)
    backend = change.get("backend", "pytorch")
    if backend == "gemlite":
        package("gemlite")
        import gemlite
        from gemlite.helper import A16W2_HQQ_INT, A16W4_HQQ_INT

        if axis != 1 or bits not in {2, 4}:
            raise ValueError("GemLite supports this HQQ deployment for axis=1, 2/4 bits")
        gemlite.set_autotune("fast")
    events = []
    for full, (root, name, linear) in selected_linears(policy, change["scope"]).items():
        group_size = change.get("group_size", 64)
        if linear.in_features % group_size:
            events.append(
                {
                    "method": "HQQ",
                    "module": full,
                    "converted": False,
                    "reason": "input width does not fit declared group size",
                }
            )
            continue
        values, meta = Quantizer.quantize(
            linear.weight.detach(),
            nbits=bits,
            group_size=group_size,
            axis=axis,
            optimize=True,
            round_zero=False,
            bitpack=True,
            compute_dtype=torch.bfloat16,
            device=str(linear.weight.device),
        )
        if backend == "gemlite":
            unpacked = Quantizer.unpack[meta["packing"]](values, dtype=torch.uint8).reshape(
                meta["shape"]
            )
            helper = A16W2_HQQ_INT if bits == 2 else A16W4_HQQ_INT
            kernel = helper(device=linear.weight.device, dtype=torch.bfloat16).from_weights(
                unpacked,
                meta["scale"].to(torch.bfloat16),
                meta["zero"].to(torch.bfloat16),
                bias=linear.bias.detach() if linear.bias is not None else None,
            )
            converted = GemliteAdapter(linear, kernel)
        else:
            converted = HQQPackedLinear(linear, Quantizer, values, meta)
        replace(root, name, converted)
        events.append(
            {
                "method": "HQQ",
                "revision": manifest["revision"],
                "module": full,
                "bits": bits,
                "axis": axis,
                "group_size": group_size,
                "backend": backend,
                "converted": True,
                "optimizer_training": False,
                "quantizer": "upstream proximal solver",
                "compute_dtype": "bfloat16",
                "metadata_offload": False,
            }
        )
    if not any(event["converted"] for event in events):
        raise ValueError("HQQ converted no weights")
    return events
