"""Calibrated GPTQ/SparseGPT adapters and packed INT4 deployment.

The upstream algorithms are pinned in methods.json; their code remains in the
ignored runtime vendor directory. No optimizer training is performed.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

from evaluation.common import ROOT, file_hash, read_json
from evaluation.compression.transforms import replace, selected_linears


class PackedInt4(nn.Module):
    """Groupwise affine INT4 using PyTorch's Ampere-compatible tinygemm kernel."""

    def __init__(self, source, values, scales, zeros, group_size=128, input_scale=None):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.group_size = group_size
        k = math.ceil(self.in_features / 1024) * 1024
        n = math.ceil(self.out_features / 8) * 8
        self.padded_k, self.padded_n = k, n
        q = torch.full((n, k), 8, device=values.device, dtype=torch.uint8)
        q[: self.out_features, : self.in_features] = values.to(torch.uint8)
        packed = (q[:, ::2] << 4) | q[:, 1::2]
        self.register_buffer(
            "packed_weight", torch._convert_weight_to_int4pack(packed.contiguous(), 8)
        )
        scale = torch.ones((n, k // group_size), device=values.device, dtype=torch.float32)
        zero = torch.zeros_like(scale)
        groups = self.in_features // group_size
        scale[: self.out_features, :groups] = scales
        zero[: self.out_features, :groups] = scales * (8 - zeros)
        self.register_buffer(
            "scales_zeros",
            torch.stack((scale, zero), -1).transpose(0, 1).contiguous().to(torch.bfloat16),
        )
        self.bias = (
            nn.Parameter(source.bias.detach().clone(), requires_grad=False)
            if source.bias is not None
            else None
        )
        if input_scale is not None:
            self.register_buffer("input_scale", input_scale.to(torch.bfloat16))
        else:
            self.input_scale = None

    def forward(self, x):
        if self.input_scale is not None:
            x = x / self.input_scale
        flat = x.reshape(-1, self.in_features).to(torch.bfloat16)
        if self.padded_k != self.in_features:
            flat = F.pad(flat, (0, self.padded_k - self.in_features))
        output = torch._weight_int4pack_mm(
            flat, self.packed_weight, self.group_size, self.scales_zeros
        )
        output = output[:, : self.out_features].reshape(*x.shape[:-1], self.out_features)
        return output if self.bias is None else output + self.bias


def affine_quantize(weight, group_size=128):
    groups = weight.float().reshape(weight.shape[0], -1, group_size)
    low = torch.minimum(groups.amin(-1), torch.zeros_like(groups[..., 0]))
    high = torch.maximum(groups.amax(-1), torch.zeros_like(groups[..., 0]))
    scales = ((high - low) / 15).clamp_min(1e-8)
    zeros = (-low / scales).round().clamp(0, 15)
    q = (groups / scales[..., None] + zeros[..., None]).round().clamp(0, 15)
    reconstructed = ((q - zeros[..., None]) * scales[..., None]).reshape_as(weight)
    return q.reshape_as(weight).to(torch.uint8), scales, zeros, reconstructed


def load_upstream(method):
    folder = ROOT / "outputs/compression/vendor" / ("gptq" if method == "gptq" else "sparsegpt")
    manifest = read_json(ROOT / "studies/compression/methods.json")["upstream"][method]
    for name, digest in manifest["files"].items():
        if file_hash(folder / name) != digest:
            raise ValueError(f"Upstream source changed: {method}/{name}")
    # Both repositories use a top-level quant.py import; isolate that name during loading.
    previous = sys.modules.pop("quant", None)
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(
            f"compression_upstream_{method}", folder / f"{method}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        quant = sys.modules["quant"]
    finally:
        sys.path.remove(str(folder))
        sys.modules.pop("quant", None)
        if previous is not None:
            sys.modules["quant"] = previous
    return module, quant


def second_order(policy, change, calibration):
    """Calibrate one block at a time, including preceding compressed blocks."""
    from evaluation.compression.study import inputs, observation_bank
    from evaluation.run import PROTOCOL, processors

    method = change["method"]
    upstream, quant = load_upstream(method)
    selected = selected_linears(policy, change["scope"])
    groups = {}
    for full, entry in selected.items():
        parts = full.split(".")
        # Group complete decoder, vision or DiT blocks; interface linears form their own group.
        group = ".".join(parts[:3]) if parts[0] != "action" else ".".join(parts[:4])
        groups.setdefault(group, []).append((full, entry))
    bank = observation_bank()
    pipeline = processors(policy)
    events = []
    for group, entries in groups.items():
        helpers, handles = {}, []
        for full, (_, _, linear) in entries:
            helper = upstream.GPTQ(linear) if method == "gptq" else upstream.SparseGPT(linear)
            if method == "gptq":
                helper.quantizer = quant.Quantizer()
                helper.quantizer.configure(4, perchannel=True, sym=False, mse=False)
            helpers[full] = helper

            def hook(module, args, output, helper=helper):
                # Calibration outer autocast must not turn Hessian products into BF16.
                with torch.autocast("cuda", enabled=False):
                    helper.add_batch(args[0].detach().float(), output.detach().float())

            handles.append(linear.register_forward_hook(hook))
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for index in calibration["bank_indices"]:
                    torch.manual_seed(PROTOCOL["seed"] + index)
                    policy.predict_action_chunk(inputs(bank[index], pipeline))
        finally:
            for handle in handles:
                handle.remove()
        for full, (root, name, linear) in entries:
            helper = helpers[full]
            if method == "gptq":
                scales, zeros = [], []
                original = helper.quantizer.find_params

                def save_qparams(
                    x, weight=False, original=original, scales=scales, zeros=zeros, helper=helper
                ):
                    original(x, weight=weight)
                    if x.shape[-1] == 128:
                        scales.append(helper.quantizer.scale.detach().clone().flatten())
                        zeros.append(helper.quantizer.zero.detach().clone().flatten())

                helper.quantizer.find_params = save_qparams
                helper.fasterquant(blocksize=128, percdamp=0.01, groupsize=128, actorder=False)
                scale = torch.stack(scales, 1)
                zero = torch.stack(zeros, 1)
                q = (
                    (
                        linear.weight.detach().float().reshape(linear.out_features, -1, 128)
                        / scale[..., None]
                        + zero[..., None]
                    )
                    .round()
                    .clamp(0, 15)
                )
                replace(root, name, PackedInt4(linear, q.reshape_as(linear.weight), scale, zero))
                events.append(
                    {
                        "method": "GPTQ",
                        "module": full,
                        "bits": 4,
                        "group_size": 128,
                        "act_order": False,
                        "damping": 0.01,
                        "packed_kernel": "aten tinygemm",
                    }
                )
            else:
                pattern = change.get("pattern")
                helper.fasterprune(
                    change.get("fraction", 0.5),
                    prunen=2 if pattern == "2:4" else 0,
                    prunem=4 if pattern == "2:4" else 0,
                    percdamp=0.01,
                )
                removed = int((linear.weight == 0).sum())
                if change.get("sparse_kernel"):
                    if pattern != "2:4":
                        raise ValueError("Packed sparse deployment requires a 2:4 pattern")
                    linear.weight = nn.Parameter(
                        torch.sparse.to_sparse_semi_structured(linear.weight),
                        requires_grad=False,
                    )
                events.append(
                    {
                        "method": "SparseGPT",
                        "module": full,
                        "fraction": change.get("fraction", 0.5),
                        "pattern": pattern,
                        "removed_weights": removed,
                        "storage": "2:4 packed" if change.get("sparse_kernel") else "dense zeros",
                    }
                )
            helper.free()
        print(f"{method} completed {group}", flush=True)
    return events


def activation_scaled_int4(policy, change, calibration):
    """AWQ-inspired per-linear scale search with activation-weighted reconstruction.

    This adaptation searches diagonal input-energy reconstruction, rather than
    the original AWQ block-output objective; label that distinction in results.
    """
    events = []
    for full, (root, name, linear) in selected_linears(policy, change["scope"]).items():
        if linear.in_features % 128:
            raise ValueError(f"INT4 group alignment unsupported: {full}")
        weight = linear.weight.detach().float()
        activation = calibration["linear_inputs"][full].to(weight.device).clamp_min(1e-8)
        best = None
        for alpha in torch.linspace(0, 0.9, 10).tolist():
            scale = activation.sqrt().pow(alpha).clamp_min(1e-4)
            scale = scale / (scale.amin() * scale.amax()).sqrt()
            q, s, z, reconstruction = affine_quantize(weight * scale)
            error = float(((reconstruction / scale - weight).square() * activation[None]).mean())
            if best is None or error < best[0]:
                best = (error, alpha, scale, q, s, z)
        error, alpha, scale, q, s, z = best
        replace(root, name, PackedInt4(linear, q, s, z, input_scale=scale))
        events.append(
            {
                "method": "AWQ-inspired diagonal reconstruction",
                "module": full,
                "bits": 4,
                "group_size": 128,
                "alpha": alpha,
                "weighted_weight_error": error,
                "packed_kernel": "aten tinygemm",
            }
        )
    return events


def torchao_quantize(policy, change):
    folder = ROOT / "outputs/compression/vendor_packages"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    from torchao.quantization import (
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        quantize_,
    )
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat

    config = (
        Int4WeightOnlyConfig(
            group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D
        )
        if change["bits"] == 4
        else Int8DynamicActivationInt8WeightConfig()
    )
    entries = selected_linears(policy, change["scope"])
    converted, skipped = {}, []
    for full, (root, name, linear) in entries.items():
        holder = nn.Module()
        holder.add_module("linear", linear)
        quantize_(holder, config)
        replace(root, name, holder.linear)
        if hasattr(holder.linear.weight, "__tensor_flatten__"):
            converted[full] = type(holder.linear.weight).__name__
        else:
            skipped.append(
                {
                    "module": full,
                    "shape": list(holder.linear.weight.shape),
                    "reason": "backend retained dense weight",
                }
            )
    if not converted:
        raise ValueError("torchao converted no weights; refusing mislabeled quantization")
    return [
        {
            "method": "torchao",
            "version": "0.18.0",
            "configuration": repr(config),
            "converted_modules": converted,
            "skipped_modules": skipped,
        }
    ]
