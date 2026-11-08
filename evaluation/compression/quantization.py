"""Weight quantization: bitsandbytes INT8/NF4, packed INT4 (AWQ-style), TorchAO, embeddings.

Every handler takes (policy, change, calibration) and returns a list of event records.
"""

import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

from evaluation.common import ROOT
from evaluation.compression.topology import replace, roots, selected_linears
from evaluation.models import quantize_decoder

ACTION_SCOPES = {"action", "action_interfaces", "all"}


def guard_action_dtype(policy, scope):
    """Keep the action head's timestep encoder in BF16 after its Linear weights are packed.

    The native TimestepEncoder takes its activation dtype from its first parameter, which
    becomes integer data once quantized. A zero-element BF16 parameter restores the contract.
    """
    if scope not in ACTION_SCOPES:
        return []
    encoder = roots(policy)["action"].model.timestep_encoder
    if hasattr(encoder, "_activation_dtype_anchor"):
        return []
    device = next(encoder.parameters()).device
    encoder.register_parameter(
        "_activation_dtype_anchor",
        nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False),
    )
    return [
        {
            "method": "activation_dtype_guard",
            "module": "action.model.timestep_encoder",
            "activation_dtype": "bfloat16",
            "parameter_elements": 0,
        }
    ]


class Int8Embedding(nn.Module):
    """Real row-wise packed INT8 weights, with per-row FP32 scales."""

    def __init__(self, source):
        super().__init__()
        weight = source.weight.detach().float()
        scales = weight.abs().amax(1).clamp_min(1e-8) / 127
        self.register_buffer(
            "values", (weight / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
        )
        self.register_buffer("scales", scales)
        self.num_embeddings, self.embedding_dim = weight.shape
        self.output_dtype = source.weight.dtype

    def forward(self, ids):
        return (F.embedding(ids, self.values).float() * self.scales[ids, None]).to(
            self.output_dtype
        )


class CompactEmbedding(nn.Module):
    """Fixed-task vocabulary pruning; unseen words fail instead of silently remapping."""

    def __init__(self, source, token_ids):
        super().__init__()
        ids = torch.unique(token_ids.to(source.weight.device)).sort().values
        lookup = torch.full((source.num_embeddings,), -1, device=ids.device, dtype=torch.int32)
        lookup[ids] = torch.arange(len(ids), device=ids.device, dtype=torch.int32)
        self.register_buffer("lookup", lookup)
        self.register_buffer("original_ids", ids)
        self.weight = nn.Parameter(source.weight.detach()[ids].clone(), requires_grad=False)
        self.num_embeddings, self.embedding_dim = self.weight.shape

    def forward(self, ids):
        mapped = self.lookup[ids].long()
        if bool((mapped < 0).any()):
            raise ValueError("Unseen token ID in restricted LIBERO-Spatial vocabulary")
        return F.embedding(mapped, self.weight)


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


def bitsandbytes(policy, change, calibration):
    """LLM.int8 (bits=8) or NF4 (bits=4) on every Linear in `scope`."""
    events = guard_action_dtype(policy, change["scope"])
    for full, (root, name, module) in selected_linears(policy, change["scope"]).items():
        holder = nn.Module()
        holder.add_module("linear", module)
        converted = quantize_decoder(holder, change["bits"])
        replace(root, name, holder.linear)
        events.append({"method": "bnb", **converted[0], "module": full})
    return events


def embedding(policy, change, calibration):
    """INT8 rows for the full vocabulary, or BF16 rows for LIBERO's prompt tokens only."""
    text = roots(policy)["text"]
    # The policy calls the inner model, so the tied vocabulary output head is unused.
    policy.model.qwen.model.lm_head = nn.Identity()
    if change["method"] == "int8_embedding":
        text.embed_tokens = Int8Embedding(text.embed_tokens)
    else:
        text.embed_tokens = CompactEmbedding(text.embed_tokens, calibration["token_ids"])
    return [{"method": change["method"], "stored_rows": text.embed_tokens.num_embeddings}]


def activation_scaled_int4(policy, change, calibration):
    """AWQ-inspired per-linear scale search with activation-weighted reconstruction.

    This adaptation searches diagonal input-energy reconstruction, rather than
    the original AWQ block-output objective; label that distinction in results.
    """
    events = guard_action_dtype(policy, change["scope"])
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


def torchao_quantize(policy, change, calibration):
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
    guard = guard_action_dtype(policy, change["scope"])
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
    return guard + [
        {
            "method": "torchao",
            "version": "0.18.0",
            "configuration": repr(config),
            "converted_modules": converted,
            "skipped_modules": skipped,
        }
    ]
