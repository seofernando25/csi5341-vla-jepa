"""Weight sparsity and structural removal of channels, attention heads and layers.

Every handler takes (policy, change, calibration) and returns a list of event records.
Importance uses calibration input energy: Wanda scores |W_ij|·sqrt(E[x_j²]).
"""

import math

import torch
from torch import nn

from evaluation.common import ROOT, read_json
from evaluation.compression.topology import (
    SkipBlock,
    blocks,
    replace,
    roots,
    selected_linears,
    slice_linear,
)


def prune_mask(weight, fraction, importance=None, pattern=None):
    """Wanda: per-output row ranking of |W_ij| sqrt(E[x_j²])."""
    scores = weight.float().abs()
    if importance is not None:
        scores *= importance.to(scores.device).clamp_min(0).sqrt()[None]
    if pattern == "2:4":
        if weight.shape[1] % 4:
            raise ValueError("2:4 requires input width divisible by four")
        grouped = scores.reshape(weight.shape[0], -1, 4)
        indices = grouped.argsort(dim=-1, stable=True)[..., :2]
        mask = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, indices, True)
        return mask.reshape_as(weight)
    count = math.floor(weight.shape[1] * fraction)
    return torch.zeros_like(weight, dtype=torch.bool).scatter_(
        1, scores.argsort(dim=1, stable=True)[:, :count], True
    )


def keep_channels(linear, fraction, importance=None, multiple=64):
    scores = linear.weight.detach().float().abs().sum(0)
    if importance is not None:
        scores *= importance.to(scores.device).clamp_min(0).sqrt()
    keep = max(multiple, int(linear.in_features * (1 - fraction)) // multiple * multiple)
    keep = min(keep, linear.in_features)
    return scores.argsort(descending=True, stable=True)[:keep].sort().values


def independent_heads(attn, fraction, importance, vision=False):
    """Physically remove complete heads, preserving the per-head geometry."""
    output = attn.proj if vision else attn.to_out[0]
    count = attn.num_heads if vision else attn.heads
    dim = output.in_features // count
    if not vision and (attn.norm_q is not None or attn.norm_k is not None):
        raise ValueError("Independent head pruning requires unnormalized action Q/K")
    scores = output.weight.float().abs().sum(0) * importance.to(output.weight.device).sqrt()
    scores = scores.reshape(count, dim).sum(-1)
    kept = max(1, math.floor(count * (1 - fraction)))
    heads = scores.argsort(descending=True, stable=True)[:kept].sort().values
    rows = (heads[:, None] * dim + torch.arange(dim, device=heads.device)).flatten()
    if vision:
        qkv_rows = torch.cat([rows + component * count * dim for component in range(3)])
        attn.qkv = slice_linear(attn.qkv, rows=qkv_rows)
        attn.proj = slice_linear(output, columns=rows)
        attn.num_heads = kept
    else:
        for name in ["to_q", "to_k", "to_v"]:
            setattr(attn, name, slice_linear(getattr(attn, name), rows=rows))
        attn.to_out[0] = slice_linear(output, columns=rows)
        attn.heads = kept
        attn.inner_dim = attn.inner_kv_dim = kept * dim
        attn.sliceable_head_dim = kept
    return {
        "original_heads": count,
        "kept_heads": kept,
        "head_dim": dim,
    }


def sparsity(policy, change, calibration):
    """Magnitude or Wanda masks per output row, or 2:4 groups; optional packed 2:4 kernel."""
    method, events = change["method"], []
    stats = calibration.get("linear_inputs", {})
    for full, (_, _, linear) in selected_linears(policy, change["scope"]).items():
        importance = stats.get(full) if method == "wanda" else None
        if method == "wanda" and importance is None:
            raise ValueError(f"Missing calibration: {full}")
        mask = prune_mask(
            linear.weight, change.get("fraction", 0), importance, change.get("pattern")
        )
        linear.weight.data.masked_fill_(mask, 0)
        if change.get("sparse_kernel"):
            from torch.sparse import to_sparse_semi_structured

            linear.weight = nn.Parameter(
                to_sparse_semi_structured(linear.weight), requires_grad=False
            )
        events.append(
            {
                "method": method,
                "module": full,
                "removed_weights": int(mask.sum()),
                "sparse_kernel": change.get("sparse_kernel", False),
            }
        )
    return events


def mlp_pairs(policy, scope):
    """(calibration key of the down projection, MLP module, input projections, output projection)."""
    r = roots(policy)
    if scope == "text_mlp":
        return [
            (f"text.layers.{i}.mlp.down_proj", block.mlp, ["gate_proj", "up_proj"], "down_proj")
            for i, block in enumerate(r["text"].layers)
        ]
    if scope in {"vision_mlp", "merger"}:
        return [
            (f"vision.{name}.linear_fc2", module, ["linear_fc1"], "linear_fc2")
            for name, module in r["vision"].named_modules()
            if hasattr(module, "linear_fc1")
            and hasattr(module, "linear_fc2")
            and (scope == "merger") == ("merger" in name)
        ]
    if scope == "action_mlp":
        return [
            (f"action.model.transformer_blocks.{i}.ff.net.2", block.ff, ["net.0.proj"], "net.2")
            for i, block in enumerate(r["action"].model.transformer_blocks)
        ]
    raise ValueError(f"Unsupported channel scope: {scope}")


def channels(policy, change, calibration):
    """Remove the least important hidden MLP channels from both projections."""
    events = []
    for key, module, ins, out in mlp_pairs(policy, change["scope"]):
        downstream = module.get_submodule(out)
        keep = keep_channels(
            downstream, change.get("fraction", 0), calibration["linear_inputs"][key]
        )
        for name in ins:
            replace(module, name, slice_linear(module.get_submodule(name), rows=keep))
        replace(module, out, slice_linear(downstream, columns=keep))
        events.append(
            {
                "method": "channels",
                "module": key,
                "original_width": downstream.in_features,
                "kept_width": len(keep),
            }
        )
    return events


def grouped_query_heads(attn, fraction, importance):
    """Remove whole key/value groups of Qwen's grouped-query attention with their query heads."""
    dim, groups = attn.head_dim, attn.num_key_value_groups
    kv_count = attn.k_proj.out_features // dim
    keep_count = max(1, round(kv_count * (1 - fraction)))
    scores = (
        (attn.o_proj.weight.float().abs().sum(0) * importance.to(attn.o_proj.weight.device).sqrt())
        .reshape(kv_count, groups, dim)
        .sum((1, 2))
    )
    kv = scores.argsort(descending=True, stable=True)[:keep_count].sort().values
    kv_rows = (kv[:, None] * dim + torch.arange(dim, device=kv.device)).flatten()
    q_heads = (kv[:, None] * groups + torch.arange(groups, device=kv.device)).flatten()
    q_rows = (q_heads[:, None] * dim + torch.arange(dim, device=kv.device)).flatten()
    attn.q_proj = slice_linear(attn.q_proj, rows=q_rows)
    attn.k_proj = slice_linear(attn.k_proj, rows=kv_rows)
    attn.v_proj = slice_linear(attn.v_proj, rows=kv_rows)
    attn.o_proj = slice_linear(attn.o_proj, columns=q_rows)
    return {"original_kv_heads": kv_count, "kept_kv_heads": keep_count}


def heads(policy, change, calibration):
    scope, fraction = change["scope"], change.get("fraction", 0)
    stats = calibration["linear_inputs"]
    events = []
    if scope == "text_attention":
        for index, block in enumerate(blocks(policy, "text")):
            importance = stats[f"text.layers.{index}.self_attn.o_proj"]
            events.append(
                {
                    "method": "heads",
                    "module": f"text.layers.{index}.self_attn",
                    **grouped_query_heads(block.self_attn, fraction, importance),
                }
            )
    elif scope in {"vision_attention", "action_attention"}:
        vision = scope == "vision_attention"
        for index, block in enumerate(blocks(policy, "vision" if vision else "action")):
            key = (
                f"vision.blocks.{index}.attn.proj"
                if vision
                else f"action.model.transformer_blocks.{index}.attn1.to_out.0"
            )
            attn = block.attn if vision else block.attn1
            events.append(
                {
                    "method": "heads",
                    "module": key,
                    **independent_heads(attn, fraction, stats[key], vision),
                }
            )
    else:
        raise ValueError(f"Unsupported head scope: {scope}")
    return events


def layers(policy, change, calibration):
    """Replace the least influential blocks (residual energy or CKA ranking) with SkipBlock."""
    scope = change["scope"]
    layer_list = blocks(policy, scope)
    candidates = list(range(len(layer_list)))
    if scope == "text":
        candidates = candidates[3:]  # Qwen3-VL injects DeepStack visual features into layers 0-2.
    if change.get("criterion") == "cka":
        scores = read_json(ROOT / "studies/compression/cka.json")["scores"]
        order = sorted(candidates, key=lambda i: -scores[f"{scope}.{i}"])
    else:
        influence = calibration["block_influence"]
        order = sorted(candidates, key=lambda i: influence[f"{scope}.{i}"])
    count = change.get("count", max(1, round(len(layer_list) * change.get("fraction", 0))))
    for index in order[:count]:
        layer_list[index] = SkipBlock()
    return [
        {
            "method": "layers",
            "scope": scope,
            "removed_indices": order[:count],
            "criterion": change.get(
                "criterion", "mean relative residual energy on calibration states"
            ),
        }
    ]
