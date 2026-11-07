"""Reversible in-memory compression recipes; never edit the published weights."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from evaluation.models import quantize_decoder


class SkipBlock(nn.Module):
    """Preserve original layer indices and DeepStack injection positions."""

    def forward(self, hidden_states=None, *args, **kwargs):
        return hidden_states if hidden_states is not None else args[0]


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


def replace(root, name, module):
    parent, _, leaf = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, leaf, module)


def slice_linear(linear, rows=None, columns=None):
    weight = linear.weight.detach()
    if rows is not None:
        weight = weight[rows]
    if columns is not None:
        weight = weight[:, columns]
    result = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=linear.bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    result.weight = nn.Parameter(weight.clone(), requires_grad=False)
    if linear.bias is not None:
        bias = linear.bias.detach() if rows is None else linear.bias.detach()[rows]
        result.bias = nn.Parameter(bias.clone(), requires_grad=False)
    return result.eval()


def roots(policy):
    inner = policy.model.qwen.model.model
    return {
        "text": inner.language_model,
        "vision": inner.visual,
        "action": policy.model.action_model,
    }


def selected_linears(policy, scope):
    result = {}
    for root_name, root in roots(policy).items():
        for name, module in root.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if scope == root_name or scope == "all":
                include = True
            elif scope == "text_mlp":
                include = root_name == "text" and ".mlp." in name
            elif scope == "text_attention":
                include = root_name == "text" and ".self_attn." in name
            elif scope == "vision_blocks":
                include = root_name == "vision" and name.startswith("blocks.")
            elif scope == "merger":
                include = root_name == "vision" and ("merger" in name)
            elif scope == "vision_mlp":
                include = root_name == "vision" and ".mlp." in name
            elif scope == "vision_attention":
                include = root_name == "vision" and ".attn." in name
            elif scope == "action_mlp":
                include = root_name == "action" and ".ff." in name
            elif scope == "action_attention":
                include = root_name == "action" and ".attn1." in name
            elif scope == "action_interfaces":
                include = root_name == "action" and not name.startswith("model.transformer_blocks.")
            else:
                include = False
            if include:
                result[f"{root_name}.{name}"] = (root, name, module)
    if not result:
        raise ValueError(f"No Linear modules selected for {scope}")
    return result


def tensor_digest(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


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
        "kept_indices_sha256": tensor_digest(heads),
    }


def apply(policy, recipe, calibration=None):
    """Apply one isolated recipe or an explicitly ordered combination."""
    calibration = calibration or {}
    stats = calibration.get("linear_inputs", {})
    events = []
    r = roots(policy)
    for change in recipe.get("changes", []):
        method, scope = change["method"], change.get("scope")
        fraction = change.get("fraction", 0)
        if method in {"bnb", "gptq", "awq_diagonal", "torchao", "hqq"} and scope in {
            "action",
            "action_interfaces",
            "all",
        }:
            # Native TimestepEncoder derives activation dtype from its first parameter.
            # Keep that BF16 contract when a child's weight becomes packed integer data.
            encoder = r["action"].model.timestep_encoder
            if not hasattr(encoder, "_activation_dtype_anchor"):
                encoder.register_parameter(
                    "_activation_dtype_anchor",
                    nn.Parameter(
                        torch.empty(
                            0, device=next(encoder.parameters()).device, dtype=torch.bfloat16
                        ),
                        requires_grad=False,
                    ),
                )
                events.append(
                    {
                        "method": "activation_dtype_guard",
                        "module": "action.model.timestep_encoder",
                        "activation_dtype": "bfloat16",
                        "parameter_elements": 0,
                    }
                )
        if method == "hqq":
            from evaluation.compression.lowbit import hqq_quantize

            events.extend(hqq_quantize(policy, change))
        elif method == "attention_matching":
            from evaluation.compression.attention_matching import matching

            events.extend(matching(policy, change, calibration))
        elif method in {"gptq", "sparsegpt", "awq_diagonal", "torchao"}:
            from evaluation.compression.advanced import (
                activation_scaled_int4,
                second_order,
                torchao_quantize,
            )

            if method in {"gptq", "sparsegpt"}:
                events.extend(second_order(policy, change, calibration))
            elif method == "awq_diagonal":
                events.extend(activation_scaled_int4(policy, change, calibration))
            else:
                events.extend(torchao_quantize(policy, change))
        elif method == "compile_linears":
            from evaluation.compression.compiled import compile_linears

            events.append(compile_linears(policy, change))
        elif method == "visual_tokens":
            from evaluation.compression.tokens import visual_token_pruning

            events.append(
                visual_token_pruning(policy, fraction, change.get("criterion", "magnitude"))
            )
        elif method == "world_remove":
            policy.model.video_encoder = None
            policy.model.video_predictor = None
            policy.model.video_processor = None
            events.append({"method": method, "outputs_expected_identical": True})
        elif method in {"int8_embedding", "compact_embedding"}:
            # This policy calls the inner model, so the tied vocabulary output head is unused.
            policy.model.qwen.model.lm_head = nn.Identity()
            source = r["text"].embed_tokens
            r["text"].embed_tokens = (
                Int8Embedding(source)
                if method == "int8_embedding"
                else CompactEmbedding(source, calibration["token_ids"])
            )
            events.append({"method": method, "stored_rows": r["text"].embed_tokens.num_embeddings})
        elif method == "bnb":
            modules = selected_linears(policy, scope)
            # Existing audited conversion accepts any module root containing Linear leaves.
            for full, (root, name, module) in modules.items():
                holder = nn.Module()
                holder.add_module("linear", module)
                converted = quantize_decoder(holder, change["bits"])
                replace(root, name, holder.linear)
                events.append({"method": method, **converted[0], "module": full})
        elif method in {"magnitude", "wanda"}:
            for full, (_, _, linear) in selected_linears(policy, scope).items():
                importance = stats.get(full) if method == "wanda" else None
                if method == "wanda" and importance is None:
                    raise ValueError(f"Missing calibration: {full}")
                mask = prune_mask(linear.weight, fraction, importance, change.get("pattern"))
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
                        "mask_sha256": tensor_digest(mask),
                        "sparse_kernel": change.get("sparse_kernel", False),
                    }
                )
        elif method == "channels":
            modules = []
            if scope == "text_mlp":
                for index, block in enumerate(r["text"].layers):
                    mlp = block.mlp
                    modules.append(
                        (
                            f"text.layers.{index}.mlp.down_proj",
                            mlp,
                            ["gate_proj", "up_proj"],
                            "down_proj",
                        )
                    )
            elif scope in {"vision_mlp", "merger"}:
                for name, module in r["vision"].named_modules():
                    if hasattr(module, "linear_fc1") and hasattr(module, "linear_fc2"):
                        if (scope == "merger") != ("merger" in name):
                            continue
                        modules.append(
                            (f"vision.{name}.linear_fc2", module, ["linear_fc1"], "linear_fc2")
                        )
            elif scope == "action_mlp":
                for index, block in enumerate(r["action"].model.transformer_blocks):
                    modules.append(
                        (
                            f"action.model.transformer_blocks.{index}.ff.net.2",
                            block.ff,
                            ["net.0.proj"],
                            "net.2",
                        )
                    )
            else:
                raise ValueError(f"Unsupported channel scope: {scope}")
            for key, module, ins, out in modules:
                downstream = module.get_submodule(out)
                keep = keep_channels(downstream, fraction, stats[key])
                for name in ins:
                    replace(module, name, slice_linear(module.get_submodule(name), rows=keep))
                replace(module, out, slice_linear(downstream, columns=keep))
                events.append(
                    {
                        "method": method,
                        "module": key,
                        "original_width": downstream.in_features,
                        "kept_width": len(keep),
                        "indices_sha256": tensor_digest(keep),
                    }
                )
        elif method == "layers":
            layer_list = {
                "text": r["text"].layers,
                "vision": r["vision"].blocks,
                "action": r["action"].model.transformer_blocks,
            }[scope]
            candidates = list(range(len(layer_list)))
            if scope == "text":
                candidates = candidates[3:]  # Keep all early DeepStack fusion positions.
            if change.get("criterion") == "cka":
                from evaluation.common import ROOT, read_json

                scores = read_json(ROOT / "studies/compression/cka.json")["scores"]
                order = sorted(candidates, key=lambda i: -scores[f"{scope}.{i}"])
            else:
                order = sorted(
                    candidates, key=lambda i: calibration["block_influence"][f"{scope}.{i}"]
                )
            count = change.get("count", max(1, round(len(layer_list) * fraction)))
            for index in order[:count]:
                layer_list[index] = SkipBlock()
            events.append(
                {
                    "method": method,
                    "scope": scope,
                    "removed_indices": order[:count],
                    "criterion": change.get(
                        "criterion", "mean relative residual energy on calibration states"
                    ),
                }
            )
        elif method == "heads":
            if scope in {"vision_attention", "action_attention"}:
                vision = scope == "vision_attention"
                blocks = r["vision"].blocks if vision else r["action"].model.transformer_blocks
                for index, block in enumerate(blocks):
                    attn = block.attn if vision else block.attn1
                    key = (
                        f"vision.blocks.{index}.attn.proj"
                        if vision
                        else f"action.model.transformer_blocks.{index}.attn1.to_out.0"
                    )
                    events.append(
                        {
                            "method": method,
                            "module": key,
                            **independent_heads(attn, fraction, stats[key], vision),
                        }
                    )
                continue
            if scope != "text_attention":
                raise ValueError("Only grouped-query text heads implemented")
            for index, block in enumerate(r["text"].layers):
                attn = block.self_attn
                dim = attn.head_dim
                groups = attn.num_key_value_groups
                kv_count = attn.k_proj.out_features // dim
                keep_count = max(1, round(kv_count * (1 - fraction)))
                importance = stats[f"text.layers.{index}.self_attn.o_proj"].to(
                    attn.o_proj.weight.device
                )
                scores = (
                    (attn.o_proj.weight.float().abs().sum(0) * importance.sqrt())
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
                events.append(
                    {
                        "method": method,
                        "module": f"text.layers.{index}.self_attn",
                        "original_kv_heads": kv_count,
                        "kept_kv_heads": keep_count,
                    }
                )
        elif method == "conditioning_tokens":
            if scope == "latent":
                policy.model.replace_prompt = ""
            else:
                count = change["count"]
                policy.model.embodied_replace_prompt = policy.config.embodied_action_token * count
            events.append(change)
        elif method == "solver_steps":
            r["action"].num_inference_timesteps = change["count"]
            events.append(change)
        elif method == "action_horizon":
            count = change["count"]
            if not 1 <= count <= policy.config.chunk_size:
                raise ValueError("Action horizon must fit the trained chunk")
            r["action"].action_horizon = count
            policy.config.n_action_steps = count
            policy.reset()
            events.append(
                {
                    **change,
                    "trained_chunk_size": policy.config.chunk_size,
                    "executed_before_replanning": count,
                }
            )
        elif method == "replanning":
            count = change["count"]
            if not 1 <= count <= r["action"].action_horizon:
                raise ValueError("Replanning horizon must fit the predicted chunk")
            policy.config.n_action_steps = count
            policy.reset()
            events.append({**change, "predicted_chunk_size": r["action"].action_horizon})
        else:
            raise ValueError(f"Unknown compression method: {method}")
    return events


def logical_parameter_count(module):
    """Count represented learned weights independently of their packed dtype.

    Quantization changes bytes, not model width. Packed INT4 modules/INT8
    embeddings store weights as buffers; sparse tensor numel is already logical.
    """
    count = sum(parameter.numel() for parameter in module.parameters())
    seen = set()
    covered = set()
    for child in module.modules():
        if id(child) in covered:
            continue
        if hasattr(child, "_compression_logical_parameter_elements"):
            count += child._compression_logical_parameter_elements - sum(
                p.numel() for p in child.parameters()
            )
            covered.update(id(nested) for nested in child.modules())
            continue
        if isinstance(child, nn.Linear):
            if id(child.weight) not in seen:
                count += child.in_features * child.out_features - child.weight.numel()
                seen.add(id(child.weight))
        elif hasattr(child, "packed_weight") and hasattr(child, "in_features"):
            count += child.in_features * child.out_features
        elif isinstance(child, Int8Embedding):
            count += child.num_embeddings * child.embedding_dim
    return count


def storage_inventory(policy):
    """Deduplicate actual packed tensors, tied weights and quantizer metadata."""
    seen = set()
    total = 0

    def add(value):
        nonlocal total
        if not torch.is_tensor(value):
            return
        if hasattr(value, "__tensor_flatten__"):
            names, _ = value.__tensor_flatten__()
            for name in names:
                add(getattr(value, name))
            return
        try:
            storage = value.untyped_storage()
        except RuntimeError:
            add(value.values())
            add(value.indices())
            return
        key = (value.device.type, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
        state = getattr(value, "quant_state", None)
        while state is not None:
            for field in ["absmax", "code", "offset"]:
                add(getattr(state, field, None))
            state = getattr(state, "state2", None)
        for field in ["CB", "SCB"]:
            add(getattr(value, field, None))

    parameters = list(policy.parameters())
    for value in parameters + list(policy.buffers()):
        add(value)
    for module in policy.modules():
        state = getattr(module, "state", None)
        if state is not None:
            for field in ["CB", "SCB"]:
                add(getattr(state, field, None))
    result = {
        "registered_parameter_elements": sum(v.numel() for v in parameters),
        "represented_logical_parameter_elements": logical_parameter_count(policy),
        "resident_tensor_storage_bytes": total,
    }
    if hasattr(policy, "model") and hasattr(policy.model, "qwen"):
        result["inference_core_logical_parameter_elements"] = sum(
            logical_parameter_count(root) for root in roots(policy).values()
        )
    return result
