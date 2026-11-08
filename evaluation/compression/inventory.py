"""Parameter counts, resident storage and packed archives of a (compressed) policy."""

import torch
from safetensors.torch import save_file
from torch import nn

from evaluation.common import ROOT, write_json
from evaluation.compression.quantization import Int8Embedding
from evaluation.compression.topology import roots


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


def packed_state(policy):
    tensors, layouts, aliases, seen = {}, {}, {}, {}

    def visit(name, value):
        if hasattr(value, "__tensor_flatten__"):
            names, context = value.__tensor_flatten__()
            layouts[name] = {
                "class": f"{type(value).__module__}.{type(value).__name__}",
                "context": repr(context),
                "fields": names,
            }
            for field in names:
                visit(f"{name}.{field}", getattr(value, field))
        elif torch.is_tensor(value):
            if value.numel() == 0:
                return
            key = (
                str(value.device),
                value.untyped_storage().data_ptr(),
                value.storage_offset(),
                tuple(value.shape),
                tuple(value.stride()),
                value.dtype,
            )
            if key in seen:
                aliases[name] = seen[key]
                return
            seen[key] = name
            tensors[name] = value.detach().cpu().contiguous()
        elif value is not None:
            layouts[name] = {"value": repr(value)}

    for name, value in policy.state_dict().items():
        visit(name, value)
    # The upstream QuantVLA processor stores its calibrated output multiplier
    # as a Python scalar rather than a buffer. It affects inference and must
    # travel with the tensor payload.
    scalars = {
        f"{name}._ohb_beta_scalar": float(module._ohb_beta_scalar)
        for name, module in policy.named_modules()
        if hasattr(module, "_ohb_beta_scalar")
    }
    return tensors, {
        "tensor_subclass_layouts": layouts,
        "aliases": aliases,
        "inference_scalar_attributes": scalars,
    }


def serialize(policy, candidate, result_dir):
    folder = ROOT / "outputs/compression/serialized" / candidate / result_dir.name
    folder.mkdir(parents=True, exist_ok=False)
    tensors, layout = packed_state(policy)
    path = folder / "packed.safetensors"
    save_file(tensors, str(path), metadata={"compression_archive": "1"})
    write_json(result_dir / "tensor_layout.json", layout)
    summary = {
        "bytes": path.stat().st_size,
        "format": "safetensors packed tensor archive; subclass layouts and aliases recorded",
        "artifact": str(path.relative_to(ROOT)),
        "direct_HF_checkpoint": False,
    }
    # Compact evidence is saved before the large archive is removed. Candidates
    # remain reconstructible from the pinned original checkpoint and recipe.
    write_json(result_dir / "serialized.json", summary)
    del tensors
    path.unlink()
    summary["runtime_archive_removed_after_evidence_export"] = True
    return summary
