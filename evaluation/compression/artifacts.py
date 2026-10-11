"""Packed tensor archives and matched inference timing for selected candidates."""

import csv
import gc
import hashlib
import json
import time

import numpy as np
import torch
from safetensors.torch import save_file

from evaluation.common import ROOT, file_hash, write_json
from evaluation.compression.transforms import storage_inventory
from evaluation.run import PROTOCOL, inference_autocast, numpy_raw, prepare, processors


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
    # travel with the tensor payload and be checked across evaluation phases.
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


def tensor_fingerprint(policy):
    """Canonical model tensor payload, independent of archive header order."""
    tensors, layout = packed_state(policy)
    digest = hashlib.sha256()
    total = 0
    for name, value in sorted(tensors.items()):
        header = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        array = value.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(array))
        total += array.nbytes
    digest.update(json.dumps(layout, sort_keys=True, separators=(",", ":")).encode())
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": len(tensors),
        "tensor_bytes": total,
        "algorithm": "SHA256 sorted tensor names/dtypes/shapes/bytes, alias mapping, subclass layouts and calibrated QuantVLA output multipliers; remaining semantics protected by recipe/source hashes",
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
        "sha256": file_hash(path),
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


def benchmark(policy, candidate, result_dir, predictions=500, repetitions=3):
    from evaluation.compression.study import observation_bank

    bank = observation_bank()
    env_pre, pre, post = processors(policy)
    # Compile/autotune every observed input shape before measuring warmed
    # inference. The native per-repetition warmup/counts remain unchanged.
    torch.manual_seed(PROTOCOL["seed"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    priming_started = time.perf_counter()
    for sample in bank:
        policy.reset()
        env_pre.reset()
        pre.reset()
        post.reset()
        with torch.inference_mode(), inference_autocast(policy):
            batch = prepare(numpy_raw(sample["raw"]), sample["instruction"], env_pre, pre)
            action = post(policy.predict_action_chunk(batch)).cpu()
        if not torch.isfinite(action).all():
            raise ValueError("Nonfinite prediction during backend shape priming")
    torch.cuda.synchronize()
    priming = {
        "observations": len(bank),
        "included_in_timing_csv": False,
        "wall_seconds": time.perf_counter() - priming_started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "purpose": "Compile/autotune bank input shapes before steady inference measurements",
    }
    del action, batch
    rows, peaks = [], []
    with (result_dir / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["mode", "repetition", "prediction", "bank_index", "latency_ms"]
        )
        writer.writeheader()
        for mode in ["policy", "pipeline"]:
            for repetition in range(repetitions):
                torch.manual_seed(PROTOCOL["seed"] + repetition)
                gc.collect()
                torch.cuda.empty_cache()
                for iteration in range(PROTOCOL["timing"]["warmup"] + predictions):
                    index = iteration % len(bank)
                    sample = bank[index]
                    raw = numpy_raw(sample["raw"])
                    policy.reset()
                    env_pre.reset()
                    pre.reset()
                    post.reset()
                    batch = (
                        prepare(raw, sample["instruction"], env_pre, pre)
                        if mode == "policy"
                        else None
                    )
                    if iteration == PROTOCOL["timing"]["warmup"]:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode(), inference_autocast(policy):
                        if mode == "pipeline":
                            batch = prepare(raw, sample["instruction"], env_pre, pre)
                        action = policy.predict_action_chunk(batch)
                        if mode == "pipeline":
                            action = post(action).cpu()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - started) * 1000
                    if not torch.isfinite(action).all():
                        raise ValueError("Nonfinite prediction during timing")
                    if iteration >= PROTOCOL["timing"]["warmup"]:
                        row = {
                            "mode": mode,
                            "repetition": repetition,
                            "prediction": iteration - PROTOCOL["timing"]["warmup"],
                            "bank_index": index,
                            "latency_ms": elapsed,
                        }
                        writer.writerow(row)
                        rows.append(row)
                stream.flush()
                peaks.append(
                    {
                        "mode": mode,
                        "repetition": repetition,
                        "allocated_bytes": torch.cuda.max_memory_allocated(),
                        "reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                )
                print(f"{candidate} {mode} repetition {repetition} complete", flush=True)
    summary = {
        "predictions_per_repetition": predictions,
        "repetitions": repetitions,
        "warmup_per_repetition": PROTOCOL["timing"]["warmup"],
        "backend_shape_priming": priming,
        "memory_peaks": peaks,
        **storage_inventory(policy),
        "predicted_actions": policy.model.action_model.action_horizon,
        "executed_before_replanning": policy.config.n_action_steps,
    }
    for mode in ["policy", "pipeline"]:
        values = [r["latency_ms"] for r in rows if r["mode"] == mode]
        summary[mode] = {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }
    summary["serialized_weights"] = serialize(policy, candidate, result_dir)
    return summary
