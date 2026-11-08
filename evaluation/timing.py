"""Warmed, CUDA-synchronized latency and peak memory of action-chunk predictions.

Modes: `policy` times predict_action_chunk on a prepared batch; `pipeline` also times outer
observation preparation and action postprocessing. Simulator stepping is never timed.
"""

import csv
import gc
import time

import numpy as np
import torch

from evaluation.common import PROTOCOL
from evaluation.pipeline import inference_autocast
from evaluation.simulator import numpy_raw

MODES = ["policy", "pipeline"]
FIELDS = ["variant", "mode", "repetition", "prediction", "task_id", "bank_index", "latency_ms"]


def prime_shapes(policy, pipeline, bank):
    """Run every bank input once so compiled kernels see each shape before timing."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), inference_autocast():
        for index, sample in enumerate(bank):
            policy.reset()
            torch.manual_seed(PROTOCOL["seed"] + index)
            action = pipeline.post(policy.predict_action_chunk(pipeline.sample(sample)))
            if not torch.isfinite(action).all():
                raise ValueError(f"Nonfinite actions during shape priming at bank_index={index}")
    torch.cuda.synchronize()
    return {
        "observations": len(bank),
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def time_one(policy, pipeline, sample, mode):
    raw = numpy_raw(sample["raw"])
    policy.reset()
    pipeline.reset()
    batch = pipeline.prepare(raw, sample["instruction"]) if mode == "policy" else None
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), inference_autocast():
        if mode == "pipeline":
            batch = pipeline.prepare(raw, sample["instruction"])
        action = policy.predict_action_chunk(batch)
        if mode == "pipeline":
            action = pipeline.post(action).cpu()
    torch.cuda.synchronize()
    if not torch.isfinite(action).all():
        raise ValueError("Nonfinite prediction during timing")
    return (time.perf_counter() - started) * 1000


def time_predictions(policy, pipeline, bank, variant, csv_path, predictions, repetitions):
    """Write one CSV row per measured prediction; return the timing summary."""
    warmup = PROTOCOL["timing"]["warmup"]
    rows, peaks = [], []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for mode in MODES:
            for repetition in range(repetitions):
                torch.manual_seed(PROTOCOL["seed"] + repetition)
                gc.collect()
                torch.cuda.empty_cache()
                for iteration in range(warmup + predictions):
                    index = iteration % len(bank)
                    if iteration == warmup:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    elapsed = time_one(policy, pipeline, bank[index], mode)
                    if iteration >= warmup:
                        row = {
                            "variant": variant,
                            "mode": mode,
                            "repetition": repetition,
                            "prediction": iteration - warmup,
                            "task_id": bank[index]["task_id"],
                            "bank_index": index,
                            "latency_ms": elapsed,
                        }
                        writer.writerow(row)
                        rows.append(row)
                handle.flush()
                peaks.append(
                    {
                        "mode": mode,
                        "repetition": repetition,
                        "allocated_bytes": torch.cuda.max_memory_allocated(),
                        "reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                )
                print(f"{variant} {mode} repetition {repetition} complete", flush=True)
    summary = {
        "predictions_per_repetition": predictions,
        "repetitions": repetitions,
        "warmup_per_repetition": warmup,
        "memory_peaks": peaks,
    }
    for mode in MODES:
        values = [r["latency_ms"] for r in rows if r["mode"] == mode]
        summary[mode] = {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }
    return summary
