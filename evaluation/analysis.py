"""Summarize matched B16/Q8/Q4/S500 runs into summary.json and T1_summary.csv.

Example: python -m evaluation.analysis --runs <run-dir> [<run-dir> ...]
Each run directory holds run.json plus episodes.csv or timings.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from evaluation.common import ROOT, episode_seed, read_json, write_json

ORDER = ["B16", "Q8", "Q4", "S500"]


def wilson(successes, total):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return (
        0.0 if successes == 0 else max(0.0, center - radius),
        1.0 if successes == total else min(1.0, center + radius),
    )


def episode_key(row):
    return int(row["task_id"]), int(row["trial"]), row["initial_state_hash"], int(row["seed"])


def paired_difference(baseline, candidate, repetitions=10000):
    left = {episode_key(r): int(r["success"]) for r in baseline}
    right = {episode_key(r): int(r["success"]) for r in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate):
        raise ValueError("Duplicate episode keys")
    if left.keys() != right.keys():
        raise ValueError("Paired episodes must have identical task/state/seed keys")
    rng = np.random.default_rng(42)
    task_draws, differences = [], []
    for task in sorted({k[0] for k in left}):
        values = np.asarray([right[k] - left[k] for k in sorted(left) if k[0] == task])
        differences.append(float(values.mean()))
        task_draws.append(
            values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
        )
    draws = np.stack(task_draws).mean(axis=0)
    return {
        "difference_pp": 100 * float(np.mean(differences)),
        "ci95_pp": (100 * np.quantile(draws, [0.025, 0.975])).tolist(),
        "method": "Task-stratified paired percentile bootstrap; 10000 resamples; seed 42",
    }


def load_run(path):
    record = read_json(path / "run.json")
    if record["status"] != "completed":
        raise ValueError(f"Incomplete/failed run cannot enter comparison: {path.name}")
    name = "timings.csv" if record["experiment"].endswith("benchmark") else "episodes.csv"
    with (path / name).open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(r["variant"] != record["variant"] for r in rows):
        raise ValueError("Empty or mislabeled measurements")
    if record["experiment"] == "rollout":
        tasks = read_json(ROOT / "studies/evaluation/initial_states.json")["tasks"]
        phase, protocol = record["phase"], record["protocol"]
        for row in rows:
            task, trial = int(row["task_id"]), int(row["trial"])
            if task not in protocol["task_ids"] or row["phase"] != phase:
                raise ValueError("Unexpected rollout task or phase")
            hashes = tasks[task][f"{phase}_hashes"]
            if (
                not 0 <= trial < len(hashes)
                or row["initial_state_hash"] != hashes[trial]
                or int(row["seed"]) != episode_seed(task, trial, phase, protocol)
            ):
                raise ValueError("Rollout state/seed does not match the frozen manifest")
    return record, rows


def controls(record):
    """Settings that must match across runs before they can be compared."""
    env = record["environment"]
    return {
        "gpu": env["gpu"],
        "driver": env["driver"],
        "packages": env["packages"],
        "protocol": record["protocol"],
        "sources": record["model"]["sources"],
    }


def timing_summary(record, rows):
    summary = record["summary"]
    predictions, repetitions = summary["predictions_per_repetition"], summary["repetitions"]
    expected = {
        (mode, rep, index)
        for mode in ["policy", "pipeline"]
        for rep in range(repetitions)
        for index in range(predictions)
    }
    keys = {(r["mode"], int(r["repetition"]), int(r["prediction"])) for r in rows}
    if keys != expected or len(rows) != len(expected):
        raise ValueError("Timing records are missing or duplicated")
    result = {"predictions_per_mode": predictions * repetitions, "repetitions": repetitions}
    for mode in ["policy", "pipeline"]:
        values = np.asarray([float(r["latency_ms"]) for r in rows if r["mode"] == mode])
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Invalid timing measurement")
        result[mode] = {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }
        result[mode]["repetition_medians_ms"] = [
            float(
                np.median(
                    [
                        float(r["latency_ms"])
                        for r in rows
                        if r["mode"] == mode and int(r["repetition"]) == rep
                    ]
                )
            )
            for rep in range(repetitions)
        ]
    result["peak_allocated_bytes"] = max(p["allocated_bytes"] for p in summary["memory_peaks"])
    result["peak_reserved_bytes"] = max(p["reserved_bytes"] for p in summary["memory_peaks"])
    result["serialized_weights_bytes"] = summary.get("serialized_weights", {}).get("bytes")
    result["logical_parameters"] = record["model"]["logical_parameters"]
    return result


def core_summary(record, rows):
    expected = {
        (rep, index)
        for rep in range(record["repetitions"])
        for index in range(record["predictions_per_repetition"])
    }
    actual = {(int(r["repetition"]), int(r["prediction"])) for r in rows}
    if actual != expected or len(rows) != len(expected) or any(r["mode"] != "neural" for r in rows):
        raise ValueError("Incomplete/duplicate neural timings")
    equivalence = record["equivalence"]
    if equivalence["observations"] != 100 or not equivalence["identical_flow_noise_seeds"]:
        raise ValueError("Native-policy equivalence was not checked on the full observation bank")
    values = np.asarray([float(r["latency_ms"]) for r in rows])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid neural timings")
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "predictions": len(rows),
        "equivalence": equivalence,
    }


def success_summary(record, rows):
    tasks = record["protocol"]["task_ids"]
    if any(r["status"] != "completed" or r["success"] not in {"0", "1"} for r in rows):
        raise ValueError("Invalid robot outcomes")
    counts = {t: [int(r["success"]) for r in rows if int(r["task_id"]) == t] for t in tasks}
    if (
        {int(r["task_id"]) for r in rows} != set(tasks)
        or not all(counts.values())
        or len({len(v) for v in counts.values()}) != 1
    ):
        raise ValueError("Success comparison requires balanced trials for every task")
    if len({episode_key(r) for r in rows}) != len(rows):
        raise ValueError("Duplicate episode outcomes")
    successes = sum(sum(v) for v in counts.values())
    return {
        "episodes": len(rows),
        "successes": successes,
        "rate": successes / len(rows),
        "ci95": list(wilson(successes, len(rows))),
        "interval_method": "Approximate Wilson interval for balanced independent trials on the fixed task suite",
        "per_task": {
            str(t): {"successes": sum(v), "episodes": len(v), "rate": float(np.mean(v))}
            for t, v in counts.items()
        },
    }


def checkpoint_step(record):
    """Optimizer step of an S500 checkpoint path (.../checkpoints/<step>/pretrained_model)."""
    checkpoint = record.get("checkpoint") or record.get("arguments", {}).get("checkpoint")
    return int(Path(checkpoint).parent.name) if checkpoint else None


def t1_row(variant, result, base):
    row = {"variant": variant, "precision_scope": PRECISION[variant]}
    if "success" in result:
        s = result["success"]
        row.update(
            successes=s["successes"],
            episodes=s["episodes"],
            success_percent=100 * s["rate"],
            ci95_low=100 * s["ci95"][0],
            ci95_high=100 * s["ci95"][1],
        )
        if "paired_baseline" in s:
            paired = s["paired_baseline"]
            row.update(
                difference_pp=paired["difference_pp"],
                difference_ci95_low_pp=paired["ci95_pp"][0],
                difference_ci95_high_pp=paired["ci95_pp"][1],
            )
    if "timing" in result:
        t = result["timing"]
        row.update(
            median_ms=t["pipeline"]["median_ms"],
            p95_ms=t["pipeline"]["p95_ms"],
            peak_allocated_gib=t["peak_allocated_bytes"] / 2**30,
            weights_gib=(t["serialized_weights_bytes"] or 0) / 2**30 or "",
            logical_parameters=t["logical_parameters"],
        )
        if base:
            row.update(
                speedup_vs_B16=base["pipeline"]["median_ms"] / t["pipeline"]["median_ms"],
                memory_reduction_percent=100
                * (1 - t["peak_allocated_bytes"] / base["peak_allocated_bytes"]),
            )
    if "neural" in result:
        row.update(
            neural_median_ms=result["neural"]["median_ms"],
            neural_p95_ms=result["neural"]["p95_ms"],
        )
    training = result.get("study_training", {})
    row.update(
        study_optimizer_steps=training.get("optimizer_steps", ""),
        study_update_gpu_hours=training.get("update_gpu_hours", ""),
        training_seed=training.get("seed", ""),
    )
    return row


PRECISION = {
    "B16": "BF16",
    "Q8": "LLM.int8 decoder linear; other modules BF16",
    "Q4": "NF4 decoder linear; other modules BF16",
    "S500": "BF16",
}
T1_FIELDS = [
    "variant",
    "precision_scope",
    "successes",
    "episodes",
    "success_percent",
    "ci95_low",
    "ci95_high",
    "difference_pp",
    "difference_ci95_low_pp",
    "difference_ci95_high_pp",
    "median_ms",
    "p95_ms",
    "neural_median_ms",
    "neural_p95_ms",
    "speedup_vs_B16",
    "peak_allocated_gib",
    "memory_reduction_percent",
    "weights_gib",
    "logical_parameters",
    "study_optimizer_steps",
    "study_update_gpu_hours",
    "training_seed",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "studies/evaluation/analysis")
    args = parser.parse_args()
    loaded = [load_run(p.resolve()) for p in args.runs]
    reference = controls(loaded[0][0])
    if any(controls(record) != reference for record, _ in loaded):
        raise ValueError("Hardware, software or protocol differs between selected runs")
    results, episodes, phases, s500_step = {}, {}, set(), None
    for record, rows in loaded:
        variant = record["variant"]
        key = {"benchmark": "timing", "core_benchmark": "neural", "rollout": "success"}[
            record["experiment"]
        ]
        result = results.setdefault(variant, {})
        if key in result:
            raise ValueError(f"Multiple {key} runs selected for {variant}; choose explicitly")
        if key == "neural":
            result[key] = core_summary(record, rows)
        elif key == "timing":
            result[key] = timing_summary(record, rows)
        else:
            result[key] = success_summary(record, rows)
            episodes[variant] = rows
            phases.add(record["phase"])
        if variant == "S500" and checkpoint_step(record) is not None:
            s500_step = checkpoint_step(record)
    if len(phases) > 1:
        raise ValueError("Development and final rollouts cannot be pooled")
    if "B16" in episodes:
        for variant, rows in episodes.items():
            results[variant]["success"]["paired_baseline"] = paired_difference(
                episodes["B16"], rows
            )
    for variant, result in results.items():
        if variant != "S500":
            result["study_training"] = {"optimizer_steps": 0, "update_gpu_hours": 0.0}
    adaptation_path = ROOT / "studies/evaluation/adaptation/summary.json"
    if s500_step and adaptation_path.exists():
        adaptation = read_json(adaptation_path)
        updates = [r for r in adaptation["training"] if r["step"] <= s500_step]
        results["S500"]["study_training"] = {
            "optimizer_steps": s500_step,
            "seed": adaptation["recipe"]["seed"],
            "update_gpu_hours": sum(r["update_seconds"] for r in updates) / 3600,
            "scope": "Optimizer updates through this checkpoint; validation and I/O excluded",
        }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "summary.json",
        {"results": results, "controls": reference, "rollout_phase": next(iter(phases), None)},
    )
    base = results.get("B16", {}).get("timing")
    with (args.output / "T1_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=T1_FIELDS)
        writer.writeheader()
        for variant in ORDER:
            if variant in results:
                writer.writerow(t1_row(variant, results[variant], base))
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
