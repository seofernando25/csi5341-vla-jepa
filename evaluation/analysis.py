"""Analyze explicitly selected, complete measurements and render publication figures.

Example: python -m evaluation.analysis --runs <B16-run> <Q8-run> <Q4-run>
Run arguments are directories containing run.json and raw CSV evidence.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.figures import COLORS, ORDER, save, style


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
        manifest_path = ROOT / "studies/evaluation/initial_states.json"
        if file_hash(manifest_path) != record["initial_states_manifest_sha256"]:
            raise ValueError("Initial-state manifest differs from recorded rollout protocol")
        tasks = read_json(manifest_path)["tasks"]
        phase, protocol = record["phase"], record["protocol"]
        for row in rows:
            task, trial = int(row["task_id"]), int(row["trial"])
            if task not in protocol["task_ids"] or row["phase"] != phase:
                raise ValueError("Unexpected rollout task or phase")
            hashes = tasks[task][f"{phase}_hashes"]
            seed = (
                protocol["seed"]
                + task * 1000
                + trial
                + (protocol["final_seed_offset"] if phase == "final" else 0)
            )
            if (
                not 0 <= trial < len(hashes)
                or row["initial_state_hash"] != hashes[trial]
                or int(row["seed"]) != seed
            ):
                raise ValueError("Rollout state/seed does not match the frozen manifest")
    return (
        record,
        rows,
        {
            "run": str(path.relative_to(ROOT)),
            "metadata_sha256": file_hash(path / "run.json"),
            "measurements_sha256": file_hash(path / name),
        },
    )


def controls(record):
    env = record["environment"]
    files = env["source_manifest"]
    return {
        "gpu": env["gpu"],
        "driver": env["driver"],
        "packages": env["packages"],
        "protocol": record["protocol"],
        "sources": record["model"]["sources"],
        "initial_states": record["initial_states_manifest_sha256"],
        "inference_source": {
            p: h
            for p, h in files.items()
            if p in {"evaluation/run.py", "evaluation/models.py", "evaluation/common.py"}
            or p.startswith("src/")
        },
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


def resource_figure(results, directory, label):
    variants = [v for v in ORDER if results.get(v, {}).get("timing")]
    if not variants:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.8), layout="constrained")
    xs = np.arange(len(variants))
    colors = [COLORS[v] for v in variants]
    rows = [results[v]["timing"] for v in variants]
    medians = [r["pipeline"]["median_ms"] for r in rows]
    p95 = [r["pipeline"]["p95_ms"] for r in rows]
    axes[0].bar(xs, medians, color=colors, width=0.6)
    axes[0].scatter(xs, p95, marker="_", s=120, color="#253547", label="p95", zorder=3)
    axes[0].set(ylabel="Observation-to-action latency (ms)", title="a   Inference time")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(xs, [r["peak_allocated_bytes"] / 2**30 for r in rows], color=colors, width=0.6)
    axes[1].scatter(
        xs,
        [r["peak_reserved_bytes"] / 2**30 for r in rows],
        marker="_",
        s=120,
        color="#253547",
        label="reserved",
        zorder=3,
    )
    axes[1].set(ylabel="Peak GPU memory (GiB)", title="b   Inference footprint")
    axes[1].legend(frameon=False, fontsize=8)
    for x, row, color in zip(xs, rows, colors, strict=True):
        value = row["serialized_weights_bytes"]
        if value is None:
            axes[2].text(x, 0.02, "pending", rotation=90, ha="center", va="bottom", fontsize=8)
        else:
            axes[2].bar(x, value / 2**30, width=0.6, color=color)
    axes[2].set(ylabel="Serialized weights (GiB)", title="c   Storage")
    for axis in axes:
        axis.set_xticks(xs, variants)
        axis.set_ylim(bottom=0)
        axis.set_axisbelow(True)
        axis.grid(axis="y")
    fig.suptitle(label, fontsize=10)
    save(fig, directory, "F2_resources")


def success_figures(results, directory, phase):
    variants = [v for v in ORDER if results.get(v, {}).get("success")]
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.0), layout="constrained")
    for x, variant in enumerate(variants):
        r = results[variant]["success"]
        y = 100 * r["rate"]
        low, high = np.asarray(r["ci95"]) * 100
        ax.errorbar(
            x,
            y,
            yerr=[[y - low], [high - y]],
            fmt="o",
            color=COLORS[variant],
            capsize=4,
            markersize=7,
        )
        ax.annotate(
            f"{r['successes']}/{r['episodes']}",
            (x, y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set(
        xticks=range(len(variants)),
        xticklabels=variants,
        ylabel="Task success (%)",
        ylim=(-3, 110),
        title=f"LIBERO-Spatial · {phase} · 95% Wilson intervals",
    )
    ax.grid(axis="y")
    save(fig, directory, "F1_success")
    tasks = sorted(results[variants[0]]["success"]["per_task"], key=int)
    matrix = [[100 * results[v]["success"]["per_task"][t]["rate"] for v in variants] for t in tasks]
    fig, ax = plt.subplots(figsize=(4.5, 4.0), layout="constrained")
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="cividis", aspect="auto")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                color="white" if value < 50 else "#202C36",
                fontsize=9,
            )
    ax.set(
        xticks=range(len(variants)),
        xticklabels=variants,
        yticks=range(len(tasks)),
        yticklabels=[f"Task {int(t)}" for t in tasks],
        title=f"Per-task success · {phase}",
    )
    fig.colorbar(im, ax=ax, label="Success (%)", shrink=0.8)
    save(fig, directory, "F4_tasks")
    paired = [v for v in variants if results[v].get("timing")]
    if paired:
        fig, ax = plt.subplots(figsize=(5.8, 3.6), layout="constrained")
        for variant in paired:
            r = results[variant]
            x, y = r["timing"]["pipeline"]["median_ms"], 100 * r["success"]["rate"]
            memory = r["timing"]["peak_allocated_bytes"] / 2**30
            ax.scatter(
                x,
                y,
                s=memory * 55,
                color=COLORS[variant],
                alpha=0.8,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.annotate(
                variant,
                (x, y),
                xytext=(8, 8),
                textcoords="offset points",
                color=COLORS[variant],
                weight="bold",
            )
        ax.set(
            xlabel="Median observation-to-action latency (ms)",
            ylabel="Task success (%)",
            ylim=(-5, 108),
            title=f"RTX 3090 · {phase} · bubble area ∝ peak allocated VRAM",
        )
        ax.grid(alpha=0.6)
        ax.margins(x=0.25)
        save(fig, directory, "F3_tradeoff")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "studies/evaluation/analysis")
    args = parser.parse_args()
    loaded = [load_run(p.resolve()) for p in args.runs]
    reference = controls(loaded[0][0])
    if any(controls(r) != reference for r, _, _ in loaded):
        raise ValueError(
            "Hardware, scientific configuration or inference source differs between selected runs"
        )
    results, evidence, episodes = {}, [], {}
    timing_protocols, phases = set(), set()
    checkpoints = {}
    core_sources = set()
    for record, rows, provenance in loaded:
        variant = record["variant"]
        checkpoint = record.get("checkpoint_sha256")
        if variant in checkpoints and checkpoints[variant] != checkpoint:
            raise ValueError("A variant's success and timing must use the same checkpoint")
        checkpoints[variant] = checkpoint
        key = {"benchmark": "timing", "core_benchmark": "neural", "rollout": "success"}[
            record["experiment"]
        ]
        result = results.setdefault(variant, {})
        if key in result:
            raise ValueError(f"Multiple {key} runs selected for {variant}; choose explicitly")
        if key == "neural":
            result[key] = core_summary(record, rows)
            timing_protocols.add(
                (
                    record["observation_bank_sha256"],
                    record["predictions_per_repetition"],
                    record["repetitions"],
                )
            )
            core_sources.add(
                record["environment"]["source_manifest"]["evaluation/core_benchmark.py"]
            )
        elif key == "timing":
            result[key] = timing_summary(record, rows)
            summary = record["summary"]
            timing_protocols.add(
                (
                    summary["observation_bank_sha256"],
                    summary["predictions_per_repetition"],
                    summary["repetitions"],
                )
            )
        else:
            result[key] = success_summary(record, rows)
            episodes[variant] = rows
            phases.add(record["phase"])
        evidence.append(provenance)
    if len(timing_protocols) > 1 or len(phases) > 1 or len(core_sources) > 1:
        raise ValueError("Different timing protocols or rollout phases cannot be pooled")
    if "B16" in episodes:
        for variant, rows in episodes.items():
            results[variant]["success"]["paired_baseline"] = paired_difference(
                episodes["B16"], rows
            )
    full_timing = bool(timing_protocols) and next(iter(timing_protocols))[1:] == (500, 3)
    for variant, result in results.items():
        if variant != "S500":
            result["study_training"] = {"optimizer_steps": 0, "update_gpu_hours": 0.0}
    adaptation_path = ROOT / "studies/evaluation/adaptation/summary.json"
    if "S500" in results and adaptation_path.exists():
        adaptation = read_json(adaptation_path)
        step = adaptation["checkpoint_steps"].get(checkpoints["S500"])
        if step is not None:
            updates = [r for r in adaptation["training"] if r["step"] <= step]
            if {r["step"] for r in updates} != set(range(1, step + 1)):
                raise ValueError("Checkpoint training cost requires every preceding update")
            results["S500"]["study_training"] = {
                "optimizer_steps": step,
                "seed": adaptation["recipe"]["seed"],
                "update_gpu_hours": sum(r["update_seconds"] for r in updates) / 3600,
                "scope": "Optimizer updates through this checkpoint; validation and I/O excluded",
                "adaptation_summary_sha256": file_hash(adaptation_path),
            }
    phase = next(iter(phases), "not measured")
    label = (
        f"RTX 3090 · {'500 predictions × 3 repetitions/mode' if full_timing else 'timing pilot'}"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "summary.json",
        {
            "results": results,
            "evidence": evidence,
            "controls": reference,
            "rollout_phase": phase,
            "full_timing_protocol": full_timing,
        },
    )
    style()
    resource_figure(results, args.output / "figures", label)
    success_figures(results, args.output / "figures", phase)
    with (args.output / "T1_summary.csv").open("w", newline="") as handle:
        fields = [
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in ORDER:
            r, row = results.get(variant, {}), {"variant": variant}
            row["precision_scope"] = {
                "B16": "BF16",
                "Q8": "LLM.int8 decoder linear; other modules BF16",
                "Q4": "NF4 decoder linear; other modules BF16",
                "S500": "BF16",
            }[variant]
            if "success" in r:
                s = r["success"]
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
            if "timing" in r:
                t = r["timing"]
                row.update(
                    median_ms=t["pipeline"]["median_ms"],
                    p95_ms=t["pipeline"]["p95_ms"],
                    peak_allocated_gib=t["peak_allocated_bytes"] / 2**30,
                    weights_gib=t["serialized_weights_bytes"] / 2**30
                    if t["serialized_weights_bytes"]
                    else "",
                    logical_parameters=t["logical_parameters"],
                )
                if results.get("B16", {}).get("timing"):
                    base = results["B16"]["timing"]
                    row.update(
                        speedup_vs_B16=base["pipeline"]["median_ms"] / t["pipeline"]["median_ms"],
                        memory_reduction_percent=100
                        * (1 - t["peak_allocated_bytes"] / base["peak_allocated_bytes"]),
                    )
            if "neural" in r:
                row.update(
                    neural_median_ms=r["neural"]["median_ms"], neural_p95_ms=r["neural"]["p95_ms"]
                )
            if "study_training" in r:
                row.update(
                    study_optimizer_steps=r["study_training"]["optimizer_steps"],
                    study_update_gpu_hours=r["study_training"]["update_gpu_hours"],
                    training_seed=r["study_training"].get("seed", ""),
                )
            writer.writerow(row)
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
