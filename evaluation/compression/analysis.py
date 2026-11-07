"""Export measured component coverage and paired task-success comparisons."""

import csv

import numpy as np

from evaluation.analysis import episode_key, paired_difference, timing_summary, wilson
from evaluation.common import file_hash, read_json, write_json
from evaluation.compression.protocol import STUDY, matched_controls, validated_episodes
from evaluation.compression.queue import ledger


def validate_confirmation_record(record, candidate):
    frozen_path = STUDY / "final_selection.json"
    if record.get("final_registration", {}).get("selection_sha256") != file_hash(frozen_path):
        raise ValueError("Confirmation run differs from the frozen selection")
    frozen = read_json(frozen_path)
    if (
        record.get("deployed_tensor_fingerprint", {}).get("sha256")
        != frozen["deployed_tensor_sha256"][candidate]
    ):
        raise ValueError("Confirmation model payload differs from the frozen selection")


def paired_outcomes(left, right, phase):
    baseline = {episode_key(row): int(row["success"]) for row in left}
    candidate = {episode_key(row): int(row["success"]) for row in right}
    if (
        len(baseline) != len(left)
        or len(candidate) != len(right)
        or baseline.keys() != candidate.keys()
    ):
        raise ValueError("Paired task/state/seed membership differs or duplicates exist")
    counts = {"both_success": 0, "both_failure": 0, "gained_success": 0, "lost_success": 0}
    for key, first in baseline.items():
        second = candidate[key]
        name = (
            "both_success"
            if first and second
            else "lost_success"
            if first
            else "gained_success"
            if second
            else "both_failure"
        )
        counts[name] += 1
    result = {
        "transitions": counts,
        "difference_pp": 100 * (sum(candidate.values()) - sum(baseline.values())) / len(left),
    }
    if phase == "smoke":
        result["ci95_pp"] = None
        result["uncertainty"] = (
            "One state per task; within-task bootstrap cannot estimate uncertainty"
        )
    else:
        result.update(paired_difference(left, right))
    return result


def rollout_summary(candidate, phase, baseline):
    folder = STUDY / "measurements" / candidate / phase
    record, rows = validated_episodes(folder)
    successes = sum(int(row["success"]) for row in rows)
    result = {
        "episodes": len(rows),
        "successes": successes,
        "success_fraction": successes / len(rows),
        "wilson95_fraction": list(wilson(successes, len(rows))),
        "task_successes": {
            str(task): sum(int(row["success"]) for row in rows if int(row["task_id"]) == task)
            for task in range(10)
        },
        "episodes_sha256": file_hash(folder / "episodes.csv"),
        "record_sha256": file_hash(folder / "run.json"),
    }
    if baseline is not None:
        matched_controls(baseline[0], record)
        result["paired_vs_B16"] = paired_outcomes(baseline[1], rows, phase)
    if phase == "final":
        validate_confirmation_record(record, candidate)
        result["meets_90_percent_observed_success"] = successes >= 450
    return result


def export():
    rows = ledger()
    baselines = {}
    for phase in ["smoke", "development", "final"]:
        path = STUDY / "measurements/B16" / phase / "run.json"
        if path.exists() and read_json(path)["status"] == "completed":
            baselines[phase] = validated_episodes(path.parent)
    results, table = {}, []
    for row in rows:
        candidate = row["candidate"]
        result = {
            "family": row["family"],
            "domains": row["domains"],
            "deployment_scope": "restricted LIBERO-Spatial vocabulary"
            if any(c["method"] == "compact_embedding" for c in row["changes"])
            else "full vocabulary",
            "changes": row["changes"],
            "phases": {},
        }
        for phase, phase_record in row["phases"].items():
            result["phases"][phase] = {
                "status": phase_record["status"],
                "error": phase_record["error"],
            }
            if phase_record["status"] != "completed":
                continue
            folder = STUDY / "measurements" / candidate / phase
            fingerprint = read_json(folder / "run.json").get("deployed_tensor_fingerprint")
            if fingerprint is not None:
                result["phases"][phase]["deployed_tensor_fingerprint"] = fingerprint
            if phase in ["smoke", "development", "final"]:
                result["phases"][phase]["summary"] = rollout_summary(
                    candidate, phase, baselines.get(phase)
                )
            elif phase == "benchmark":
                record = read_json(folder / "run.json")
                validate_confirmation_record(record, candidate)
                with (folder / "timings.csv").open() as stream:
                    timings = list(csv.DictReader(stream))
                result["phases"][phase]["summary"] = timing_summary(record, timings)
                timing = result["phases"][phase]["summary"]
                # Native metadata describes the loaded original checkpoint.
                # Keep it distinct from the measured transformed policy.
                timing["original_checkpoint_parameter_elements"] = timing.pop(
                    "logical_parameters"
                )
                result["phases"][phase]["summary"].update(
                    {
                        k: v
                        for k, v in record["summary"].items()
                        if "parameter_elements" in k
                        or k in {
                            "predicted_actions",
                            "executed_before_replanning",
                            "backend_shape_priming",
                            "warmup_per_repetition",
                        }
                    }
                )
            else:
                result["phases"][phase]["summary"] = phase_record["summary"]
        results[candidate] = result
        screen = result["phases"].get("screen", {}).get("summary", {})
        flat = {
            "candidate": candidate,
            "family": row["family"],
            "deployment_scope": result["deployment_scope"],
            "screen_status": result["phases"].get("screen", {}).get("status", "planned"),
            "screen_peak_allocated_bytes": screen.get("peak_allocated_bytes"),
            "screen_pipeline_median_ms": screen.get("pipeline_median_ms"),
            "resident_tensor_storage_bytes": screen.get("resident_tensor_storage_bytes"),
            "represented_logical_parameter_elements": screen.get(
                "represented_logical_parameter_elements"
            ),
            "inference_core_logical_parameter_elements": screen.get(
                "inference_core_logical_parameter_elements"
            ),
        }
        for phase in ["smoke", "development", "final"]:
            summary = result["phases"].get(phase, {}).get("summary", {})
            flat[f"{phase}_successes"] = summary.get("successes")
            flat[f"{phase}_episodes"] = summary.get("episodes")
            flat[f"{phase}_paired_change_pp"] = summary.get("paired_vs_B16", {}).get(
                "difference_pp"
            )
        size = result["phases"].get("size", {}).get("summary", {})
        flat["packed_archive_bytes"] = size.get("serialized_weights", {}).get("bytes")
        benchmark = result["phases"].get("benchmark", {}).get("summary", {})
        flat["benchmark_peak_allocated_bytes"] = benchmark.get("peak_allocated_bytes")
        flat["benchmark_pipeline_median_ms"] = benchmark.get("pipeline", {}).get("median_ms")
        flat["benchmark_pipeline_p95_ms"] = benchmark.get("pipeline", {}).get("p95_ms")
        for field in ["predicted_actions", "executed_before_replanning"]:
            flat[field] = benchmark.get(field, screen.get(field))
        flat["deployed_tensor_sha256"] = next(
            (
                result["phases"][phase]["deployed_tensor_fingerprint"]["sha256"]
                for phase in ["benchmark", "final", "development", "size"]
                if result["phases"].get(phase, {}).get("deployed_tensor_fingerprint")
            ),
            None,
        )
        table.append(flat)
    baseline_benchmark = results.get("B16", {}).get("phases", {}).get("benchmark", {}).get(
        "summary", {}
    )
    baseline_size = results.get("B16", {}).get("phases", {}).get("size", {}).get("summary", {})
    for row in table:
        candidate = results[row["candidate"]]
        candidate_benchmark = candidate["phases"].get("benchmark", {}).get("summary", {})
        candidate_size = candidate["phases"].get("size", {}).get("summary", {})
        core = candidate_benchmark.get(
            "inference_core_logical_parameter_elements",
            candidate_size.get("inference_core_logical_parameter_elements"),
        )
        original_core = baseline_benchmark.get(
            "inference_core_logical_parameter_elements",
            baseline_size.get("inference_core_logical_parameter_elements"),
        )
        row["measured_inference_core_removed_fraction"] = (
            1 - core / original_core if core is not None and original_core else None
        )
        peak, original_peak = row["benchmark_peak_allocated_bytes"], baseline_benchmark.get(
            "peak_allocated_bytes"
        )
        row["benchmark_peak_memory_reduction_fraction"] = (
            1 - peak / original_peak if peak is not None and original_peak else None
        )
    payload = {
        "interpretation": "Action agreement is not task success; failed runs are not architecture evidence. Smoke=10, development=100, final=500 trials. Wilson intervals are approximate episode-level intervals; paired development/final intervals use a task-stratified bootstrap.",
        "coverage": read_json(STUDY / "coverage.json")["domains"],
        "candidates": results,
    }
    write_json(STUDY / "results.json", payload)
    with (STUDY / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return payload


def coverage_figure(payload):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(payload["coverage"])
    data = np.asarray(
        [
            [
                payload["coverage"][name][key]
                for key in ["screened", "rollout_tested", "development_100", "confirmed_500"]
            ]
            for name in names
        ]
    )
    fig, ax = plt.subplots(figsize=(7.6, 8.2))
    ax.imshow(data > 0, cmap="Blues", vmin=0, vmax=1, aspect="auto", alpha=0.65)
    for y in range(len(names)):
        for x in range(4):
            ax.text(
                x,
                y,
                str(data[y, x]),
                ha="center",
                va="center",
                color="white" if data[y, x] else "#555555",
            )
    ax.set(
        xticks=range(4),
        xticklabels=["Finite screens", "Rollout tested", "100-trial dev", "500-trial final"],
        yticks=range(len(names)),
        yticklabels=names,
        title="Component coverage (completed configurations)",
    )
    ax.tick_params(length=0)
    fig.tight_layout()
    folder = STUDY / "figures"
    folder.mkdir(exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(folder / f"coverage.{suffix}", dpi=180)
    plt.close(fig)


def main():
    payload = export()
    coverage_figure(payload)
    print(
        f"Exported {len(payload['candidates'])} registered candidates; actual episode outcomes and measured coverage"
    )


if __name__ == "__main__":
    main()
