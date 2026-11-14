"""Aggregate Study B per-recipe measurements into studies/compression/summary.csv.

Example: python -m evaluation.compression.analysis
Reads studies/compression/recipes/*.json and measurements/<recipe>/<phase>/.
"""

import argparse
import csv

from evaluation.analysis import paired_difference
from evaluation.common import ROOT, read_json

STUDY = ROOT / "studies/compression"

ROLLOUT_PHASES = ["smoke", "development", "final"]


def completed(candidate, phase):
    folder = STUDY / "measurements" / candidate / phase
    path = folder / "run.json"
    if not path.exists():
        return None
    record = read_json(path)
    return (record, folder) if record["status"] == "completed" else None


def episodes(folder):
    with (folder / "episodes.csv").open() as stream:
        return list(csv.DictReader(stream))


def rollout_columns(candidate, phase, baseline_rows):
    run = completed(candidate, phase)
    if run is None:
        return {}
    rows = episodes(run[1])
    columns = {
        f"{phase}_successes": sum(int(r["success"]) for r in rows),
        f"{phase}_episodes": len(rows),
    }
    if baseline_rows is not None:
        if phase == "smoke":  # one state per task: no interval, plain difference
            change = (
                100
                * (columns[f"{phase}_successes"] - sum(int(r["success"]) for r in baseline_rows))
                / len(rows)
            )
        else:
            change = paired_difference(baseline_rows, rows)["difference_pp"]
        columns[f"{phase}_paired_change_pp"] = change
    return columns


def summary_row(recipe, baselines):
    candidate = recipe["id"]
    row = {
        "candidate": candidate,
        "family": recipe["family"],
        "deployment_scope": "restricted LIBERO-Spatial vocabulary"
        if any(c["method"] == "compact_embedding" for c in recipe["changes"])
        else "full vocabulary",
    }
    screen_path = STUDY / "measurements" / candidate / "screen" / "run.json"
    if screen_path.exists():
        record = read_json(screen_path)
        screen = record.get("summary", {})
        row.update(
            screen_status=record["status"],
            screen_peak_allocated_bytes=screen.get("peak_allocated_bytes"),
            screen_pipeline_median_ms=screen.get("pipeline_median_ms"),
            screen_mean_arm_change_mse=screen.get("mean_physical_arm_change_mse"),
            resident_tensor_storage_bytes=screen.get("resident_tensor_storage_bytes"),
        )
    for phase in ROLLOUT_PHASES:
        row.update(rollout_columns(candidate, phase, baselines.get(phase)))
    size = completed(candidate, "size")
    if size:
        row["packed_archive_bytes"] = size[0]["summary"]["serialized_weights"]["bytes"]
    benchmark = completed(candidate, "benchmark")
    if benchmark:
        summary = benchmark[0]["summary"]
        row.update(
            benchmark_peak_allocated_bytes=max(
                p["allocated_bytes"] for p in summary["memory_peaks"]
            ),
            benchmark_pipeline_median_ms=summary["pipeline"]["median_ms"],
            benchmark_pipeline_p95_ms=summary["pipeline"]["p95_ms"],
            inference_core_logical_parameter_elements=summary.get(
                "inference_core_logical_parameter_elements"
            ),
        )
    return row


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    baselines = {}
    for phase in ROLLOUT_PHASES:
        run = completed("B16", phase)
        if run:
            baselines[phase] = episodes(run[1])
    recipes = [read_json(p) for p in sorted((STUDY / "recipes").glob("*.json"))]
    rows = [summary_row(recipe, baselines) for recipe in recipes]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (STUDY / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} recipes to {STUDY / 'summary.csv'}")


if __name__ == "__main__":
    main()
