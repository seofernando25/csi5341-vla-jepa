"""Summarize S500 training runs and select the checkpoint by held-out loss.

Example: python -m evaluation.adaptation --training-runs <run-dir> [<run-dir> ...]
Simulator success never enters selection.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from evaluation.common import ROOT, read_json, write_json


def summarize(paths):
    """Join training segments of one continuing run into training and validation curves."""
    validation, training, recipe = defaultdict(list), {}, None
    for path in paths:
        record = read_json(path / "run.json")
        if recipe is not None and recipe != record["recipe"]:
            raise ValueError("Different training recipes cannot form one learning curve")
        recipe = record["recipe"]
        for line in (path / "metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["phase"] == "training":
                if row["step"] in training:
                    raise ValueError("Overlapping training steps; select a single continuing run")
                training[row["step"]] = row
            elif row["phase"] == "validation":
                validation[row["step"]].append(row)
    expected_samples = recipe["validation_samples_per_task"] * 10
    curve = []
    for step, batches in sorted(validation.items()):
        count = sum(r["samples"] for r in batches)
        if count != expected_samples:
            continue  # interrupted validation pass
        curve.append(
            {
                "step": step,
                "samples": count,
                **{
                    key: sum(r[key] * r["samples"] for r in batches) / count
                    for key in ["loss", "action_loss", "wm_loss"]
                },
            }
        )
    rows = [training[s] for s in sorted(training)]
    return {
        "recipe": recipe,
        "validation": curve,
        "training": rows,
        "measured_update_gpu_hours": sum(r["update_seconds"] for r in rows) / 3600,
        "peak_training_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
    }


def choose(validation, milestones):
    """Lowest held-out loss among the milestone checkpoints; ties choose fewer steps."""
    by_step = {row["step"]: row for row in validation}
    if not set(milestones) <= by_step.keys():
        raise ValueError("Required validation checkpoints are incomplete")
    candidates = [by_step[step] for step in milestones]
    if any(not math.isfinite(row["loss"]) for row in candidates):
        raise ValueError("Selection requires finite held-out loss")
    return min(candidates, key=lambda row: (row["loss"], row["step"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "studies/evaluation")
    args = parser.parse_args()
    summary = summarize([p.resolve() for p in args.training_runs])
    milestones = read_json(ROOT / "evaluation/protocol.json")["training"]["checkpoints"]
    winner = choose(summary["validation"], milestones)
    checkpoints = [
        c
        for path in args.training_runs
        for c in read_json(path / "run.json").get("checkpoints", [])
        if c["step"] == winner["step"]
    ]
    if not checkpoints:
        raise ValueError("Selected checkpoint is missing")
    write_json(args.output / "adaptation" / "summary.json", summary)
    write_json(
        args.output / "selection.json",
        {
            "rule": summary["recipe"]["selection"],
            "selected_step": winner["step"],
            "validation": winner,
            "checkpoint": checkpoints[-1],
            "eligible_steps": milestones,
            "simulator_scores_used": False,
        },
    )
    print(checkpoints[-1]["artifact"])


if __name__ == "__main__":
    main()
