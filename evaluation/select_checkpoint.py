"""Select S500 using the frozen held-out-loss rule, never simulator success."""

import argparse
import math
from datetime import datetime
from pathlib import Path

from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.training_analysis import summarize


def choose(validation, milestones):
    by_step = {row["step"]: row for row in validation}
    if len(by_step) != len(validation):
        raise ValueError("Duplicate validation checkpoints")
    if not set(milestones) <= by_step.keys():
        raise ValueError("Required validation checkpoints are incomplete")
    candidates = [by_step[step] for step in milestones]
    if any(not math.isfinite(row["loss"]) or row["samples"] != 200 for row in candidates):
        raise ValueError("Selection requires finite loss on all 200 frozen validation samples")
    return min(candidates, key=lambda row: (row["loss"], row["step"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--budget", type=Path, default=ROOT / "studies/evaluation/budget.json")
    parser.add_argument("--output", type=Path, default=ROOT / "studies/evaluation/selection.json")
    args = parser.parse_args()
    budget = read_json(args.budget)
    milestones = budget["checkpoints"]
    planned = read_json(ROOT / "evaluation/protocol.json")["training"]["checkpoints"]
    if not milestones or milestones != planned[: len(milestones)]:
        raise ValueError("Budget must preserve the preregistered checkpoint order")
    if budget["maximum_optimizer_steps"] != milestones[-1]:
        raise ValueError("Budget cap differs from its final checkpoint")
    frozen_at = datetime.fromisoformat(budget["frozen_at"])
    records = [read_json(path / "run.json") for path in args.training_runs]
    for record in records:
        if record["status"] != "completed":
            raise ValueError("Selection requires completed training segments")
        if (
            record["requested_steps"] > 500
            and datetime.fromisoformat(record["environment"]["recorded_at"]) < frozen_at
        ):
            raise ValueError("Adaptation budget was frozen after extended training began")
    summary = summarize([p.resolve() for p in args.training_runs])
    winner = choose(summary["validation"], milestones)
    checkpoints = [
        c for r in records for c in r.get("checkpoints", []) if c["step"] == winner["step"]
    ]
    if not checkpoints or len({c["sha256"] for c in checkpoints}) != 1:
        raise ValueError("Selected checkpoint is missing or ambiguous")
    checkpoint = checkpoints[-1]
    if file_hash(ROOT / checkpoint["artifact"] / "model.safetensors") != checkpoint["sha256"]:
        raise ValueError("Selected checkpoint file changed")
    result = {
        "rule": summary["recipe"]["selection"],
        "selected_step": winner["step"],
        "validation": winner,
        "checkpoint": checkpoint,
        "eligible_steps": milestones,
        "budget_sha256": file_hash(args.budget),
        "training_evidence": summary["evidence"],
        "simulator_scores_used": False,
    }
    if args.output.exists() and read_json(args.output) != result:
        raise ValueError("An existing final checkpoint selection cannot be silently replaced")
    write_json(args.output, result)
    print(checkpoint["artifact"])


if __name__ == "__main__":
    main()
