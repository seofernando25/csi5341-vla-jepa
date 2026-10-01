"""Export adaptation curves from recorded optimizer/validation measurements."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.analysis import controls, load_run, success_summary
from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.figures import COLORS, save, style


def summarize(paths):
    validation, training, evidence = defaultdict(list), {}, []
    recipe = None
    parent_checkpoints = set()
    checkpoint_steps = {}
    split_hash = None
    for path in paths:
        record = read_json(path / "run.json")
        if not record.get("fixed_schedule"):
            raise ValueError(
                "Engineering preflight without fixed schedule cannot enter learning curves"
            )
        if recipe is not None and recipe != record["recipe"]:
            raise ValueError("Different training recipes cannot form one learning curve")
        recipe = record["recipe"]
        if split_hash is not None and split_hash != record["training_split_sha256"]:
            raise ValueError("Different datasets/splits cannot form one learning curve")
        split_hash = record["training_split_sha256"]
        if evidence and record.get("resume_checkpoint_sha256") not in parent_checkpoints:
            raise ValueError("Training segments must resume a checkpoint from the selected chain")
        parent_checkpoints.update(c["sha256"] for c in record.get("checkpoints", []))
        checkpoint_steps.update({c["sha256"]: c["step"] for c in record.get("checkpoints", [])})
        metrics = path / "metrics.jsonl"
        for line in metrics.read_text().splitlines():
            row = json.loads(line)
            if row["phase"] == "training":
                if row["step"] in training:
                    raise ValueError("Overlapping training steps; select a single continuing run")
                training[row["step"]] = row
            elif row["phase"] == "validation":
                validation[row["step"]].append(row)
        evidence.append(
            {
                "run": str(path.relative_to(ROOT)),
                "status": record["status"],
                "metrics_sha256": file_hash(metrics),
                "metadata_sha256": file_hash(path / "run.json"),
            }
        )
    curve = []
    expected_samples = recipe["validation_samples_per_task"] * 10
    for step, batches in sorted(validation.items()):
        count = sum(r["samples"] for r in batches)
        if count != expected_samples or len({r["batch"] for r in batches}) != len(batches):
            # A still-running validation is not a complete checkpoint estimate.
            continue
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
    return {
        "validation": curve,
        "training": [training[s] for s in sorted(training)],
        "recipe": recipe,
        "evidence": evidence,
        "checkpoint_steps": checkpoint_steps,
    }


def render(summary, directory):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    curve = summary["validation"]
    if curve:
        axes[0].plot(
            [r["step"] for r in curve],
            [r["loss"] for r in curve],
            "o-",
            color=COLORS["S500"],
            linewidth=1.3,
        )
    else:
        axes[0].text(
            0.5,
            0.5,
            "Validation measurement pending",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    development = summary.get("development", [])
    if development:
        axes[1].errorbar(
            [r["step"] for r in development],
            [100 * r["rate"] for r in development],
            yerr=[
                [100 * (r["rate"] - r["ci95"][0]) for r in development],
                [100 * (r["ci95"][1] - r["rate"]) for r in development],
            ],
            fmt="o-",
            capsize=3,
            color=COLORS["S500"],
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "Development rollouts pending",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[0].set(title="a   Held-out prediction loss", ylabel="Action + weighted world-model loss")
    axes[1].set(title="b   Development manipulation", ylabel="Task success (%)", ylim=(-2, 102))
    for ax in axes:
        ax.set_xlabel("Optimizer steps")
        ax.grid(alpha=0.6)
        ax.set_axisbelow(True)
        if summary["training"]:
            ax.set_xlim(0, max(r["step"] for r in summary["training"]) * 1.08)
    fig.suptitle("SmolVLM2-500M adaptation · seed 42 · measurements to date", fontsize=10)
    save(fig, directory, "F5_adaptation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--development-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "studies/evaluation/adaptation")
    args = parser.parse_args()
    summary = summarize([p.resolve() for p in args.runs])
    development = []
    reference = None
    for path in args.development_runs:
        record, rows, evidence = load_run(path.resolve())
        if (
            record["phase"] != "development"
            or record["variant"] != "S500"
            or record["experiment"] != "rollout"
        ):
            raise ValueError("Adaptation curves require S500 development rollouts")
        current = controls(record)
        if reference is not None and reference != current:
            raise ValueError("Development rollout controls differ")
        reference = current
        checkpoint = record["checkpoint_sha256"]
        if checkpoint not in summary["checkpoint_steps"]:
            raise ValueError("Development checkpoint is not in the selected training chain")
        result = success_summary(record, rows)
        result.update(step=summary["checkpoint_steps"][checkpoint], evidence=evidence)
        development.append(result)
    if len({r["step"] for r in development}) != len(development):
        raise ValueError("Duplicate development checkpoint measurements")
    summary["development"] = sorted(development, key=lambda r: r["step"])
    summary["measured_update_gpu_hours"] = (
        sum(r["update_seconds"] for r in summary["training"]) / 3600
    )
    summary["peak_training_allocated_bytes"] = max(
        r["peak_allocated_bytes"] for r in summary["training"]
    )
    write_json(args.output / "summary.json", summary)
    render(summary, args.output / "figures")
    print(
        json.dumps(
            {"steps": len(summary["training"]), "validation_points": len(summary["validation"])}
        )
    )


if __name__ == "__main__":
    main()
