"""CLI for Study A observation collection, serial LIBERO rollouts, and full-pipeline timing.

Examples:
  python -m evaluation.run collect
  python -m evaluation.run rollout --variant Q4 --phase final --episodes 50
  python -m evaluation.run benchmark --variant Q4
Each run writes studies/evaluation/runs/<run-id>/run.json plus episodes.csv or timings.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from evaluation.common import (
    PROTOCOL,
    ROOT,
    environment,
    episode_seed,
    recorded_run,
    run_id,
    write_json,
)
from evaluation.models import load_policy
from evaluation.pipeline import BANK, Pipeline, observation_bank
from evaluation.rollout import rollout
from evaluation.simulator import batch_raw, make_environment, tensor_raw
from evaluation.timing import time_predictions


def collect():
    """Save the first settled observation of 10 development states per task."""
    if BANK.exists():
        raise FileExistsError("Observation bank already exists; preserve it for matched timing")
    samples = []
    for task_id in PROTOCOL["task_ids"]:
        env, task = make_environment(task_id, "development")
        try:
            for index in range(10):
                raw, _ = env.reset(seed=episode_seed(task_id, index, "development"))
                samples.append(
                    {
                        "raw": tensor_raw(batch_raw(raw)),
                        "instruction": task["instruction"],
                        "task_id": task_id,
                        "state_hash": task["development_hashes"][index],
                    }
                )
        finally:
            env.close()
    BANK.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, BANK)
    print(f"Collected {len(samples)} simulator observations", flush=True)


def benchmark(args, policy, pipeline, result_dir):
    summary = time_predictions(
        policy,
        pipeline,
        observation_bank(),
        args.variant,
        result_dir / "timings.csv",
        args.predictions,
        args.repetitions,
    )
    from safetensors.torch import save_model

    weights_dir = ROOT / "outputs/evaluation/serialized" / result_dir.name
    weights_dir.mkdir(parents=True, exist_ok=False)
    weights = weights_dir / "model.safetensors"
    save_model(policy, str(weights))
    summary["serialized_weights"] = {
        "bytes": weights.stat().st_size,
        "format": "safetensors state_dict; shared tensors deduplicated; packing metadata included",
        "artifact": str(weights.relative_to(ROOT)),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "rollout", "benchmark"])
    parser.add_argument("--variant", choices=["B16", "Q8", "Q4", "S500"], default="B16")
    parser.add_argument("--phase", choices=["development", "final"], default="development")
    parser.add_argument("--tasks", type=int, nargs="+", default=PROTOCOL["task_ids"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--predictions", type=int, default=500)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "collect":
        collect()
        return
    maximum = 10 if args.phase == "development" else 50
    if not 1 <= args.episodes <= maximum or args.predictions < 1 or args.repetitions < 1:
        parser.error("Invalid trial/prediction count")
    if len(set(args.tasks)) != len(args.tasks) or not set(args.tasks) <= set(PROTOCOL["task_ids"]):
        parser.error("Tasks must be unique IDs from the frozen suite")
    if args.variant == "S500" and args.checkpoint is None:
        parser.error("S500 task/efficiency evaluation requires a trained checkpoint")
    name = run_id(f"{args.variant}-{args.command}")
    result_dir = ROOT / "studies/evaluation/runs" / name
    result_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "run_id": name,
        "variant": args.variant,
        "experiment": args.command,
        "phase": args.phase,
        "environment": environment(),
        "protocol": PROTOCOL,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with recorded_run(result_dir / "run.json", record):
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        load_started = time.perf_counter()
        policy, record["model"] = load_policy(args.variant, args.checkpoint)
        torch.cuda.synchronize()
        record["load_seconds"] = time.perf_counter() - load_started
        write_json(result_dir / "run.json", record)
        pipeline = Pipeline(policy, args.checkpoint)
        if args.command == "rollout":
            record["summary"] = rollout(
                args.variant, args.tasks, args.phase, args.episodes, policy, pipeline, result_dir
            )
        else:
            record["summary"] = benchmark(args, policy, pipeline, result_dir)
    print(json.dumps({"run_id": name, "summary": record["summary"]}), flush=True)


if __name__ == "__main__":
    main()
