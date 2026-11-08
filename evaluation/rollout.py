"""Matched closed-loop LIBERO trials: fixed initial states and per-episode seeds."""

import csv
import os
import time

import numpy as np
import torch

from evaluation.common import episode_seed
from evaluation.pipeline import inference_autocast
from evaluation.simulator import batch_raw, make_environment

EPISODE_FIELDS = [
    "variant",
    "phase",
    "task_id",
    "task_name",
    "trial",
    "initial_state_hash",
    "seed",
    "success",
    "steps",
    "wall_seconds",
    "status",
]


def run_episode(policy, pipeline, env, task, seed):
    from lerobot.utils.random_utils import set_seed

    set_seed(seed)
    policy.reset()
    pipeline.reset()
    raw, _ = env.reset(seed=seed)
    success = False
    for step in range(env._max_episode_steps):
        inputs = pipeline.prepare(batch_raw(raw), task["instruction"])
        with torch.inference_mode(), inference_autocast():
            action = pipeline.post(policy.select_action(inputs))
        raw, _, terminated, truncated, info = env.step(action[0].float().cpu().numpy())
        success = bool(info.get("is_success", False))
        if terminated or truncated or success:
            break
    return success, step + 1


def rollout(variant, tasks, phase, episodes, policy, pipeline, result_dir):
    """Run `episodes` matched trials per task; write episodes.csv as they finish."""
    records = []
    with (result_dir / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        for task_id in tasks:
            env, task = make_environment(task_id, phase)
            try:
                for index in range(episodes):
                    seed = episode_seed(task_id, index, phase)
                    started = time.perf_counter()
                    success, steps = run_episode(policy, pipeline, env, task, seed)
                    row = {
                        "variant": variant,
                        "phase": phase,
                        "task_id": task_id,
                        "task_name": task["name"],
                        "trial": index,
                        "initial_state_hash": task[f"{phase}_hashes"][index],
                        "seed": seed,
                        "success": int(success),
                        "steps": steps,
                        "wall_seconds": time.perf_counter() - started,
                        "status": "completed",
                    }
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
                    records.append(row)
                    print(f"{variant} task={task_id} trial={index} success={success}", flush=True)
            finally:
                env.close()
    per_task = [np.mean([r["success"] for r in records if r["task_id"] == t]) for t in tasks]
    return {
        "episodes": len(records),
        "successes": sum(r["success"] for r in records),
        "task_ids": tasks,
        "phase": phase,
        "task_macro_success": float(np.mean(per_task)),
    }
