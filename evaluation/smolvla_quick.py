"""Ten LIBERO-Spatial tasks in one native GPU batch: one episode per task."""
import csv
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
import time

import gymnasium as gym
import torch

from evaluation.common import ROOT, environment, file_hash, write_json
from evaluation.run import PROTOCOL, STATES
from evaluation.smolvla import load_policy, processors
from evaluation.smolvla_spatial import worker


def main():
    from lerobot.envs.configs import LiberoEnv
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-SmolVLA-Spatial10-rollout"
    folder = ROOT / "studies/evaluation/runs" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    record = {"run_id": run_id, "variant": "SmolVLA", "experiment": "rollout",
              "phase": "spatial_smoke", "status": "running", "environment": environment(),
              "protocol": PROTOCOL, "arguments": {"tasks": list(range(10)), "episodes": 1,
                                                   "batch_size": 10},
              "initial_states_manifest_sha256": file_hash(ROOT / "studies/evaluation/initial_states.json"),
              "evaluator_implementation_sha256": file_hash(Path(__file__)),
              "evaluation_semantics": "One episode per task in one ten-environment batch; first official pruned state each, episode seeds1000+task_id, native disabled AMP/FP32 buffers, TF32 disabled, no videos/training. Quick diagnostic, not final500 or isolated backbone ablation."}
    write_json(folder / "run.json", record)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(1000)
    policy, record["model"] = load_policy()
    env_pre, pre, post = processors(policy)
    _, env_post = LiberoEnv(task="libero_spatial").get_env_processors()
    write_json(folder / "run.json", record)
    print(f"RUN={folder.relative_to(ROOT)}", flush=True)
    started = time.perf_counter()
    vector = None
    try:
        vector = gym.vector.AsyncVectorEnv([partial(worker, tid, 0, 1) for tid in range(10)],
            context="spawn", shared_memory=True, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
        if len(set(vector.call("_max_episode_steps"))) != 1:
            raise ValueError("Mixed task episode limits; native batch evaluator cannot share a limit")
        with torch.inference_mode():
            info = eval_policy(vector, policy, env_pre, env_post, pre, post,
                               n_episodes=10, max_episodes_rendered=0, start_seed=1000)
        write_json(folder / "eval_info.json", info)
        columns = ["variant", "phase", "task_id", "task_name", "trial", "initial_state_hash",
                   "seed", "success", "sum_reward", "status"]
        if len(info["per_episode"]) != 10:
            raise ValueError("Quick LIBERO-Spatial evaluation did not return all ten tasks")
        with (folder / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for i, ep in enumerate(info["per_episode"]):
                if ep["episode_ix"] != i or ep["seed"] != 1000 + i:
                    raise ValueError("Batched episode order or seed differs")
                task = STATES["tasks"][i]
                writer.writerow({"variant": "SmolVLA", "phase": "spatial_smoke", "task_id": i,
                    "task_name": task["name"], "trial": 0, "initial_state_hash": task["final_hashes"][0],
                    "seed": ep["seed"], "success": int(ep["success"]),
                    "sum_reward": ep["sum_reward"], "status": "completed"})
        a = info["aggregated"]
        record.update(status="completed", summary={"episodes": 10, "successes": a["n_success"],
            "task_macro_success": a["pc_success"] / 100, "task_ids": list(range(10)),
            "phase": "spatial_smoke", "ci95_percent": a["pc_success_ci95"],
            "batch_wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        print(f"Completed: {a['n_success']}/10 across ten tasks", flush=True)
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(folder / "run.json", record)
        if vector is not None:
            vector.close()


if __name__ == "__main__":
    main()
