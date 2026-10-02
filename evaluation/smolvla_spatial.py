"""Native SmolVLA: ten parallel episodes per LIBERO-Spatial task, no training."""
import argparse
import csv
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
import time

import gymnasium as gym
import torch

from evaluation.common import ROOT, environment, file_hash, read_json, write_json
from evaluation.run import PROTOCOL, STATES, make_environment, seed_for
from evaluation.smolvla import load_policy, processors


def worker(task_id, index, batch_size):
    from lerobot.envs.utils import FreezeAfterEpisodeEnd
    env, _ = make_environment(task_id, "final")
    env.init_state_id = index
    env._reset_stride = batch_size
    return FreezeAfterEpisodeEnd(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.batch_size not in (1, 2, 5, 10):
        parser.error("Batch size must divide the fixed ten episodes per task")
    from lerobot.envs.configs import LiberoEnv
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-SmolVLA-Spatial100-rollout"
    folder = args.resume or ROOT / "studies/evaluation/runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    env_info = environment()
    if args.resume:
        record = read_json(folder / "run.json")
        for key in ("gpu", "driver", "packages"):
            if record["environment"][key] != env_info[key]:
                raise ValueError(f"Recorded environment differs: {key}")
        for name in ("evaluation/smolvla_spatial.py", "evaluation/smolvla.py", "evaluation/smolvla_sources.json"):
            if record["environment"]["source_manifest"][name] != env_info["source_manifest"][name]:
                raise ValueError(f"Recorded inference source differs: {name}")
        if record["arguments"]["batch_size"] != args.batch_size:
            raise ValueError("Batch size differs; batched RNG protocol cannot change on resume")
        if record["status"] == "completed":
            return
    else:
        record = {"run_id": run_id, "variant": "SmolVLA", "experiment": "rollout",
                  "phase": "spatial_subset", "status": "running", "environment": env_info,
                  "protocol": PROTOCOL, "arguments": {"tasks": list(range(10)), "episodes": 10,
                                                       "batch_size": args.batch_size},
                  "initial_states_manifest_sha256": file_hash(ROOT / "studies/evaluation/initial_states.json"),
                  "evaluator_implementation_sha256": file_hash(Path(__file__)),
                  "evaluation_semantics": "First ten official pruned states per task; native LeRobot eval_policy, spawn simulator workers, batched policy, native disabled AMP, TF32 disabled. Batched flow RNG differs from single-episode evaluation. Diagnostic100, not final500.",
                  "task_results": {}}
    record["status"] = "running"
    write_json(folder / "run.json", record)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    policy, metadata = load_policy()
    if args.resume and record.get("model") != metadata:
        raise ValueError("Recorded checkpoint or native execution differs")
    record["model"] = metadata
    env_pre, pre, post = processors(policy)
    _, env_post = LiberoEnv(task="libero_spatial").get_env_processors()
    write_json(folder / "run.json", record)
    print(f"RUN={folder.relative_to(ROOT)}", flush=True)
    columns = ["variant", "phase", "task_id", "task_name", "trial", "initial_state_hash",
               "seed", "success", "sum_reward", "status"]
    # Only complete task batches are exported; interruption resumes the unfinished task.
    def export():
        temporary = folder / "episodes.csv.tmp"
        with temporary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for tid, info in sorted(record["task_results"].items(), key=lambda item: int(item[0])):
                task = STATES["tasks"][int(tid)]
                for ep in info["per_episode"]:
                    i = int(ep["episode_ix"])
                    writer.writerow({"variant": "SmolVLA", "phase": "spatial_subset", "task_id": tid,
                        "task_name": task["name"], "trial": i,
                        "initial_state_hash": task["final_hashes"][i], "seed": ep["seed"],
                        "success": int(ep["success"]), "sum_reward": ep["sum_reward"], "status": "completed"})
        temporary.replace(folder / "episodes.csv")
        write_json(folder / "run.json", record)
    export()
    try:
        for tid in range(10):
            if str(tid) in record["task_results"]:
                continue
            set_seed(seed_for(tid, 0, "final"))
            record["active_task"] = tid
            write_json(folder / "run.json", record)
            started = time.perf_counter()
            vector = gym.vector.AsyncVectorEnv(
                [partial(worker, tid, i, args.batch_size) for i in range(args.batch_size)],
                context="spawn", shared_memory=True, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
            try:
                with torch.inference_mode():
                    info = eval_policy(vector, policy, env_pre, env_post, pre, post,
                        n_episodes=10, max_episodes_rendered=0,
                        start_seed=seed_for(tid, 0, "final"))
            finally:
                vector.close()
            info["task_wall_seconds"] = time.perf_counter() - started
            record["task_results"][str(tid)] = info
            export()
            print(f"task={tid}: {info['aggregated']['n_success']}/10, {info['task_wall_seconds']:.1f}s", flush=True)
        eps = [ep for t in record["task_results"].values() for ep in t["per_episode"]]
        successes = sum(int(ep["success"]) for ep in eps)
        record.update(status="completed", active_task=None,
                      summary={"episodes": len(eps), "successes": successes,
                               "task_macro_success": successes / len(eps), "task_ids": list(range(10)),
                               "phase": "spatial_subset"})
        export()
        print(f"Completed: {successes}/{len(eps)}", flush=True)
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        export()
        raise


if __name__ == "__main__":
    main()
