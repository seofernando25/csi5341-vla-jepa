"""Parallel LIBERO rollout: N CPU env workers, one batched GPU inference server.

Same protocol, seeds, trial counts and success semantics as `evaluation.rollout`:
episodes stream to `episodes.csv` with identical columns, and each episode uses
`episode_seed(task_id, trial, phase)`. Workers own only MuJoCo envs (never CUDA);
the server holds the policy and one `Pipeline` per worker (processor state is
per-episode) plus one action queue per worker, replicating `select_action`
semantics: a batched `predict_action_chunk` runs only when a worker's queue is
empty, and the first `n_action_steps` rows are queued.

GPU guard: refuses to start when more than `GPU_USED_BYTES_LIMIT` is already
used, and backs off between batches when the card is tight. Never kills or
touches other processes.

Examples:
  python -m evaluation.parallel_rollout profile --variant B16
  python -m evaluation.parallel_rollout run --variant B16 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.common import PROTOCOL, ROOT, episode_seed
from evaluation.rollout import EPISODE_FIELDS

GPU_USED_BYTES_LIMIT = 20 * 10**9
GPU_BACKOFF_FRACTION = 0.85
RENDEZVOUS_SECONDS = 0.005


def gpu_used_bytes() -> int | None:
    """Used bytes on GPU 0 via nvidia-smi; None when the query fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.splitlines()
        return int(out[0].strip()) * 1024 * 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def check_gpu_guard(limit_bytes: int = GPU_USED_BYTES_LIMIT) -> int:
    """Raise when the shared card is too full to start; return used bytes otherwise."""
    used = gpu_used_bytes()
    if used is not None and used > limit_bytes:
        raise RuntimeError(
            f"GPU guard: {used / 1e9:.1f} GB used > {limit_bytes / 1e9:.1f} GB limit"
        )
    return used or 0


def stack_batches(batches: list[dict]) -> dict:
    """Stack N single-row prepared batches into one N-row batch."""
    stacked = {}
    for key in batches[0]:
        values = [b[key] for b in batches]
        if isinstance(values[0], torch.Tensor):
            stacked[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list):
            stacked[key] = [v for value in values for v in value]
        else:
            raise TypeError(f"Cannot batch prepared key {key!r} of type {type(values[0])}")
    return stacked


def profile_episode(policy, pipeline, env, task, seed) -> tuple[bool, int, dict[str, float]]:
    """Serial episode identical to `rollout.run_episode`, with per-stage wall times."""
    from lerobot.utils.random_utils import set_seed

    from evaluation.pipeline import inference_autocast
    from evaluation.simulator import batch_raw

    elapsed: dict[str, float] = {}

    def timed(stage, fn):
        started = time.perf_counter()
        try:
            return fn()
        finally:
            elapsed[stage] = elapsed.get(stage, 0.0) + time.perf_counter() - started

    timed("seed_reset", lambda: (set_seed(seed), policy.reset(), pipeline.reset()))
    raw, _ = timed("env_reset", lambda: env.reset(seed=seed))
    success, step = False, 0
    for step in range(1, env._max_episode_steps + 1):
        started = time.perf_counter()
        inputs = pipeline.prepare(batch_raw(raw), task["instruction"])
        elapsed["prepare"] = elapsed.get("prepare", 0.0) + time.perf_counter() - started
        with torch.inference_mode(), inference_autocast():
            started = time.perf_counter()
            action = policy.select_action(inputs)
            elapsed["inference"] = elapsed.get("inference", 0.0) + time.perf_counter() - started
        started = time.perf_counter()
        command = pipeline.post(action)
        elapsed["post"] = elapsed.get("post", 0.0) + time.perf_counter() - started
        started = time.perf_counter()
        raw, _, terminated, truncated, info = env.step(command[0].float().cpu().numpy())
        elapsed["sim_step"] = elapsed.get("sim_step", 0.0) + time.perf_counter() - started
        success = bool(info.get("is_success", False))
        if terminated or truncated or success:
            break
    return success, step, elapsed


def _worker_main(conn, task_ids: list[int], phase: str, episodes: int):
    """Own whole tasks: reset/step envs, ask the server for each action."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from lerobot.utils.random_utils import set_seed

    from evaluation.simulator import make_environment

    try:
        for task_id in task_ids:
            env, _ = make_environment(task_id, phase)
            try:
                for trial in range(episodes):
                    seed = episode_seed(task_id, trial, phase)
                    set_seed(seed)
                    raw, _ = env.reset(seed=seed)
                    conn.send(("episode_start", task_id, trial))
                    steps = 0
                    success = False
                    for steps in range(1, env._max_episode_steps + 1):
                        conn.send(("infer", raw))
                        action = conn.recv()
                        raw, _, terminated, truncated, info = env.step(np.asarray(action))
                        success = bool(info.get("is_success", False))
                        if terminated or truncated or success:
                            break
                    conn.send(("finished", task_id, trial, bool(success), steps))
            finally:
                env.close()
        conn.send(("exhausted",))
    except BaseException as exc:  # noqa: BLE001 - any worker fault must reach the server
        try:
            conn.send(("worker_error", repr(exc)))
        except OSError:
            pass
    finally:
        conn.close()


def _drain_requests(workers: dict[int, dict], block: bool):
    """Collect one pending obs per worker that has one waiting."""
    pending = []
    if block:
        while not pending and any(not w["done"] for w in workers.values()):
            for wid, w in workers.items():
                if not w["done"] and w["conn"].poll(0.01):
                    pending.append(wid)
                    break
    for wid, w in workers.items():
        if not w["done"] and wid not in pending and w["conn"].poll():
            pending.append(wid)
    return pending


def parallel_rollout(
    variant: str,
    tasks: list[int],
    phase: str,
    episodes: int,
    policy,
    make_pipeline,
    result_dir: Path,
    workers: int = 4,
    manifest: dict | None = None,
) -> dict:
    """Run matched trials across `workers` env processes with batched inference."""
    from evaluation.pipeline import inference_autocast
    from evaluation.simulator import batch_raw

    check_gpu_guard()
    n_action_steps = int(policy.config.n_action_steps)
    ctx = mp.get_context("spawn")
    assignment = [tasks[i::workers] for i in range(workers)]
    pipes, procs = {}, {}
    for wid, owned in enumerate(assignment):
        if not owned:
            continue
        parent, child = ctx.Pipe(duplex=True)
        proc = ctx.Process(target=_worker_main, args=(child, owned, phase, episodes), daemon=True)
        proc.start()
        child.close()
        pipes[wid] = parent
        procs[wid] = proc
    state = {
        wid: {"conn": conn, "done": False, "pipeline": make_pipeline(), "queue": [], "key": None}
        for wid, conn in pipes.items()
    }
    manifest = manifest or {}
    records: list[dict] = []
    result_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    handle = (result_dir / "episodes.csv").open("w", newline="")
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            alive = len(state)
            first = True
            while alive:
                for wid in _drain_requests(state, block=True):
                    w = state[wid]
                    try:
                        message = w["conn"].recv()
                    except EOFError:
                        message = ("exhausted",)
                    kind = message[0]
                    if kind == "episode_start":
                        _, task_id, trial = message
                        w["key"] = (task_id, trial)
                        w["pipeline"].reset()
                        w["queue"] = []
                        w["started"] = time.perf_counter()
                    elif kind == "infer":
                        w["pending_raw"] = message[1]
                    elif kind == "finished":
                        _, task_id, trial, success, steps = message
                        row = {
                            "variant": variant,
                            "phase": phase,
                            "task_id": task_id,
                            "task_name": manifest.get(task_id, {}).get("name", ""),
                            "trial": trial,
                            "initial_state_hash": manifest.get(task_id, {}).get(
                                f"{phase}_hashes", [""]
                            )[trial]
                            if manifest.get(task_id, {}).get(f"{phase}_hashes")
                            else "",
                            "seed": episode_seed(task_id, trial, phase),
                            "success": int(success),
                            "steps": steps,
                            "wall_seconds": time.perf_counter() - w.get("started", started_all),
                            "status": "completed",
                        }
                        writer.writerow(row)
                        handle.flush()
                        records.append(row)
                        print(
                            f"{variant} task={task_id} trial={trial} success={success}", flush=True
                        )
                        w.pop("pending_raw", None)
                    elif kind in ("exhausted", "worker_error"):
                        if kind == "worker_error":
                            raise RuntimeError(f"Env worker {wid} failed: {message[1]}")
                        w["done"] = True
                        w["conn"].close()
                        alive -= 1
                        procs[wid].join(timeout=60)
                # Batched inference for workers that asked and whose queue is empty.
                needy = [
                    wid
                    for wid, w in state.items()
                    if not w["done"] and "pending_raw" in w and not w["queue"]
                ]
                if needy:
                    if first:
                        time.sleep(RENDEZVOUS_SECONDS)  # let stragglers join the first batch
                        first = False
                    used = gpu_used_bytes()
                    if used is not None and used / (24 * 10**9) > GPU_BACKOFF_FRACTION:
                        time.sleep(5.0)
                    batches = [
                        state[wid]["pipeline"].prepare(
                            batch_raw(state[wid].pop("pending_raw")),
                            manifest.get(state[wid]["key"][0], {}).get("instruction", ""),
                        )
                        for wid in needy
                    ]
                    with torch.inference_mode(), inference_autocast():
                        chunk = policy.predict_action_chunk(stack_batches(batches))
                    if not torch.isfinite(chunk).all():
                        raise ValueError("Nonfinite actions during parallel rollout")
                    # Match select_action: queue the first n_action_steps rows per env.
                    queued = chunk.transpose(0, 1)[:n_action_steps]  # [n, N, D]
                    for slot, wid in enumerate(needy):
                        state[wid]["queue"].extend(queued[:, slot, :].unbind(0))
                # Serve every worker holding a queued action (no inference needed).
                for wid, w in state.items():
                    if w["done"] or "pending_raw" not in w or not w["queue"]:
                        continue
                    action = w["queue"].pop(0)
                    command = w["pipeline"].post(action.unsqueeze(0))
                    w["conn"].send(command[0].float().cpu().numpy())
                    del w["pending_raw"]
    finally:
        for proc in procs.values():
            if proc.is_alive():
                proc.terminate()
        for proc in procs.values():
            proc.join(timeout=30)
    per_task = [np.mean([r["success"] for r in records if r["task_id"] == t]) for t in tasks]
    return {
        "episodes": len(records),
        "successes": sum(r["success"] for r in records),
        "task_ids": tasks,
        "phase": phase,
        "task_macro_success": float(np.mean(per_task)),
        "wall_seconds": time.perf_counter() - started_all,
        "workers": workers,
    }


def compare_trials(first: list[dict], second: list[dict]) -> dict:
    """Diff two episode record lists by (task_id, trial).

    Returns missing/extra keys plus per-trial success and step diffs. Steps may
    differ under simulator nondeterminism; successes must not.
    """
    key = lambda r: (int(r["task_id"]), int(r["trial"]))
    a = {key(r): r for r in first}
    b = {key(r): r for r in second}
    success_diffs = [
        {"task_id": t, "trial": i, "first": a[k]["success"], "second": b[k]["success"]}
        for k in a.keys() & b.keys()
        for (t, i) in [k]
        if int(a[k]["success"]) != int(b[k]["success"])
    ]
    steps_diffs = [
        {"task_id": t, "trial": i, "first": a[k]["steps"], "second": b[k]["steps"]}
        for k in a.keys() & b.keys()
        for (t, i) in [k]
        if int(a[k]["steps"]) != int(b[k]["steps"])
    ]
    return {
        "first_trials": len(a),
        "second_trials": len(b),
        "missing_in_second": sorted(a.keys() - b.keys()),
        "missing_in_first": sorted(b.keys() - a.keys()),
        "success_diffs": success_diffs,
        "steps_diffs": steps_diffs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["profile", "run"])
    parser.add_argument("--variant", default="B16", choices=["B16", "Q8", "Q4"])
    parser.add_argument("--phase", default="development", choices=["development", "final"])
    parser.add_argument("--tasks", type=int, nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from evaluation.common import environment, recorded_run, run_id, write_json
    from evaluation.models import load_policy
    from evaluation.pipeline import Pipeline
    from evaluation.simulator import make_environment

    tasks = args.tasks or PROTOCOL["task_ids"]
    episodes = args.episodes if args.episodes is not None else 1
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    policy, model_meta = load_policy(args.variant)
    torch.cuda.synchronize()

    if args.command == "profile":
        from evaluation.simulator import MANIFEST, read_json

        manifest = {t["task_id"]: t for t in read_json(MANIFEST)["tasks"]}
        totals: dict[str, float] = {}
        successes = 0
        wall_start = time.perf_counter()
        for task_id in tasks:
            env, task = make_environment(task_id, args.phase)
            try:
                for trial in range(episodes):
                    seed = episode_seed(task_id, trial, args.phase)
                    ok, _, elapsed = profile_episode(policy, Pipeline(policy), env, task, seed)
                    successes += ok
                    for stage, seconds in elapsed.items():
                        totals[stage] = totals.get(stage, 0.0) + seconds
            finally:
                env.close()
        wall = time.perf_counter() - wall_start
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"profile: {successes}/{len(tasks) * episodes} successes in {wall:.1f}s")
        print(
            f"profile: peak_allocated={peak:.2f} GB, trials_per_min={len(tasks) * episodes / wall * 60:.2f}"
        )
        for stage, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {stage:10s} {seconds:8.1f}s ({100 * seconds / wall:5.1f}%)")
        return

    from evaluation.simulator import MANIFEST, read_json

    manifest = {t["task_id"]: t for t in read_json(MANIFEST)["tasks"]}
    name = run_id(f"{args.variant}-parallel{args.workers}")
    result_dir = args.out or (ROOT / "outputs/evaluation/parallel" / name)
    record = {
        "run_id": name,
        "variant": args.variant,
        "experiment": "parallel_rollout",
        "phase": args.phase,
        "environment": environment(),
        "original_protocol": PROTOCOL,
        "model": model_meta,
        "workers": args.workers,
    }
    with recorded_run(result_dir / "run.json", record):
        record["summary"] = parallel_rollout(
            args.variant,
            tasks,
            args.phase,
            episodes,
            policy,
            lambda: Pipeline(policy),
            result_dir,
            workers=args.workers,
            manifest=manifest,
        )
        write_json(result_dir / "run.json", record)
    print(f"parallel: {record['summary']}")


if __name__ == "__main__":
    main()
