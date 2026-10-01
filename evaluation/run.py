"""Record matched LIBERO rollouts and fixed-input policy benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from evaluation.common import ROOT, environment, file_hash, read_json, write_json
from evaluation.libero_setup import configure
from evaluation.models import SOURCES, artifact, load_policy

PROTOCOL = read_json(ROOT / "evaluation/protocol.json")
STATES = read_json(ROOT / "studies/evaluation/initial_states.json")


def setup_simulator():
    root = configure()
    import libero.libero

    source = SOURCES["simulator_assets"]
    assets = snapshot_download(
        source["repo"], repo_type="dataset", revision=source["revision"], local_files_only=True
    )
    libero.libero._assets_path_cache = assets
    return root


def make_environment(task_id, phase):
    root = setup_simulator()
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    env = LiberoEnv(
        suite,
        task_id,
        "libero_spatial",
        obs_type="pixels_agent_pos",
        observation_height=PROTOCOL["observation_height"],
        observation_width=PROTOCOL["observation_width"],
    )
    task = STATES["tasks"][task_id]
    states = np.asarray(torch.load(root / task[f"{phase}_file"], weights_only=False))
    env._init_states = states[task[f"{phase}_indices"]]
    return env, task


def batch_raw(raw):
    if isinstance(raw, dict):
        return {k: batch_raw(v) for k, v in raw.items()}
    return np.expand_dims(raw, 0)


def tensor_raw(raw):
    if isinstance(raw, dict):
        return {k: tensor_raw(v) for k, v in raw.items()}
    return torch.from_numpy(np.asarray(raw).copy())


def numpy_raw(raw):
    if isinstance(raw, dict):
        return {k: numpy_raw(v) for k, v in raw.items()}
    return raw.numpy()


def seed_for(task_id, index, phase):
    return (
        PROTOCOL["seed"]
        + task_id * 1000
        + index
        + (PROTOCOL["final_seed_offset"] if phase == "final" else 0)
    )


def processors(policy, checkpoint=None):
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies import make_pre_post_processors

    env_pre, _ = LiberoEnv(task="libero_spatial").get_env_processors()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint or artifact("baseline"),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    return env_pre, pre, post


def prepare(raw, instruction, env_pre, pre):
    from lerobot.envs import preprocess_observation

    observation = preprocess_observation(raw)
    observation["task"] = [instruction]
    return pre(env_pre(observation))


def collect(args):
    destination = ROOT / "outputs/evaluation/observations.pt"
    if destination.exists():
        raise FileExistsError("Observation bank already exists; preserve it for matched timing")
    samples = []
    for task_id in PROTOCOL["task_ids"]:
        env, task = make_environment(task_id, "development")
        try:
            for index in range(10):
                raw, _ = env.reset(seed=seed_for(task_id, index, "development"))
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, destination)
    write_json(
        ROOT / "studies/evaluation/observation_bank.json",
        {
            "sha256": file_hash(destination),
            "samples": len(samples),
            "source": "LIBERO-Spatial development initial observations after standard settling",
            "states_manifest_sha256": file_hash(ROOT / "studies/evaluation/initial_states.json"),
        },
    )
    print(f"Collected {len(samples)} simulator observations", flush=True)


def rollout(args, policy, metadata, result_dir):
    from lerobot.utils.random_utils import set_seed

    env_pre, pre, post = processors(policy, args.checkpoint)
    records = []
    columns = [
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
    with (result_dir / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for task_id in args.tasks:
            env, task = make_environment(task_id, args.phase)
            try:
                for index in range(args.episodes):
                    seed = seed_for(task_id, index, args.phase)
                    set_seed(seed)
                    policy.reset()
                    env_pre.reset()
                    pre.reset()
                    post.reset()
                    started = time.perf_counter()
                    raw, _ = env.reset(seed=seed)
                    success = False
                    for step in range(env._max_episode_steps):
                        inputs = prepare(batch_raw(raw), task["instruction"], env_pre, pre)
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            action = post(policy.select_action(inputs))
                        raw, _, terminated, truncated, info = env.step(
                            action[0].float().cpu().numpy()
                        )
                        success = bool(info.get("is_success", False))
                        if terminated or truncated or success:
                            break
                    row = {
                        "variant": args.variant,
                        "phase": args.phase,
                        "task_id": task_id,
                        "task_name": task["name"],
                        "trial": index,
                        "initial_state_hash": task[f"{args.phase}_hashes"][index],
                        "seed": seed,
                        "success": int(success),
                        "steps": step + 1,
                        "wall_seconds": time.perf_counter() - started,
                        "status": "completed",
                    }
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
                    records.append(row)
                    print(
                        f"{args.variant} task={task_id} trial={index} success={success} steps={step + 1}",
                        flush=True,
                    )
            finally:
                env.close()
    return {
        "episodes": len(records),
        "successes": sum(r["success"] for r in records),
        "task_ids": args.tasks,
        "phase": args.phase,
        "task_macro_success": float(
            np.mean(
                [np.mean([r["success"] for r in records if r["task_id"] == t]) for t in args.tasks]
            )
        ),
    }


def benchmark(args, policy, metadata, result_dir):
    bank_path = ROOT / "outputs/evaluation/observations.pt"
    bank_meta = read_json(ROOT / "studies/evaluation/observation_bank.json")
    if file_hash(bank_path) != bank_meta["sha256"]:
        raise ValueError("Observation bank changed")
    bank = torch.load(bank_path, weights_only=True)
    env_pre, pre, post = processors(policy, args.checkpoint)
    records = []
    peaks = []
    with (result_dir / "timings.csv").open("w", newline="") as handle:
        fields = [
            "variant",
            "mode",
            "repetition",
            "prediction",
            "task_id",
            "bank_index",
            "latency_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in ["policy", "pipeline"]:
            for repetition in range(args.repetitions):
                torch.manual_seed(PROTOCOL["seed"] + repetition)
                torch.cuda.empty_cache()
                for index in range(PROTOCOL["timing"]["warmup"] + args.predictions):
                    sample = bank[index % len(bank)]
                    raw = numpy_raw(sample["raw"])
                    policy.reset()
                    env_pre.reset()
                    pre.reset()
                    post.reset()
                    inputs = (
                        prepare(raw, sample["instruction"], env_pre, pre)
                        if mode == "policy"
                        else None
                    )
                    if index == PROTOCOL["timing"]["warmup"]:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        if mode == "pipeline":
                            inputs = prepare(raw, sample["instruction"], env_pre, pre)
                        action = policy.predict_action_chunk(inputs)
                        if mode == "pipeline":
                            action = post(action).cpu()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - started) * 1000
                    if not torch.isfinite(action).all():
                        raise ValueError("Nonfinite prediction during timing")
                    if index >= PROTOCOL["timing"]["warmup"]:
                        row = {
                            "variant": args.variant,
                            "mode": mode,
                            "repetition": repetition,
                            "prediction": index - PROTOCOL["timing"]["warmup"],
                            "task_id": sample["task_id"],
                            "bank_index": index % len(bank),
                            "latency_ms": elapsed,
                        }
                        writer.writerow(row)
                        records.append(row)
                handle.flush()
                peaks.append(
                    {
                        "mode": mode,
                        "repetition": repetition,
                        "allocated_bytes": torch.cuda.max_memory_allocated(),
                        "reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                )
                print(f"{args.variant} {mode} repetition {repetition} complete", flush=True)
    summary = {
        "observation_bank_sha256": bank_meta["sha256"],
        "predictions_per_repetition": args.predictions,
        "repetitions": args.repetitions,
        "memory_peaks": peaks,
    }
    for mode in ["policy", "pipeline"]:
        values = [r["latency_ms"] for r in records if r["mode"] == mode]
        summary[mode] = {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }
    # Measure the same safetensors encoding for every variant, including packing
    # metadata. Serialization happens after timing and is not counted as inference.
    from safetensors.torch import save_model

    weights_dir = ROOT / "outputs/evaluation/serialized" / result_dir.name
    weights_dir.mkdir(parents=True, exist_ok=False)
    weights = weights_dir / "model.safetensors"
    save_model(policy, str(weights))
    summary["serialized_weights"] = {
        "bytes": weights.stat().st_size,
        "sha256": file_hash(weights),
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
    parser.add_argument("--label", help="Distinct result label for a supplied architecture")
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
        return
    maximum = 10 if args.phase == "development" else 50
    if not 1 <= args.episodes <= maximum or args.predictions < 1 or args.repetitions < 1:
        parser.error("Invalid trial/prediction count")
    if len(set(args.tasks)) != len(args.tasks) or not set(args.tasks) <= set(PROTOCOL["task_ids"]):
        parser.error("Tasks must be unique IDs from the frozen suite")
    if args.variant == "S500" and args.checkpoint is None:
        parser.error("S500 task/efficiency evaluation requires a trained checkpoint")
    if args.label:
        import re
        if args.variant != 'S500' or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', args.label):
            parser.error('Custom labels require a SmolVLM checkpoint and a safe label')
    loader_variant = args.variant
    args.variant = args.label or args.variant
    run_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + args.variant + "-" + args.command
    )
    result_dir = ROOT / "studies/evaluation/runs" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "run_id": run_id,
        "variant": args.variant,
        "experiment": args.command,
        "phase": args.phase,
        "environment": environment(),
        "protocol": PROTOCOL,
        "status": "running",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "initial_states_manifest_sha256": file_hash(
            ROOT / "studies/evaluation/initial_states.json"
        ),
    }
    if args.checkpoint:
        record["checkpoint_sha256"] = file_hash(args.checkpoint / "model.safetensors")
    write_json(result_dir / "run.json", record)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        load_started = time.perf_counter()
        policy, metadata = load_policy(loader_variant, args.checkpoint)
        metadata["loader_variant"] = loader_variant
        metadata["variant"] = args.variant
        torch.cuda.synchronize()
        record["load_seconds"] = time.perf_counter() - load_started
        record["model"] = metadata
        write_json(result_dir / "run.json", record)
        summary = (
            rollout(args, policy, metadata, result_dir)
            if args.command == "rollout"
            else benchmark(args, policy, metadata, result_dir)
        )
        record.update(status="completed", summary=summary)
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(result_dir / "run.json", record)
        raise
    write_json(result_dir / "run.json", record)
    print(json.dumps({"run_id": run_id, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
