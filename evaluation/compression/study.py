"""Measure one compression recipe in one phase.

Examples:
  python -m evaluation.compression.study calibrate
  python -m evaluation.compression.study run --recipe studies/compression/recipes/deploy-hqq2-vision-action.json --phase smoke

Phases: screen (action change on 80 fixed frames), smoke/development/final (1/10/50 rollouts
per task), size (packed archive bytes), benchmark (latency and memory). Results go to
studies/compression/measurements/<recipe>/<phase>/.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.common import PROTOCOL, environment, read_json, recorded_run, write_json
from evaluation.compression import calibration as calib
from evaluation.compression.inventory import serialize, storage_inventory
from evaluation.compression.recipes import apply
from evaluation.models import load_policy
from evaluation.pipeline import Pipeline, inference_autocast, observation_bank
from evaluation.rollout import rollout
from evaluation.timing import prime_shapes, time_predictions

ROLLOUT_EPISODES = {"smoke": 1, "development": 10, "final": 50}
SCREEN_WARMUP = 10


def screen(recipe, policy, calibration, predictions=80):
    """Latency, memory and action change against B16 on bank frames outside calibration."""
    bank = observation_bank()
    references = torch.load(calib.REFERENCES, weights_only=True)
    pipeline = Pipeline(policy)
    test_indices = [i for i in range(len(bank)) if i not in calibration["bank_indices"]]
    compiled = any(change["method"] == "compile_linears" for change in recipe["changes"])
    priming = prime_shapes(policy, pipeline, bank) if compiled else None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    records = []
    with torch.inference_mode(), inference_autocast():
        for iteration in range(SCREEN_WARMUP + predictions):
            index = test_indices[iteration % len(test_indices)]
            policy.reset()
            torch.manual_seed(PROTOCOL["seed"] + index)
            torch.cuda.synchronize()
            started = time.perf_counter()
            predicted = policy.predict_action_chunk(pipeline.sample(bank[index])).float()
            physical = pipeline.post(predicted.clone()).float()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - started) * 1000
            if not torch.isfinite(predicted).all():
                raise ValueError(f"Nonfinite actions at bank_index={index}")
            if iteration < SCREEN_WARMUP:
                continue
            reference = references[index : index + 1].to("cuda")[:, : predicted.shape[1]]
            physical_ref = pipeline.post(reference.clone()).float()
            records.append(
                {
                    "bank_index": index,
                    "task_id": bank[index]["task_id"],
                    "latency_ms": ms,
                    "physical_arm_change_mse": float(
                        (physical[..., :6] - physical_ref[..., :6]).square().mean()
                    ),
                    "gripper_disagreement": float(
                        (physical[..., 6] != physical_ref[..., 6]).float().mean()
                    ),
                    "maximum_normalized_action_difference": float(
                        (predicted - reference).abs().max()
                    ),
                }
            )
    arm = [r["physical_arm_change_mse"] for r in records]
    latency = [r["latency_ms"] for r in records]
    summary = {
        "diagnostic_observations": predictions,
        "warmup": SCREEN_WARMUP,
        "mean_physical_arm_change_mse": float(np.mean(arm)),
        "maximum_physical_arm_change_mse": float(np.max(arm)),
        "mean_gripper_disagreement": float(np.mean([r["gripper_disagreement"] for r in records])),
        "maximum_normalized_action_difference": max(
            r["maximum_normalized_action_difference"] for r in records
        ),
        "pipeline_median_ms": float(np.median(latency)),
        "pipeline_p95_ms": float(np.percentile(latency, 95)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "predicted_actions": policy.model.action_model.action_horizon,
        "executed_before_replanning": policy.config.n_action_steps,
        **({"backend_shape_priming": priming} if priming is not None else {}),
        **storage_inventory(policy),
    }
    return summary, records


def benchmark(policy, recipe_id, folder):
    bank = observation_bank()
    pipeline = Pipeline(policy)
    priming = prime_shapes(policy, pipeline, bank)
    summary = time_predictions(
        policy,
        pipeline,
        bank,
        recipe_id,
        folder / "timings.csv",
        predictions=500,
        repetitions=3,
    )
    return {
        **summary,
        "backend_shape_priming": priming,
        **storage_inventory(policy),
        "predicted_actions": policy.model.action_model.action_horizon,
        "executed_before_replanning": policy.config.n_action_steps,
        "serialized_weights": serialize(policy, recipe_id, folder),
    }


def deployed_architecture(policy):
    model = policy.model
    return {
        "predicted_actions": model.action_model.action_horizon,
        "executed_before_replanning": policy.config.n_action_steps,
        "denoising_steps": model.action_model.num_inference_timesteps,
        "world_encoder_resident": model.video_encoder is not None,
        "world_predictor_resident": model.video_predictor is not None,
        "embodied_conditioning_tokens": model.embodied_replace_prompt.count(
            policy.config.embodied_action_token
        ),
        "latent_prompt_tokens": sum(model.replace_prompt.count(t) for t in model.action_tokens),
    }


def run(recipe_path, phase, predictions):
    calib.setup()
    recipe = read_json(recipe_path)
    folder = calib.STUDY / "measurements" / recipe["id"] / phase
    if folder.exists():
        raise FileExistsError(f"Preserve previous {recipe['id']}/{phase}")
    folder.mkdir(parents=True)
    record = {
        "candidate": recipe,
        "phase": phase,
        "environment": environment(),
        "original_protocol": PROTOCOL,
    }
    with recorded_run(folder / "run.json", record):
        calibration = calib.load()
        policy, record["model"] = load_policy("B16")
        record["transformations"] = apply(policy, recipe, calibration)
        record["deployed_architecture"] = deployed_architecture(policy)
        gc.collect()
        torch.cuda.empty_cache()
        write_json(folder / "run.json", record)
        if phase == "screen":
            record["summary"], rows = screen(recipe, policy, calibration, predictions)
            write_json(folder / "observations.json", rows)
        elif phase == "benchmark":
            record["summary"] = benchmark(policy, recipe["id"], folder)
        elif phase == "size":
            record["summary"] = {
                **storage_inventory(policy),
                "serialized_weights": serialize(policy, recipe["id"], folder),
            }
        else:
            record["summary"] = rollout(
                recipe["id"],
                list(range(10)),
                "final" if phase == "final" else "development",
                ROLLOUT_EPISODES[phase],
                policy,
                Pipeline(policy),
                folder,
            )
    print(json.dumps({"candidate": recipe["id"], "phase": phase, "summary": record["summary"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("calibrate")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--recipe", type=Path, required=True)
    run_parser.add_argument(
        "--phase", choices=["screen", *ROLLOUT_EPISODES, "size", "benchmark"], default="screen"
    )
    run_parser.add_argument("--predictions", type=int, default=80)
    args = parser.parse_args()
    if args.command == "calibrate":
        calib.calibrate()
    else:
        run(args.recipe, args.phase, args.predictions)


if __name__ == "__main__":
    main()
