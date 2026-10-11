"""Calibration, component screens and matched LIBERO trials for compression."""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluation.common import ROOT, environment, file_hash, read_json, write_json
from evaluation.compression.transforms import apply, roots, selected_linears, storage_inventory
from evaluation.models import artifact, load_policy
from evaluation.run import PROTOCOL, numpy_raw, prepare, processors, rollout

STUDY = ROOT / "studies/compression"
RUNTIME = ROOT / "outputs/compression"


def setup():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def observation_bank():
    path = ROOT / "outputs/evaluation/observations.pt"
    expected = read_json(ROOT / "studies/evaluation/observation_bank.json")["sha256"]
    if file_hash(path) != expected:
        raise ValueError("Frozen observation bank changed")
    return torch.load(path, weights_only=True)


def calibration_indices(bank):
    seen, result = defaultdict(int), []
    for index, sample in enumerate(bank):
        task = sample["task_id"]
        if seen[task] < 2:
            result.append(index)
        seen[task] += 1
    if len(result) != 20:
        raise ValueError("Expected exactly two calibration states per task")
    return result


def inputs(sample, pipeline):
    env_pre, pre, post = pipeline
    env_pre.reset()
    pre.reset()
    post.reset()
    return prepare(numpy_raw(sample["raw"]), sample["instruction"], env_pre, pre)


def calibrate():
    destination = RUNTIME / "calibration.pt"
    if destination.exists():
        raise FileExistsError("Preserve frozen calibration")
    setup()
    bank = observation_bank()
    indices = calibration_indices(bank)
    policy, metadata = load_policy("B16")
    pipeline = processors(policy)
    sums, counts, influence, token_ids, handles = {}, defaultdict(int), defaultdict(list), [], []
    from evaluation.core_benchmark import prepared_inputs

    def linear_hook(name):
        def hook(module, args):
            x = args[0].detach().reshape(-1, module.in_features).float()
            values = x.square().sum(0)
            sums[name] = sums.get(name, torch.zeros_like(values)) + values
            counts[name] += len(x)

        return hook

    def block_hook(name):
        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs["hidden_states"]
            y = output[0] if isinstance(output, tuple) else output
            influence[name].append(
                float(
                    (y.float() - x.float()).square().mean()
                    / x.float().square().mean().clamp_min(1e-8)
                )
            )

        return hook

    for full, (_, _, module) in selected_linears(policy, "all").items():
        handles.append(module.register_forward_pre_hook(linear_hook(full)))
    for scope, root in roots(policy).items():
        layers = {"text": "layers", "vision": "blocks", "action": "model.transformer_blocks"}[scope]
        for index, block in enumerate(root.get_submodule(layers)):
            handles.append(
                block.register_forward_hook(block_hook(f"{scope}.{index}"), with_kwargs=True)
            )
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index in indices:
                batch = inputs(bank[index], pipeline)
                encoded, _ = prepared_inputs(policy, batch)
                token_ids.append(encoded["input_ids"].detach().cpu().flatten())
                torch.manual_seed(PROTOCOL["seed"] + index)
                output = policy.predict_action_chunk(batch)
                if not torch.isfinite(output).all():
                    raise ValueError("Nonfinite baseline during calibration")
                print(f"calibration bank_index={index}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    calibration = {
        "linear_inputs": {k: (v / counts[k]).cpu() for k, v in sums.items()},
        "block_influence": {k: float(np.mean(v)) for k, v in influence.items()},
        "token_ids": torch.unique(torch.cat(token_ids)),
        "bank_indices": indices,
        "observation_bank_sha256": read_json(ROOT / "studies/evaluation/observation_bank.json")[
            "sha256"
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibration, destination)
    write_json(
        STUDY / "calibration.json",
        {
            "purpose": "Unlabeled calibration; no optimizer updates or robot outcomes",
            "bank_indices": indices,
            "states_per_task": 2,
            "linear_modules": len(sums),
            "calibration_sha256": file_hash(destination),
            "token_count": len(calibration["token_ids"]),
            "block_influence": calibration["block_influence"],
            "model": metadata,
            "environment": environment(),
        },
    )
    references = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, sample in enumerate(bank):
            policy.reset()
            torch.manual_seed(PROTOCOL["seed"] + index)
            references.append(policy.predict_action_chunk(inputs(sample, pipeline)).float().cpu())
    torch.save(torch.cat(references), RUNTIME / "references.pt")
    print("Calibration and 100 matched reference action chunks frozen", flush=True)


def prime_screen_shapes(policy, bank, pipeline):
    """Prime compiled projections without changing diagnostic seeds or rows."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, sample in enumerate(bank):
            policy.reset()
            torch.manual_seed(PROTOCOL["seed"] + index)
            predicted = policy.predict_action_chunk(inputs(sample, pipeline))
            if not torch.isfinite(predicted).all():
                raise ValueError(f"Nonfinite actions during shape priming at bank_index={index}")
    torch.cuda.synchronize()
    return {
        "observations": len(bank),
        "included_in_diagnostic_rows": False,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "purpose": "Compile bank input shapes before warmed screen measurements",
    }


def screen(recipe, policy, calibration, predictions=80):
    bank = observation_bank()
    references = torch.load(RUNTIME / "references.pt", weights_only=True)
    pipeline = processors(policy)
    test_indices = [i for i in range(len(bank)) if i not in calibration["bank_indices"]]
    errors, latency, records = [], [], []
    priming = (
        prime_screen_shapes(policy, bank, pipeline)
        if any(change["method"] == "compile_linears" for change in recipe["changes"])
        else None
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for iteration in range(10 + predictions):
            index = test_indices[iteration % len(test_indices)]
            policy.reset()
            torch.manual_seed(PROTOCOL["seed"] + index)
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch = inputs(bank[index], pipeline)
            predicted = policy.predict_action_chunk(batch).float()
            physical = pipeline[2](predicted.clone()).float()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - started) * 1000
            if not torch.isfinite(predicted).all():
                raise ValueError(f"Nonfinite actions at bank_index={index}")
            if iteration >= 10:
                reference = references[index : index + 1].to("cuda")
                reference = reference[:, : predicted.shape[1]]
                physical_ref = pipeline[2](reference.clone()).float()
                arm = float((physical[..., :6] - physical_ref[..., :6]).square().mean())
                grip = float((physical[..., 6] != physical_ref[..., 6]).float().mean())
                row = {
                    "bank_index": index,
                    "task_id": bank[index]["task_id"],
                    "latency_ms": ms,
                    "physical_arm_change_mse": arm,
                    "gripper_disagreement": grip,
                    "maximum_normalized_action_difference": float(
                        (predicted - reference).abs().max()
                    ),
                }
                errors.append(arm)
                latency.append(ms)
                records.append(row)
    return {
        "diagnostic_observations": predictions,
        "warmup": 10,
        "mean_physical_arm_change_mse": float(np.mean(errors)),
        "maximum_physical_arm_change_mse": float(np.max(errors)),
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
    }, records


def run(args):
    setup()
    recipe = read_json(args.recipe)
    phase = args.phase
    from evaluation.compression.protocol import verify_final_registration, verify_state_assets

    final_registration = (
        verify_final_registration(args.recipe) if phase in {"final", "benchmark"} else None
    )
    folder = STUDY / "measurements" / recipe["id"] / phase
    if folder.exists():
        raise FileExistsError(f"Preserve previous {recipe['id']}/{phase}")
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "evaluation/compression").glob("*.py")):
            archive.write(path, path.relative_to(ROOT))
        for path in [ROOT / "evaluation/models.py", ROOT / "evaluation/run.py"]:
            archive.write(path, path.relative_to(ROOT))
    record = {
        "candidate": recipe,
        "phase": phase,
        "status": "running",
        "recipe_sha256": file_hash(args.recipe),
        "environment": environment(),
        "original_protocol": PROTOCOL,
        "states_sha256": file_hash(ROOT / "studies/evaluation/initial_states.json"),
        "calibration_sha256": file_hash(RUNTIME / "calibration.pt"),
        "checkpoint_sha256": file_hash(artifact("baseline") / "model.safetensors"),
        "interpretation": "New post-training compression study; action agreement is not task success.",
        "final_registration": final_registration,
    }
    write_json(folder / "run.json", record)
    started = time.perf_counter()
    try:
        if (
            final_registration
            and record["checkpoint_sha256"] != final_registration["checkpoint_sha256"]
        ):
            raise ValueError("Baseline checkpoint changed after final selection was frozen")
        calibration = torch.load(RUNTIME / "calibration.pt", weights_only=True)
        policy, record["model"] = load_policy("B16")
        record["transformations"] = apply(policy, recipe, calibration)
        record["deployed_architecture"] = {
            "predicted_actions": policy.model.action_model.action_horizon,
            "executed_before_replanning": policy.config.n_action_steps,
            "denoising_steps": policy.model.action_model.num_inference_timesteps,
            "world_encoder_resident": policy.model.video_encoder is not None,
            "world_predictor_resident": policy.model.video_predictor is not None,
            "embodied_conditioning_tokens": policy.model.embodied_replace_prompt.count(
                policy.config.embodied_action_token
            ),
            "latent_prompt_tokens": sum(
                policy.model.replace_prompt.count(token) for token in policy.model.action_tokens
            ),
            "parameters_require_grad": any(
                parameter.requires_grad for parameter in policy.parameters()
            ),
        }
        if phase in {"development", "size", "final", "benchmark"}:
            from evaluation.compression.artifacts import tensor_fingerprint

            record["deployed_tensor_fingerprint"] = tensor_fingerprint(policy)
            if (
                final_registration
                and record["deployed_tensor_fingerprint"]["sha256"]
                != final_registration["deployed_tensor_sha256"]
            ):
                raise ValueError("Compressed tensors differ from the frozen selected model")
        gc.collect()
        torch.cuda.empty_cache()
        write_json(folder / "run.json", record)
        if phase == "screen":
            record["summary"], rows = screen(recipe, policy, calibration, args.predictions)
            write_json(folder / "observations.json", rows)
        elif phase == "benchmark":
            from evaluation.compression.artifacts import benchmark

            record["summary"] = benchmark(policy, recipe["id"], folder)
        elif phase == "size":
            from evaluation.compression.artifacts import serialize

            record["summary"] = {
                **storage_inventory(policy),
                "serialized_weights": serialize(policy, recipe["id"], folder),
            }
        else:
            record["state_asset_verification"] = verify_state_assets()
            evaluation_phase = "final" if phase == "final" else "development"
            episodes = {"smoke": 1, "development": 10, "final": 50}[phase]
            arguments = SimpleNamespace(
                variant=recipe["id"],
                checkpoint=None,
                tasks=list(range(10)),
                phase=evaluation_phase,
                episodes=episodes,
            )
            from evaluation.compression.protocol import guarded_environments

            with guarded_environments() as membership:
                record["summary"] = rollout(arguments, policy, record["model"], folder)
                record["state_membership"] = membership
        record.update(status="completed", wall_seconds=time.perf_counter() - started)
        print(
            json.dumps({"candidate": recipe["id"], "phase": phase, "summary": record["summary"]}),
            flush=True,
        )
    except BaseException as exc:
        record.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            wall_seconds=time.perf_counter() - started,
        )
        traceback.print_exc()
        raise
    finally:
        write_json(folder / "run.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("calibrate")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--recipe", type=Path, required=True)
    run_parser.add_argument(
        "--phase",
        choices=["screen", "smoke", "development", "size", "final", "benchmark"],
        default="screen",
    )
    run_parser.add_argument("--predictions", type=int, default=80)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate()
    else:
        run(args)


if __name__ == "__main__":
    main()
