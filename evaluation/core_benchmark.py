"""Time neural execution separately from tokenization and image preprocessing.

Every observation is first checked against the unmodified native policy under
identical random seeds. This benchmark supplements, not replaces, pipeline timing.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluation.common import ROOT, environment, file_hash, read_json, write_json
from evaluation.models import load_policy
from evaluation.run import PROTOCOL, numpy_raw, prepare, processors


def prepared_inputs(policy, batch):
    """Reproduce native predict_action's preprocessing, outside the timed region."""
    inputs = policy._prepare_model_inputs(batch, training=False)
    model = policy.model
    images = inputs["images"]
    if model.config.resize_images_to is not None:
        images = [
            [
                F.interpolate(image[None], size=model.config.resize_images_to, mode="area")[0]
                for image in views
            ]
            for views in images
        ]
    encoded = model.qwen.build_inputs(
        images=images,
        instructions=inputs["instructions"],
        action_prompt=model.replace_prompt,
        embodied_prompt=model.embodied_replace_prompt,
    )
    return encoded, inputs.get("state")


def neural_forward(policy, encoded, state):
    """The native decoder/token-gather/action-head path, with prepared inputs."""
    model = policy.model
    positions = (encoded["input_ids"] == model.embodied_action_token_id).nonzero(as_tuple=True)
    hidden = model._qwen_last_decoder_hidden(encoded)
    batch, _, width = hidden.shape
    conditioning = hidden[positions[0], positions[1], :].view(batch, -1, width)
    return model.action_model.predict_action(
        conditioning.float(),
        state.float() if state is not None else None,
    ).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["B16", "Q8", "Q4", "S500"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=int, default=500)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.predictions < 1 or args.repetitions < 1:
        parser.error("Prediction/repetition counts must be positive")
    if args.variant == "S500" and args.checkpoint is None:
        parser.error("S500 requires a trained checkpoint")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + f"-{args.variant}-core"
    directory = ROOT / "studies/evaluation/runs" / run_id
    directory.mkdir(parents=True)
    bank_path = ROOT / "outputs/evaluation/observations.pt"
    bank_meta = read_json(ROOT / "studies/evaluation/observation_bank.json")
    if file_hash(bank_path) != bank_meta["sha256"]:
        raise ValueError("Observation bank changed")
    bank = torch.load(bank_path, weights_only=True)
    record = {
        "run_id": run_id,
        "variant": args.variant,
        "experiment": "core_benchmark",
        "phase": "development",
        "status": "running",
        "protocol": PROTOCOL,
        "environment": environment(),
        "observation_bank_sha256": bank_meta["sha256"],
        "initial_states_manifest_sha256": file_hash(
            ROOT / "studies/evaluation/initial_states.json"
        ),
        "predictions_per_repetition": args.predictions,
        "repetitions": args.repetitions,
        "timing_scope": "VLM neural forward, token gathering, adapter if present, flow-matching head; excludes all image/token/outer processing",
    }
    if args.checkpoint:
        record["checkpoint_sha256"] = file_hash(args.checkpoint / "model.safetensors")
    write_json(directory / "run.json", record)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        policy, metadata = load_policy(args.variant, args.checkpoint)
        record["model"] = metadata
        env_pre, pre, post = processors(policy, args.checkpoint)

        def inputs(sample):
            policy.reset()
            env_pre.reset()
            pre.reset()
            post.reset()
            return prepare(numpy_raw(sample["raw"]), sample["instruction"], env_pre, pre)

        errors = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, sample in enumerate(bank):
                batch = inputs(sample)
                encoded, state = prepared_inputs(policy, batch)
                torch.manual_seed(PROTOCOL["seed"] + index)
                reference = policy.predict_action_chunk(batch)
                torch.manual_seed(PROTOCOL["seed"] + index)
                candidate = neural_forward(policy, encoded, state)
                error = float((reference - candidate).abs().max())
                if not torch.isfinite(candidate).all() or not torch.allclose(
                    reference, candidate, rtol=1e-5, atol=1e-5
                ):
                    raise ValueError(
                        f"Prepared-input execution differs from native policy at observation {index}: {error}"
                    )
                errors.append(error)
            record["equivalence"] = {
                "observations": len(errors),
                "maximum_absolute_error": max(errors),
                "rtol": 1e-5,
                "atol": 1e-5,
                "identical_flow_noise_seeds": True,
            }
            write_json(directory / "run.json", record)
            measured = []
            with (directory / "timings.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "variant",
                        "mode",
                        "repetition",
                        "prediction",
                        "task_id",
                        "bank_index",
                        "latency_ms",
                    ],
                )
                writer.writeheader()
                for repetition in range(args.repetitions):
                    torch.manual_seed(PROTOCOL["seed"] + repetition)
                    for index in range(PROTOCOL["timing"]["warmup"] + args.predictions):
                        bank_index = index % len(bank)
                        sample = bank[bank_index]
                        encoded, state = prepared_inputs(policy, inputs(sample))
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        action = neural_forward(policy, encoded, state)
                        torch.cuda.synchronize()
                        elapsed = 1000 * (time.perf_counter() - started)
                        if not torch.isfinite(action).all():
                            raise ValueError("Nonfinite prediction")
                        if index >= PROTOCOL["timing"]["warmup"]:
                            writer.writerow(
                                {
                                    "variant": args.variant,
                                    "mode": "neural",
                                    "repetition": repetition,
                                    "prediction": index - PROTOCOL["timing"]["warmup"],
                                    "task_id": sample["task_id"],
                                    "bank_index": bank_index,
                                    "latency_ms": elapsed,
                                }
                            )
                            measured.append(elapsed)
                    handle.flush()
                    print(f"{args.variant} neural repetition {repetition} complete", flush=True)
        record.update(
            status="completed",
            summary={
                "median_ms": float(np.median(measured)),
                "p95_ms": float(np.percentile(measured, 95)),
                "predictions": len(measured),
            },
        )
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(directory / "run.json", record)
    print(directory.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
