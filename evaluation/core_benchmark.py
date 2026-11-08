"""Time neural execution separately from tokenization and image preprocessing.

Every observation is first checked against the unmodified native policy under
identical random seeds. This benchmark supplements, not replaces, pipeline timing.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluation.common import PROTOCOL, ROOT, environment, recorded_run, run_id, write_json
from evaluation.models import load_policy
from evaluation.pipeline import Pipeline, inference_autocast, observation_bank
from evaluation.timing import FIELDS


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


def check_equivalence(policy, pipeline, bank):
    """Prepared-input execution must reproduce the native policy on every bank observation."""
    errors = []
    for index, sample in enumerate(bank):
        policy.reset()
        batch = pipeline.sample(sample)
        encoded, state = prepared_inputs(policy, batch)
        torch.manual_seed(PROTOCOL["seed"] + index)
        reference = policy.predict_action_chunk(batch)
        torch.manual_seed(PROTOCOL["seed"] + index)
        candidate = neural_forward(policy, encoded, state)
        error = float((reference - candidate).abs().max())
        if not torch.isfinite(candidate).all() or not torch.allclose(
            reference, candidate, rtol=1e-5, atol=1e-5
        ):
            raise ValueError(f"Prepared-input execution differs at observation {index}: {error}")
        errors.append(error)
    return {
        "observations": len(errors),
        "maximum_absolute_error": max(errors),
        "rtol": 1e-5,
        "atol": 1e-5,
        "identical_flow_noise_seeds": True,
    }


def time_neural(policy, pipeline, bank, variant, csv_path, predictions, repetitions):
    warmup = PROTOCOL["timing"]["warmup"]
    measured = []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for repetition in range(repetitions):
            torch.manual_seed(PROTOCOL["seed"] + repetition)
            for iteration in range(warmup + predictions):
                index = iteration % len(bank)
                policy.reset()
                encoded, state = prepared_inputs(policy, pipeline.sample(bank[index]))
                torch.cuda.synchronize()
                started = time.perf_counter()
                action = neural_forward(policy, encoded, state)
                torch.cuda.synchronize()
                elapsed = 1000 * (time.perf_counter() - started)
                if not torch.isfinite(action).all():
                    raise ValueError("Nonfinite prediction")
                if iteration >= warmup:
                    writer.writerow(
                        {
                            "variant": variant,
                            "mode": "neural",
                            "repetition": repetition,
                            "prediction": iteration - warmup,
                            "task_id": bank[index]["task_id"],
                            "bank_index": index,
                            "latency_ms": elapsed,
                        }
                    )
                    measured.append(elapsed)
            handle.flush()
            print(f"{variant} neural repetition {repetition} complete", flush=True)
    return {
        "median_ms": float(np.median(measured)),
        "p95_ms": float(np.percentile(measured, 95)),
        "predictions": len(measured),
    }


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
    name = run_id(f"{args.variant}-core")
    directory = ROOT / "studies/evaluation/runs" / name
    directory.mkdir(parents=True)
    bank = observation_bank()
    record = {
        "run_id": name,
        "variant": args.variant,
        "experiment": "core_benchmark",
        "phase": "development",
        "protocol": PROTOCOL,
        "environment": environment(),
        "predictions_per_repetition": args.predictions,
        "repetitions": args.repetitions,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "timing_scope": "decoder, token gathering, adapter and flow-matching head; "
        "excludes image, token and outer processing",
    }
    with recorded_run(directory / "run.json", record):
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        policy, record["model"] = load_policy(args.variant, args.checkpoint)
        pipeline = Pipeline(policy, args.checkpoint)
        with torch.inference_mode(), inference_autocast():
            record["equivalence"] = check_equivalence(policy, pipeline, bank)
            write_json(directory / "run.json", record)
            record["summary"] = time_neural(
                policy,
                pipeline,
                bank,
                args.variant,
                directory / "timings.csv",
                args.predictions,
                args.repetitions,
            )
    print(directory.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
