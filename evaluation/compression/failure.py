"""Failure analysis of the language-decoder precision cells on the screen's 80 fixed frames.

Example: python -m evaluation.compression.failure   # writes studies/program-v1/failure/failure.json

The screen (`study.screen`) measures action change on the 80 observation-bank frames outside
calibration. This module replays those same frames for the reference B16 and for the four
language-decoder cells, and records where representations and actions diverge:

- linear CKA against B16 for every decoder layer's output (all tokens, image tokens, text tokens)
  and for the action head's conditioning input (the 32 embodied-action tokens);
- per-frame action difference against B16: physical arm MSE, gripper disagreement and flips;
- attention mass of the embodied-action query tokens on image tokens, and its Spearman rank
  correlation with B16, measured under eager attention because the deployed sdpa kernel returns
  no weights;
- per-task successes from the committed rollout records, per phase, never pooled.

Diagnostics only: representation drift and offline action error are not LIBERO success.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from evaluation.common import PROTOCOL, ROOT, environment, read_json, recorded_run, write_json
from evaluation.compression import calibration as calib
from evaluation.compression.cka import linear_cka
from evaluation.compression.recipes import apply, validate
from evaluation.compression.topology import roots
from evaluation.models import load_policy
from evaluation.pipeline import Pipeline, inference_autocast, observation_bank

STUDY = ROOT / "studies/program-v1"
OUTPUT = STUDY / "failure" / "failure.json"
CELLS = {
    "precision-lang-int8": 8,
    "precision-lang-nf4": 4,
    "precision-lang-hqq3": 3,
    "precision-lang-hqq2": 2,
}
ROWS = 16  # evenly spaced token rows per frame, as in evaluation/compression/cka.py
PHASES = {"development": "development", "reserved": "final"}


def screen_frames(calibration, bank):
    """Bank indices the screen measures: every frame outside the 20 calibration states."""
    return [index for index in range(len(bank)) if index not in calibration["bank_indices"]]


def sample_rows(positions, rows=ROWS):
    """`rows` evenly spaced entries of a 1-D index tensor."""
    if len(positions) == 0:
        return positions
    picks = torch.linspace(0, len(positions) - 1, min(rows, len(positions))).round().long()
    return positions[picks]


def record(policy, pipeline, bank, frames):
    """Per-frame decoder features, conditioning tokens and actions for one model."""
    layers = roots(policy)["text"].layers
    image_token_id = policy.model.qwen.model.config.image_token_id
    features = {name: [[] for _ in layers] for name in ("all", "image", "text")}
    conditioning, actions, physical = [], [], []
    state = {"ids": None, "conditioning": None}
    handles = []

    def ids_hook(module, args, kwargs):
        state["ids"] = kwargs["input_ids"][0]

    def layer_hook(index):
        def hook(module, args, kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.reshape(-1, hidden.shape[-1])
            ids = state["ids"]
            image = (ids == image_token_id).nonzero().flatten()
            text = (ids != image_token_id).nonzero().flatten()
            for name, positions in (
                ("all", torch.arange(len(hidden), device=hidden.device)),
                ("image", image),
                ("text", text),
            ):
                picks = sample_rows(positions)
                features[name][index].append(hidden[picks].detach().float().cpu())

        return hook

    def conditioning_hook(module, args, kwargs):
        state["conditioning"] = kwargs["encoder_hidden_states"]

    handles.append(
        policy.model.qwen.model.model.register_forward_pre_hook(ids_hook, with_kwargs=True)
    )
    handles.append(
        policy.model.action_model.model.register_forward_pre_hook(
            conditioning_hook, with_kwargs=True
        )
    )
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(layer_hook(index), with_kwargs=True))
    try:
        with torch.inference_mode(), inference_autocast():
            for index in frames:
                policy.reset()
                torch.manual_seed(PROTOCOL["seed"] + index)
                state["ids"], state["conditioning"] = None, None
                predicted = policy.predict_action_chunk(pipeline.sample(bank[index])).float()
                actions.append(predicted[0].cpu())
                physical.append(pipeline.post(predicted.clone()).float()[0].cpu())
                conditioning.append(state["conditioning"][0].detach().float().cpu())
    finally:
        for handle in handles:
            handle.remove()
    stacked = {
        name: [torch.cat(chunks) for chunks in per_layer] for name, per_layer in features.items()
    }
    tokens = torch.cat(conditioning)
    return {
        **stacked,
        "conditioning": tokens,
        "actions": torch.stack(actions),
        "physical": torch.stack(physical),
        "max_abs": {
            name: max(float(tensor.abs().max()) for tensor in tensors)
            for name, tensors in stacked.items()
        }
        | {"conditioning": float(tokens.abs().max())},
    }


def record_attention(policy, pipeline, bank, frames):
    """Per-frame attention mass of the embodied-action queries on image tokens, per layer."""
    layers = roots(policy)["text"].layers
    roots(policy)["text"].config._attn_implementation = "eager"
    image_token_id = policy.model.qwen.model.config.image_token_id
    embodied_token_id = policy.model.embodied_action_token_id
    mass = [[] for _ in layers]
    state = {"ids": None, "check": True}
    handles = []

    def ids_hook(module, args, kwargs):
        state["ids"] = kwargs["input_ids"][0]

    def attention_hook(index):
        def hook(module, args, kwargs, output):
            weights = output[1]
            if weights is None:
                raise RuntimeError("eager attention returned no weights")
            if state["check"]:
                # A non-causal mask would make the image-token mass meaningless.
                if float(weights[0][0].triu(diagonal=1).abs().max()) != 0.0:
                    raise ValueError("eager attention is not causal; refusing to record mass")
                state["check"] = False
            ids = state["ids"]
            queries = (ids == embodied_token_id).nonzero().flatten()
            columns = (ids == image_token_id).nonzero().flatten()
            mass[index].append(
                weights[0][:, queries][:, :, columns].sum(-1).mean(0).detach().float().cpu()
            )

        return hook

    handles.append(
        policy.model.qwen.model.model.register_forward_pre_hook(ids_hook, with_kwargs=True)
    )
    for index, layer in enumerate(layers):
        handles.append(
            layer.self_attn.register_forward_hook(attention_hook(index), with_kwargs=True)
        )
    try:
        with torch.inference_mode(), inference_autocast():
            for index in frames:
                policy.reset()
                torch.manual_seed(PROTOCOL["seed"] + index)
                inputs = policy._prepare_model_inputs(pipeline.sample(bank[index]), training=False)
                policy.model._encode_qwen(
                    inputs["images"], inputs["instructions"], need_action_tokens=False
                )
    finally:
        for handle in handles:
            handle.remove()
    return [torch.stack(chunks) for chunks in mass]


def stable_cka(x, y):
    """Linear CKA on unit-Frobenius features.

    CKA is invariant to a positive rescaling of either side, and the raw Gram squares overflow
    float32 on the decoder's massive activations: one channel above about 1e4 makes
    `linear_cka` return exactly 0.0. Text rows carry those channels; image rows do not.
    """
    x = x.float() - x.float().mean(0)
    y = y.float() - y.float().mean(0)
    return linear_cka(x / x.norm().clamp_min(1e-12), y / y.norm().clamp_min(1e-12))


def cka_profile(reference, cell):
    """Linear CKA of every decoder layer and the conditioning input against the reference."""
    profile = {
        name: [stable_cka(x, y) for x, y in zip(reference[name], cell[name], strict=True)]
        for name in ("all", "image", "text")
    }
    profile["conditioning"] = stable_cka(reference["conditioning"], cell["conditioning"])
    return profile


def half_profile(reference, cell):
    """Last-layer CKA on each half of the frames: how stable the row sample is."""
    split = len(reference["all"][-1]) // 2
    return {
        name: [
            stable_cka(reference[name][-1][:split], cell[name][-1][:split]),
            stable_cka(reference[name][-1][split:], cell[name][-1][split:]),
        ]
        for name in ("all", "image")
    }


def action_summary(physical, reference):
    """Per-frame arm MSE and gripper disagreement against the reference, and their summary."""
    arm = [
        (frame[..., :6] - base[..., :6]).square().mean().item()
        for frame, base in zip(physical, reference, strict=True)
    ]
    disagreement = [
        (frame[..., 6] != base[..., 6]).float().mean().item()
        for frame, base in zip(physical, reference, strict=True)
    ]
    flips = [bool(d > 0) for d in disagreement]
    return {
        "frames": len(arm),
        "per_frame_arm_mse": arm,
        "per_frame_gripper_disagreement": disagreement,
        "mean_arm_mse": float(np.mean(arm)),
        "median_arm_mse": float(np.median(arm)),
        "max_arm_mse": float(np.max(arm)),
        "frames_with_gripper_flip": int(sum(flips)),
        "mean_gripper_disagreement": float(np.mean(disagreement)),
        "gripper_steps_disagreeing": round(sum(disagreement) * physical[0].shape[0]),
        "gripper_steps": int(physical[0].shape[0] * len(disagreement)),
    }


def attention_summary(mass, reference):
    """Mean image-token attention mass per layer and its rank correlation with the reference."""
    per_layer = [float(values.mean()) for values in mass]
    rho = []
    for cell_layer, base_layer in zip(mass, reference, strict=True):
        if float(cell_layer.std()) == 0 or float(base_layer.std()) == 0:
            rho.append(None)
            continue
        rho.append(float(spearmanr(cell_layer.flatten().numpy(), base_layer.flatten().numpy())[0]))
    return {
        "mean_mass_per_layer": per_layer,
        "mean_mass": float(np.mean(per_layer)),
        "spearman_per_layer": rho,
        "spearman_median": float(np.median([value for value in rho if value is not None])),
        "queries": "embodied-action tokens; columns: image tokens; mean over heads",
        "implementation": "eager attention; the deployed sdpa kernel returns no weights",
    }


def reproduction(cell, arm_mse, frames):
    """Largest per-frame disagreement with the committed screen record, matched by bank index."""
    path = STUDY / "measurements" / cell / "screen" / "observations.json"
    source = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    if not path.exists():
        return {"status": "missing", "source": source}
    recorded = {row["bank_index"]: row["physical_arm_change_mse"] for row in read_json(path)}
    differences = [
        abs(arm_mse[position] - recorded[index])
        for position, index in enumerate(frames)
        if index in recorded
    ]
    return {
        "status": "ok" if len(differences) == len(frames) else "partial",
        "source": source,
        "matched_frames": len(differences),
        "max_abs_arm_mse_difference": float(max(differences)) if differences else None,
    }


def task_success(study, cell, phase):
    """Per-task successes of one cell in one phase, from its committed record."""
    if phase == "reserved":
        summary = study / "reserved" / "summary.json"
        if not summary.exists():
            return None
        for entry in read_json(summary).get("cells", []):
            if entry.get("cell") == cell and entry.get("reserved"):
                return {str(k): int(v) for k, v in entry["reserved"]["per_task"].items()}
        return None
    path = study / "measurements" / cell / PHASES[phase] / "episodes.csv"
    if not path.exists():
        return None
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    counts = {}
    for row in rows:
        task = str(row["task_id"])
        counts[task] = counts.get(task, 0) + int(row["success"])
    return counts


def task_drops(study=STUDY, cells=CELLS):
    """Per-task success per phase against B16 in the same phase, with the largest drops."""
    out = {}
    for phase in PHASES:
        reference = task_success(study, "B16", phase)
        if not reference:
            continue
        entry = {"reference": reference, "cells": {}}
        for cell in cells:
            counts = task_success(study, cell, phase)
            if not counts:
                continue
            delta = {task: counts[task] - reference[task] for task in sorted(reference)}
            worst = sorted(delta.items(), key=lambda item: item[1])[:3]
            entry["cells"][cell] = {
                "per_task": counts,
                "delta_vs_reference": delta,
                "worst_tasks": [{"task_id": int(task), "delta": value} for task, value in worst],
                "tasks_dropped": sum(value < 0 for value in delta.values()),
                "tasks_gained": sum(value > 0 for value in delta.values()),
            }
        if entry["cells"]:
            out[phase] = entry
    return out


def measure(cell, frames, bank, calibration, attention=True):
    """Load one cell (B16 when `cell` is None), replay the frames, and free the model."""
    policy, metadata = load_policy("B16")
    events = []
    if cell is not None:
        recipe_path = STUDY / "recipes" / f"{cell}.json"
        recipe = validate(read_json(recipe_path), recipe_path)
        events = apply(policy, recipe, calibration)
    pipeline = Pipeline(policy)
    captured = record(policy, pipeline, bank, frames)
    captured["attention"] = record_attention(policy, pipeline, bank, frames) if attention else None
    captured["model"] = {
        key: metadata[key]
        for key in ("variant", "chunk_size", "n_action_steps", "num_inference_timesteps")
    }
    captured["transformations"] = events
    del policy, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--cells", nargs="*", default=list(CELLS))
    parser.add_argument("--frames", type=int, default=0, help="use only the first N screen frames")
    parser.add_argument("--no-attention", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Preserve earlier failure analysis: {args.out}")
    calib.setup()
    bank = observation_bank()
    calibration = calib.load()
    frames = screen_frames(calibration, bank)
    if args.frames:
        frames = frames[: args.frames]
    attention = not args.no_attention
    record_doc = {
        "question": (
            "Where does language-decoder quantization change the policy on identical observations: "
            "representations, action trajectories, spatial attention, or task success?"
        ),
        "protocol": {
            "frames": len(frames),
            "bank_indices": frames,
            "frame_source": "observation bank indices outside the 20 calibration states",
            "seed": PROTOCOL["seed"],
            "seeding": "torch.manual_seed(seed + bank_index) after policy.reset(), as in the screen",
            "token_rows": ROWS,
            "token_row_sampling": "evenly spaced rows per frame per layer, as in cka.py",
            "attention": "eager attention; the deployed sdpa kernel returns no weights"
            if attention
            else "skipped",
            "phases": "development (100 trials) and reserved (500 trials) are never pooled",
        },
        "environment": environment(),
    }
    with recorded_run(args.out, record_doc) as doc:
        reference = measure(None, frames, bank, calibration, attention)
        doc["reference"] = {
            "cell": "B16",
            "model": reference["model"],
            "actions": action_summary(reference["physical"], reference["physical"]),
            "max_abs_activation": reference["max_abs"],
            "attention_mean_mass": float(
                np.mean([values.mean() for values in reference["attention"]])
            )
            if attention
            else None,
        }
        doc["cells"] = []
        for cell in args.cells:
            captured = measure(cell, frames, bank, calibration, attention)
            summary = action_summary(captured["physical"], reference["physical"])
            entry = {
                "cell": cell,
                "nominal_bits": CELLS[cell],
                "cka": cka_profile(reference, captured),
                "cka_half_frames": half_profile(reference, captured),
                "max_abs_activation": captured["max_abs"],
                "actions": summary,
                "transformations": len(captured["transformations"]),
            }
            entry["reproduction"] = reproduction(cell, summary["per_frame_arm_mse"], frames)
            if attention:
                entry["attention"] = attention_summary(
                    captured["attention"], reference["attention"]
                )
            doc["cells"].append(entry)
            write_json(args.out, doc)
        doc["tasks"] = task_drops()
        doc["does_not_show"] = [
            "LIBERO success: CKA and offline action error are not task outcomes",
            "attention under the deployed sdpa kernel, which returns no weights",
            "whether a task drop is caused by the decoder drift rather than by the action head",
        ]
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
