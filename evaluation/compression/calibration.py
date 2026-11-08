"""Unlabeled calibration from 20 development observations (two per task); no robot outcomes.

`calibrate()` stores per-Linear input energy, per-block residual influence, prompt token IDs
and B16 reference action chunks for the whole observation bank.
"""

from collections import defaultdict

import numpy as np
import torch

from evaluation.common import PROTOCOL, ROOT, environment, write_json
from evaluation.compression.topology import blocks, roots, selected_linears
from evaluation.models import load_policy
from evaluation.pipeline import Pipeline, observation_bank

STUDY = ROOT / "studies/compression"
RUNTIME = ROOT / "outputs/compression"
CALIBRATION = RUNTIME / "calibration.pt"
REFERENCES = RUNTIME / "references.pt"


def setup():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def calibration_indices(bank):
    """Bank indices of the first two observations of every task."""
    seen, result = defaultdict(int), []
    for index, sample in enumerate(bank):
        if seen[sample["task_id"]] < 2:
            result.append(index)
        seen[sample["task_id"]] += 1
    if len(result) != 20:
        raise ValueError("Expected exactly two calibration states per task")
    return result


def predict_bank(policy, pipeline, bank, indices):
    """Seeded action chunks for the given bank indices, under inference autocast."""
    outputs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in indices:
            policy.reset()
            torch.manual_seed(PROTOCOL["seed"] + index)
            output = policy.predict_action_chunk(pipeline.sample(bank[index]))
            if not torch.isfinite(output).all():
                raise ValueError(f"Nonfinite actions at bank_index={index}")
            outputs.append(output)
    return outputs


def load():
    return torch.load(CALIBRATION, weights_only=True)


def calibrate():
    if CALIBRATION.exists():
        raise FileExistsError("Preserve frozen calibration")
    from evaluation.core_benchmark import prepared_inputs

    setup()
    bank = observation_bank()
    indices = calibration_indices(bank)
    policy, metadata = load_policy("B16")
    pipeline = Pipeline(policy)
    sums, counts, influence, handles = {}, defaultdict(int), defaultdict(list), []

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
            energy = (y.float() - x.float()).square().mean()
            influence[name].append(float(energy / x.float().square().mean().clamp_min(1e-8)))

        return hook

    for full, (_, _, module) in selected_linears(policy, "all").items():
        handles.append(module.register_forward_pre_hook(linear_hook(full)))
    for scope in roots(policy):
        for index, block in enumerate(blocks(policy, scope)):
            handles.append(
                block.register_forward_hook(block_hook(f"{scope}.{index}"), with_kwargs=True)
            )
    try:
        predict_bank(policy, pipeline, bank, indices)
    finally:
        for handle in handles:
            handle.remove()
    token_ids = []
    with torch.inference_mode():
        for index in indices:
            encoded, _ = prepared_inputs(policy, pipeline.sample(bank[index]))
            token_ids.append(encoded["input_ids"].detach().cpu().flatten())
    calibration = {
        "linear_inputs": {k: (v / counts[k]).cpu() for k, v in sums.items()},
        "block_influence": {k: float(np.mean(v)) for k, v in influence.items()},
        "token_ids": torch.unique(torch.cat(token_ids)),
        "bank_indices": indices,
    }
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibration, CALIBRATION)
    write_json(
        STUDY / "calibration.json",
        {
            "purpose": "Unlabeled calibration; no optimizer updates or robot outcomes",
            "bank_indices": indices,
            "states_per_task": 2,
            "linear_modules": len(sums),
            "token_count": len(calibration["token_ids"]),
            "block_influence": calibration["block_influence"],
            "model": metadata,
            "environment": environment(),
        },
    )
    references = predict_bank(policy, pipeline, bank, range(len(bank)))
    torch.save(torch.cat([r.float().cpu() for r in references]), REFERENCES)
    print("Calibration and 100 matched reference action chunks frozen", flush=True)
