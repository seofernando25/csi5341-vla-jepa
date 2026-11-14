"""Linear CKA between each block's input and output on the calibration states.

Example: python -m evaluation.compression.cka   # writes studies/compression/cka.json
Layer pruning with criterion "cka" removes the blocks whose output most resembles their input.
"""

import argparse
from collections import defaultdict

import torch

from evaluation.common import ROOT, write_json
from evaluation.compression.calibration import load, predict_bank, setup
from evaluation.compression.topology import blocks, roots
from evaluation.models import load_policy
from evaluation.pipeline import Pipeline, observation_bank


def linear_cka(x, y):
    x = x.float() - x.float().mean(0)
    y = y.float() - y.float().mean(0)
    k = x @ x.T
    l = y @ y.T
    return float((k * l).sum() / (k.square().sum() * l.square().sum()).sqrt().clamp_min(1e-12))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    output = ROOT / "studies/compression/cka.json"
    if output.exists():
        raise FileExistsError("Preserve earlier CKA profile")
    setup()
    bank = observation_bank()
    calibration = load()
    policy, _ = load_policy("B16")
    pipeline = Pipeline(policy)
    features = defaultdict(list)
    handles = []
    for scope in roots(policy):
        for index, block in enumerate(blocks(policy, scope)):

            def hook(module, args, kwargs, output, name=f"{scope}.{index}"):
                x = args[0] if args else kwargs["hidden_states"]
                y = output[0] if isinstance(output, tuple) else output
                x = x.reshape(-1, x.shape[-1])
                y = y.reshape(-1, y.shape[-1])
                rows = (
                    torch.linspace(0, len(x) - 1, min(16, len(x)), device=x.device).round().long()
                )
                features[name].append(
                    (x[rows].detach().float().cpu(), y[rows].detach().float().cpu())
                )

            handles.append(block.register_forward_hook(hook, with_kwargs=True))
    try:
        predict_bank(policy, pipeline, bank, calibration["bank_indices"])
    finally:
        for handle in handles:
            handle.remove()
    scores = {}
    for name, pairs in features.items():
        scores[name] = linear_cka(
            torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])
        )
    write_json(
        output,
        {
            "method": "linear CKA between block input and output",
            "scores": scores,
            "sampling": "16 evenly spaced token rows per block invocation, 20 calibration states; four action denoising steps",
            "interpretation": "CKA ranking adaptation; no fine-tuning and no robot-success claim",
        },
    )


if __name__ == "__main__":
    main()
