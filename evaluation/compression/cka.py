"""Centered-kernel-alignment depth profiling of the frozen calibration states."""

from collections import defaultdict

import torch

from evaluation.common import ROOT, file_hash, write_json
from evaluation.compression.study import inputs, observation_bank, setup
from evaluation.compression.transforms import roots
from evaluation.models import load_policy
from evaluation.run import PROTOCOL, processors


def linear_cka(x, y):
    x = x.float() - x.float().mean(0)
    y = y.float() - y.float().mean(0)
    k = x @ x.T
    l = y @ y.T
    return float((k * l).sum() / (k.square().sum() * l.square().sum()).sqrt().clamp_min(1e-12))


def main():
    output = ROOT / "studies/compression/cka.json"
    if output.exists():
        raise FileExistsError("Preserve earlier CKA profile")
    setup()
    bank = observation_bank()
    calibration = torch.load(ROOT / "outputs/compression/calibration.pt", weights_only=True)
    policy, _ = load_policy("B16")
    pipeline = processors(policy)
    features = defaultdict(list)
    handles = []
    for scope, root in roots(policy).items():
        path = {"text": "layers", "vision": "blocks", "action": "model.transformer_blocks"}[scope]
        for index, block in enumerate(root.get_submodule(path)):

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
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index in calibration["bank_indices"]:
                torch.manual_seed(PROTOCOL["seed"] + index)
                policy.predict_action_chunk(inputs(bank[index], pipeline))
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
            "calibration_sha256": file_hash(ROOT / "outputs/compression/calibration.pt"),
            "sampling": "16 evenly spaced token rows per block invocation, 20 calibration states; four action denoising steps",
            "interpretation": "CKA ranking adaptation; no fine-tuning and no robot-success claim",
        },
    )


if __name__ == "__main__":
    main()
