"""QuantVLA attention matching/output balancing adapted to VLA-JEPA's DiT.

Use the pinned upstream Diffusers processor and its published ratio/clamp
formulas. The quantizer and BF16 teacher differ from QuantVLA's GR00T setup;
these candidates are adaptations, not full QuantVLA reproductions.
"""

import importlib.util
import math
from collections import defaultdict

import torch

from evaluation.common import ROOT, file_hash, read_json


def alpha_ratio(teacher, compressed):
    alpha = torch.where(compressed > 0, teacher / (compressed + 1e-6), torch.ones_like(teacher))
    alpha = alpha.clamp(0.7, 1.4)
    return torch.where((alpha - 1).abs() < 0.02, torch.ones_like(alpha), alpha)


def beta_ratio(teacher, compressed):
    value = math.log(max(float(teacher), 1e-8) / max(float(compressed), 1e-8))
    value = max(-0.30, min(0.30, value))
    return 1.0 if abs(value) < 0.03 else math.exp(value)


def matching(policy, change, calibration):
    from evaluation.compression.study import inputs, observation_bank
    from evaluation.compression.transforms import apply
    from evaluation.run import PROTOCOL, processors

    manifest = read_json(ROOT / "studies/compression/methods.json")["upstream"]["quantvla"]
    folder = ROOT / "outputs/compression/vendor/quantvla"
    name = "gr00t/atm/dit_atm.py"
    if file_hash(folder / name) != manifest["files"][name]:
        raise ValueError("Pinned QuantVLA attention processor changed")
    spec = importlib.util.spec_from_file_location("compression_quantvla_atm", folder / name)
    upstream = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through the module registry.
    import sys

    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    root = policy.model.action_model
    upstream.ensure_dit_attention_patch(root, scope="all")
    events = [
        {
            "method": "QuantVLA processor adaptation",
            "revision": manifest["revision"],
            "teacher_precision": "accepted baseline BF16",
            "full_QuantVLA_reproduction": False,
        }
    ]
    if change.get("control"):
        return events
    bank, pipeline = observation_bank(), processors(policy)

    def capture():
        logits, outputs = defaultdict(list), defaultdict(list)
        upstream.register_atm_capture(
            root, lambda name, x: logits[name].append(x.float().cpu()), "all"
        )
        upstream.register_ohb_capture(
            root, lambda name, x: outputs[name].append(x.float().cpu()), "all"
        )
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for index in calibration["bank_indices"]:
                    policy.reset()
                    torch.manual_seed(PROTOCOL["seed"] + index)
                    policy.predict_action_chunk(inputs(bank[index], pipeline))
        finally:
            upstream.clear_atm_capture(root)
        return (
            {name: torch.stack(values).mean(0).flatten() for name, values in logits.items()},
            {name: torch.stack(values).mean() for name, values in outputs.items()},
        )

    teacher_logits, teacher_outputs = capture()
    events.extend(apply(policy, {"changes": [change["quantizer"]]}, calibration))
    quant_logits, quant_outputs = capture()
    for name, module in root.named_modules():
        if name not in teacher_logits:
            continue
        alpha = alpha_ratio(teacher_logits[name], quant_logits[name])
        beta = beta_ratio(teacher_outputs[name], quant_outputs[name])
        if change.get("atm", True):
            module.register_buffer("_atm_alpha_all", alpha.to(next(module.parameters()).device))
        if change.get("ohb", False):
            module._ohb_beta_scalar = beta
        events.append(
            {
                "method": "attention statistics matching",
                "module": name,
                "alpha": alpha.tolist(),
                "beta": beta,
                "atm_enabled": change.get("atm", True),
                "ohb_enabled": change.get("ohb", False),
                "calibration": "20 initial states; all four denoising steps pooled",
                "quantizer": change["quantizer"],
            }
        )
    return events
