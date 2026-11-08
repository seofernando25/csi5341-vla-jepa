"""Hessian-based post-training compression from pinned upstream code: GPTQ and SparseGPT."""

import importlib.util
import sys

import torch
from torch import nn

from evaluation.compression.calibration import predict_bank
from evaluation.compression.quantization import PackedInt4, guard_action_dtype
from evaluation.compression.topology import replace, selected_linears
from evaluation.compression.vendor import upstream
from evaluation.pipeline import Pipeline, observation_bank


def load_upstream(method):
    folder, _ = upstream(method)
    # Both repositories use a top-level quant.py import; isolate that name during loading.
    previous = sys.modules.pop("quant", None)
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(
            f"compression_upstream_{method}", folder / f"{method}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        quant = sys.modules["quant"]
    finally:
        sys.path.remove(str(folder))
        sys.modules.pop("quant", None)
        if previous is not None:
            sys.modules["quant"] = previous
    return module, quant


def second_order(policy, change, calibration):
    """GPTQ (4-bit, packed) or SparseGPT, calibrating one block at a time.

    Blocks are processed in order, so each sees inputs from already compressed predecessors.
    """
    method = change["method"]
    events = guard_action_dtype(policy, change["scope"]) if method == "gptq" else []
    upstream, quant = load_upstream(method)
    selected = selected_linears(policy, change["scope"])
    groups = {}
    for full, entry in selected.items():
        parts = full.split(".")
        group = ".".join(parts[:3]) if parts[0] != "action" else ".".join(parts[:4])
        groups.setdefault(group, []).append((full, entry))
    bank = observation_bank()
    pipeline = Pipeline(policy)
    for group, entries in groups.items():
        helpers, handles = {}, []
        for full, (_, _, linear) in entries:
            helper = upstream.GPTQ(linear) if method == "gptq" else upstream.SparseGPT(linear)
            if method == "gptq":
                helper.quantizer = quant.Quantizer()
                helper.quantizer.configure(4, perchannel=True, sym=False, mse=False)
            helpers[full] = helper

            def hook(module, args, output, helper=helper):
                # Calibration outer autocast must not turn Hessian products into BF16.
                with torch.autocast("cuda", enabled=False):
                    helper.add_batch(args[0].detach().float(), output.detach().float())

            handles.append(linear.register_forward_hook(hook))
        try:
            predict_bank(policy, pipeline, bank, calibration["bank_indices"])
        finally:
            for handle in handles:
                handle.remove()
        for full, (root, name, linear) in entries:
            helper = helpers[full]
            if method == "gptq":
                scales, zeros = [], []
                original = helper.quantizer.find_params

                def save_qparams(
                    x, weight=False, original=original, scales=scales, zeros=zeros, helper=helper
                ):
                    original(x, weight=weight)
                    if x.shape[-1] == 128:
                        scales.append(helper.quantizer.scale.detach().clone().flatten())
                        zeros.append(helper.quantizer.zero.detach().clone().flatten())

                helper.quantizer.find_params = save_qparams
                helper.fasterquant(blocksize=128, percdamp=0.01, groupsize=128, actorder=False)
                scale = torch.stack(scales, 1)
                zero = torch.stack(zeros, 1)
                q = (
                    (
                        linear.weight.detach().float().reshape(linear.out_features, -1, 128)
                        / scale[..., None]
                        + zero[..., None]
                    )
                    .round()
                    .clamp(0, 15)
                )
                replace(root, name, PackedInt4(linear, q.reshape_as(linear.weight), scale, zero))
                events.append(
                    {
                        "method": "GPTQ",
                        "module": full,
                        "bits": 4,
                        "group_size": 128,
                        "act_order": False,
                        "damping": 0.01,
                        "packed_kernel": "aten tinygemm",
                    }
                )
            else:
                pattern = change.get("pattern")
                helper.fasterprune(
                    change.get("fraction", 0.5),
                    prunen=2 if pattern == "2:4" else 0,
                    prunem=4 if pattern == "2:4" else 0,
                    percdamp=0.01,
                )
                removed = int((linear.weight == 0).sum())
                if change.get("sparse_kernel"):
                    if pattern != "2:4":
                        raise ValueError("Packed sparse deployment requires a 2:4 pattern")
                    linear.weight = nn.Parameter(
                        torch.sparse.to_sparse_semi_structured(linear.weight),
                        requires_grad=False,
                    )
                events.append(
                    {
                        "method": "SparseGPT",
                        "module": full,
                        "fraction": change.get("fraction", 0.5),
                        "pattern": pattern,
                        "removed_weights": removed,
                        "storage": "2:4 packed" if change.get("sparse_kernel") else "dense zeros",
                    }
                )
            helper.free()
        print(f"{method} completed {group}", flush=True)
    return events
