"""Offline check for this workspace: memory removed, action change and latency.

Not the evaluator. It loads the untouched policy, imports `solution.py`, calls
`compress(policy, calibration)`, and then measures what the evaluator charges: the tensor
storage reachable from the returned object and from this workspace's modules, the payloads
created after `compress`, live and peak CUDA bytes, and workspace files over 1 MiB. It then
compares 80 observation-bank predictions with the untouched policy's.

The evaluator also charges payloads parked outside those roots and files opened during
`compress`, so a solution that hides memory can score lower there than here. The score comes
from the evaluator.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parent
WORKSPACE = TOOLS.parent
sys.path.insert(0, str(TOOLS))

PREDICTIONS = 80
WARMUP = 10
FILE_CHARGE_BYTES = 1 << 20
REQUIRED_DEVICE = "cuda"
PROTOCOL_ENV = "VLA_PROTOCOL"
DEFAULT_CALIBRATION = "outputs/compression/calibration.pt"
POLICY_API = ("select_action", "predict_action_chunk")
TAIL_CHARS = 1500


def _protocol() -> dict:
    path = os.environ.get(PROTOCOL_ENV)
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def _calibration(protocol: dict) -> dict:
    """The artifact the study names, with the fields it excludes removed."""
    from evaluation.common import ROOT

    spec = protocol.get("calibration") or {}
    artifact = spec.get("artifact") or DEFAULT_CALIBRATION
    data = torch.load(ROOT / artifact, weights_only=True)
    for field in spec.get("exclude_fields") or ():
        data.pop(str(field), None)
    return data


def _workspace_file_bytes() -> int:
    """Bytes of every workspace file over the charge threshold, `__pycache__` aside."""
    total = 0
    for path in WORKSPACE.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        size = path.stat().st_size
        if size > FILE_CHARGE_BYTES:
            total += size
    return total


def _workspace_modules() -> list:
    """Modules imported out of the workspace, as the evaluator walks them: not this tool."""
    prefix, tools = str(WORKSPACE) + os.sep, str(TOOLS) + os.sep
    return [
        module
        for module in list(sys.modules.values())
        if (os.path.abspath(getattr(module, "__file__", "") or "")).startswith(prefix)
        and not (os.path.abspath(getattr(module, "__file__", "") or "")).startswith(tools)
    ]


def _resident(policy: object, roots: tuple, skip: frozenset | None = None) -> tuple[int, list]:
    """Resident bytes and violations, by the evaluator's own walk and payload sweep."""
    from memory import new_payloads
    from resident import ResidentWalk, live_cuda_bytes

    walk = ResidentWalk(REQUIRED_DEVICE)
    walk.add(policy, *roots)
    storages = dict(walk.storages)
    if skip is not None:
        for key, nbytes in new_payloads(skip).items():
            storages[key] = max(storages.get(key, 0), nbytes)
    return max(sum(storages.values()), live_cuda_bytes()), walk.violations


def _import_solution():
    path = WORKSPACE / "solution.py"
    spec = importlib.util.spec_from_file_location("vla_solution_check", path)
    if spec is None or spec.loader is None:
        raise ValueError("workspace has no importable solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "compress", None)):
        raise TypeError("solution.py defines no compress(policy, calibration)")
    return module


def _screen(policy, references, bank, pipeline, seed, calibration, autocast) -> list[dict]:
    """Action change and latency on the bank frames outside the calibration set."""
    used = set(calibration.get("bank_indices") or ())
    indices = [index for index in range(len(bank)) if index not in used] or list(range(len(bank)))
    rows: list[dict] = []
    with torch.inference_mode(), autocast():
        for step in range(WARMUP + PREDICTIONS):
            index = indices[step % len(indices)]
            policy.reset()
            torch.manual_seed(seed + index)
            torch.cuda.synchronize()
            started = time.perf_counter()
            predicted = policy.predict_action_chunk(pipeline.sample(bank[index])).float()
            torch.cuda.synchronize()
            if step < WARMUP:
                continue
            reference = references[index : index + 1].to(REQUIRED_DEVICE)
            rows.append(
                {
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "difference": float(
                        (predicted - reference[:, : predicted.shape[1]]).abs().max()
                    ),
                    "finite": bool(torch.isfinite(predicted).all()),
                }
            )
    return rows


def _summary(resident: int, baseline: int, rows: list[dict], compress_s: float) -> list[str]:
    reduction = max(0.0, 1 - resident / baseline) if baseline else 0.0
    latency = [row["latency_ms"] for row in rows]
    return [
        "status: ok",
        (
            f"memory removed: {100.0 * reduction:.1f}% "
            f"(resident {resident} of baseline {baseline} bytes)"
        ),
        f"action error (max normalized difference): {max(row['difference'] for row in rows):.6g}",
        (
            f"latency ms: median {float(np.median(latency)):.2f}, "
            f"p95 {float(np.percentile(latency, 95)):.2f}"
        ),
        f"compress: {compress_s:.1f} s",
        f"nonfinite predictions: {sum(not row['finite'] for row in rows)}",
    ]


def main() -> int:
    from memory import live_payloads

    from evaluation.common import ROOT
    from evaluation.models import load_policy
    from evaluation.pipeline import Pipeline, inference_autocast, observation_bank
    from rsi.gpu import gpu_slot

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    slots = Path(os.environ.get("RSI_GPU_SLOTS_DIR", "/tmp/gpu-slots"))
    count = int(os.environ.get("RSI_GPU_SLOTS", "1"))
    try:
        with gpu_slot(slots, count):
            protocol = _protocol()
            seed = int(protocol.get("seed", 1000))
            calibration = _calibration(protocol)
            policy, _metadata = load_policy("B16")
            bank = observation_bank()
            pipeline = Pipeline(policy)
            references = torch.load(ROOT / "outputs/compression/references.pt", weights_only=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), inference_autocast():
                policy.reset()
                torch.manual_seed(seed)
                policy.predict_action_chunk(pipeline.sample(bank[0]))
            peak = torch.cuda.max_memory_allocated()
            walk, baseline_violations = _resident(policy, ())
            baseline = max(walk, peak)
            skip = frozenset(live_payloads())
            sys.path.insert(0, str(WORKSPACE))
            module = _import_solution()
            torch.manual_seed(seed)
            started = time.perf_counter()
            returned = module.compress(policy, calibration)
            compress_s = time.perf_counter() - started
            if returned is None or not all(hasattr(returned, name) for name in POLICY_API):
                raise ValueError(f"compress must return an object with {POLICY_API}")
            roots = (vars(module), *(vars(item) for item in _workspace_modules()))
            before, violations = _resident(returned, roots, skip)
            file_bytes = _workspace_file_bytes()
            torch.cuda.reset_peak_memory_stats()
            rows = _screen(
                returned, references, bank, pipeline, seed, calibration, inference_autocast
            )
            after, later = _resident(returned, roots, skip)
            resident = max(before, after, torch.cuda.max_memory_allocated()) + file_bytes
    except Exception:  # noqa: BLE001 - report every failure of the workspace's own code
        print("status: solution_error")
        print("error:")
        print(traceback.format_exc()[-TAIL_CHARS:])
        return 1
    print("\n".join(_summary(resident, baseline, rows, compress_s)))
    baseline = Counter(baseline_violations)
    found = sorted(
        set((Counter(violations) - baseline).elements())
        | set((Counter(later) - baseline).elements())
    )
    if found:
        print("violations: " + "; ".join(found[:10]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
