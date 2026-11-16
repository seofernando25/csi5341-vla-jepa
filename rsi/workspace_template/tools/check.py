"""Run the evaluator's offline check on this workspace and print a compact summary.

Environment: RSI_GPU_SLOTS_DIR (default /tmp/gpu-slots), RSI_GPU_SLOTS (default 1),
RSI_CHECK_CMD (optional replacement for the evaluator command prefix). The rsi package must
be importable (PYTHONPATH) because the GPU slot wrapper is `python -m rsi.gpu`.
"""

from __future__ import annotations

import datetime
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

EVALUATOR = ("-P", "-m", "evaluation.compression.study")
TAIL_CHARS = 1500


def _evaluator_prefix() -> list[str]:
    override = os.environ.get("RSI_CHECK_CMD")
    if override:
        return shlex.split(override)
    return [sys.executable, *EVALUATOR]


def _summary(record: dict) -> list[str]:
    lines = [f"status: {record.get('status')}"]
    sizes = record.get("sizes")
    if sizes and sizes.get("size_reduction") is not None:
        reduction = 100.0 * sizes["size_reduction"]
        lines.append(
            f"memory removed: {reduction:.1f}% "
            f"(resident {sizes['resident_bytes']} of baseline {sizes['baseline_resident_bytes']} bytes)"
        )
    latency = record.get("latency")
    if latency:
        lines.append(f"score: {latency['score']:.4f} ({latency.get('score_kind')})")
        reference = (latency.get("reference") or {}).get("timing") or {}
        candidate = latency.get("candidate") or {}
        speedup = candidate.get("speedup") or {}
        if speedup:
            lines.append(
                f"speedup: point {speedup.get('point', float('nan')):.4f}, "
                f"95% [{speedup.get('low', float('nan')):.4f}, "
                f"{speedup.get('high', float('nan')):.4f}], "
                f"{speedup.get('blocks', 0)}+{speedup.get('candidate_blocks', 0)} blocks, "
                f"{candidate.get('mode', '?')}"
            )
        lines.append(
            f"latency ms: reference median {reference.get('median_ms', float('nan')):.2f}, "
            f"candidate median {candidate.get('median_ms', float('nan')):.2f}, "
            f"p95 {candidate.get('p95_ms', float('nan')):.2f}"
        )
        if "arm_action_error" in candidate:
            lines.append(
                f"arm action error: {candidate['arm_action_error']:.6g} "
                f"(limit {candidate.get('arm_action_error_limit', float('nan')):g})"
            )
        if "perturbation_correlation" in candidate:
            lines.append(
                f"perturbation correlation: {candidate['perturbation_correlation']:.3f} "
                f"(min {candidate.get('perturbation_correlation_min', float('nan')):g})"
            )
        watch = candidate.get("gpu_watch") or {}
        if watch.get("samples"):
            lines.append(
                f"other GPU use: max {watch.get('other_utilisation_max_pct')} %, "
                f"{watch['samples']} samples"
            )
        if latency.get("zero_reason"):
            lines.append(f"score is zero: {latency['zero_reason']}")
        for violation in latency.get("violations") or []:
            lines.append(f"guard: {violation}")
    if "action_error" in record:
        lines.append(f"action error (max normalized difference): {record['action_error']:.6g}")
    check = record.get("check")
    if check:
        lines.append(
            f"latency ms: median {check['pipeline_median_ms']:.2f}, "
            f"p95 {check['pipeline_p95_ms']:.2f}"
        )
    if record.get("status") != "ok":
        error = str(record.get("error", ""))
        lines.append("error:")
        lines.append(error[-TAIL_CHARS:])
    return lines


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / ".check" / stamp
    gpu = [
        sys.executable,
        "-m",
        "rsi.gpu",
        "run",
        "--slots-dir",
        os.environ.get("RSI_GPU_SLOTS_DIR", "/tmp/gpu-slots"),
        "--n",
        os.environ.get("RSI_GPU_SLOTS", "1"),
        "--",
    ]
    evaluate = [*_evaluator_prefix(), "solution", "--workspace", str(root)]
    evaluate += ["--phase", "check", "--out", str(out)]
    completed = subprocess.run(gpu + evaluate, cwd=root, check=False)
    result = out / "result.json"
    if not result.is_file():
        print(f"status: no result (evaluator exit {completed.returncode}); see {out}")
        return 1
    record = json.loads(result.read_text(encoding="utf-8"))
    print("\n".join(_summary(record)))
    return 0 if record.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
