"""Shared compression-execution helpers for Dream-RSI: GPU guard and subprocess runner."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

B16_RESIDENT_BYTES = 5532276494.0
MEASUREMENTS_DIR = Path("studies/program-v1/measurements")
GPU_BACKOFF_USED_FRACTION = 0.85


def gpu_name() -> str:
    """GPU model string; never touches other processes."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.splitlines()
        return out[0].strip() if out else "unknown"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def gpu_tight() -> bool:
    """True when the first GPU reports more than the backoff fraction of memory used."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.splitlines()
        used, total = (float(x) for x in out[0].split(","))
        return used / total > GPU_BACKOFF_USED_FRACTION
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return False


def run_compression_subprocess(
    recipe: dict[str, Any],
    phase: str,
    recipes_dir: Path,
) -> dict[str, Any]:
    """Run one compression candidate in an isolated subprocess to prevent VRAM leaks."""
    recipes_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = recipes_dir / f"{recipe['id']}.json"
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")

    run_json = Path("studies/program-v1/measurements") / recipe["id"] / phase / "run.json"
    wall = 0.0
    if not run_json.exists():
        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.compression.study",
                "run",
                "--recipe",
                str(recipe_path),
                "--phase",
                phase,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        wall = time.perf_counter() - t0
        if proc.returncode != 0 or not run_json.exists():
            return {
                "candidate": recipe["id"],
                "phase": phase,
                "status": "runtime_failure",
                "error": (proc.stderr or proc.stdout)[-400:],
                "wall_seconds": round(wall, 2),
            }

    data = json.loads(run_json.read_text())
    summary = data.get("summary") or {}
    wall = wall or float(data.get("wall_seconds", 0.0))

    # Measure resident tensor storage bytes via --phase size if not yet measured
    resident_bytes = summary.get("resident_tensor_storage_bytes")
    if resident_bytes is None and phase in {"smoke", "development"}:
        size_json = Path("studies/program-v1/measurements") / recipe["id"] / "size" / "run.json"
        if not size_json.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluation.compression.study",
                    "run",
                    "--recipe",
                    str(recipe_path),
                    "--phase",
                    "size",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        if size_json.exists():
            size_data = json.loads(size_json.read_text())
            resident_bytes = (size_data.get("summary") or {}).get("resident_tensor_storage_bytes")

    return {
        "candidate": recipe["id"],
        "phase": phase,
        "status": data.get("status", "ok"),
        "successes": int(summary.get("successes", 0)),
        "episodes": int(summary.get("episodes", 0)),
        "resident_bytes": float(resident_bytes or B16_RESIDENT_BYTES),
        "wall_seconds": round(wall, 2),
    }
