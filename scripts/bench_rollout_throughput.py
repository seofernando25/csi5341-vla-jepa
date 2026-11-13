"""Benchmark serial vs parallel LIBERO rollout throughput (same smoke protocol).

Runs one 10-trial development smoke serially (with stage breakdown) and once per
worker count in parallel, then reports trials/min, speedup, per-trial success
equivalence, and GPU peaks. All outputs go under outputs/evaluation/parallel;
nothing is written to studies/.

Example: uv run python scripts/bench_rollout_throughput.py --variant B16 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> tuple[str, float]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc.stdout + proc.stderr, wall


def read_episodes(result_dir: Path) -> list[dict]:
    with (result_dir / "episodes.csv").open() as handle:
        return list(csv.DictReader(handle))


def gpu_used_gb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.splitlines()
        return int(out[0].strip()) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="B16")
    parser.add_argument("--workers", type=int, nargs="+", default=[4])
    parser.add_argument("--tasks", type=int, nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--gpu-limit-gb", type=float, default=20.0)
    args = parser.parse_args()

    before = gpu_used_gb()
    if before is not None and before > args.gpu_limit_gb:
        raise SystemExit(f"GPU guard: {before:.1f} GB used > {args.gpu_limit_gb:.1f} GB limit")

    serial_out, serial_wall = run(
        [
            sys.executable,
            "-m",
            "evaluation.parallel_rollout",
            "profile",
            "--variant",
            args.variant,
            *(["--tasks", *[str(t) for t in args.tasks]] if args.tasks else []),
            "--episodes",
            str(args.episodes),
        ]
    )
    n_tasks = len(args.tasks) if args.tasks else 10
    trials = n_tasks * args.episodes
    serial_tpm = trials / serial_wall * 60
    print(serial_out)
    print(f"serial: {trials} trials in {serial_wall:.1f}s = {serial_tpm:.2f} trials/min")

    for workers in args.workers:
        out_dir = ROOT / f"outputs/evaluation/parallel/bench-{args.variant}-w{workers}"
        par_out, par_wall = run(
            [
                sys.executable,
                "-m",
                "evaluation.parallel_rollout",
                "run",
                "--variant",
                args.variant,
                "--workers",
                str(workers),
                *(["--tasks", *[str(t) for t in args.tasks]] if args.tasks else []),
                "--episodes",
                str(args.episodes),
                "--out",
                str(out_dir),
            ]
        )
        print(par_out)
        par_tpm = trials / par_wall * 60
        peak = gpu_used_gb()
        print(
            f"parallel w={workers}: {trials} trials in {par_wall:.1f}s = {par_tpm:.2f} "
            f"trials/min (x{par_tpm / serial_tpm:.2f} vs serial, gpu_now={peak} GB)"
        )


if __name__ == "__main__":
    main()
