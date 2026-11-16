"""Replay of one candidate policy on recorded trees, in the sandboxed replay worker under two hash seeds."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from rsi.paths import (
    REPO_ROOT,
    RSI_DIR,
    SANDBOX_EVIDENCE,
    SANDBOX_WORK,
)
from rsi.sandbox import (
    SandboxTimeoutError,
    read_agent_output,
    run_sandboxed_python,
)

RSI_LIB_ROOT = "/tmp/rsi_lib"
RSI_LIB_FILES = (
    "__init__.py",
    "api.py",
    "tree.py",
    "objective.py",
    "policy_audit.py",
    "replay_worker.py",
)
MAX_RESULT_BYTES = 4 * 1024 * 1024
MAX_TRACE_BYTES = 32 * 1024 * 1024
REPLAY_TIMEOUT_S = 120.0
REPLAY_CPU_S = 120


class PolicyReplayError(RuntimeError):
    """Raised when the sandboxed replay worker rejects a policy."""


def _rsi_ro_binds() -> dict[Path, str]:
    return {RSI_DIR / name: f"{RSI_LIB_ROOT}/rsi/{name}" for name in RSI_LIB_FILES}


def worker_failure(proc: subprocess.CompletedProcess[str]) -> str:
    """One-line reason for a worker that exited non-zero (signals name the limit hit)."""
    if proc.returncode < 0:
        return f"replay worker killed by {signal.Signals(-proc.returncode).name}"
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    return lines[-1] if lines else f"worker exited {proc.returncode}"


def _run_worker_once(
    *,
    ws: Path,
    worker_args: list[str],
    evidence: Path,
    use_bwrap: bool,
    hash_seed: str,
) -> tuple[dict[str, Any], str]:
    """Run the worker under one PYTHONHASHSEED; returns (result json, traces text)."""
    if use_bwrap:
        proc = run_sandboxed_python(
            worker_args,
            workspace_dir=ws,
            evidence_dir=evidence,
            extra_ro_binds=_rsi_ro_binds(),
            env_overrides={"PYTHONPATH": RSI_LIB_ROOT, "PYTHONHASHSEED": hash_seed},
            timeout=REPLAY_TIMEOUT_S,
            cpu_seconds=REPLAY_CPU_S,
        )
    else:
        try:
            proc = subprocess.run(
                [sys.executable, *worker_args],
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
                capture_output=True,
                text=True,
                check=False,
                timeout=REPLAY_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError(f"replay timed out after {REPLAY_TIMEOUT_S:.0f}s") from exc
    if proc.returncode != 0:
        raise PolicyReplayError(worker_failure(proc))
    result = json.loads(read_agent_output(ws, "result.json", max_bytes=MAX_RESULT_BYTES))
    traces = read_agent_output(ws, "traces.jsonl", max_bytes=MAX_TRACE_BYTES)
    return result, traces


def replay_policy(
    policy_src: Path,
    world_paths: list[Path],
    *,
    ws_dir: Path,
    evidence_dir: Path,
    max_parallelism: int,
    lambda_parallel: float,
    use_bwrap: bool,
) -> tuple[dict[str, Any], str]:
    """Replay one policy file under two hash seeds; reject it if the two runs disagree."""
    ws = Path(ws_dir)
    ws.mkdir(parents=True, exist_ok=True)
    for stale in ("result.json", "traces.jsonl", "policy.py"):
        (ws / stale).unlink(missing_ok=True)
    shutil.copyfile(policy_src, ws / "policy.py")
    ev = Path(evidence_dir).resolve()
    root = Path(SANDBOX_WORK) if use_bwrap else ws
    ev_root = Path(SANDBOX_EVIDENCE) if use_bwrap else ev
    worlds = [str(ev_root / Path(w).resolve().relative_to(ev).as_posix()) for w in world_paths]
    worker_args = [
        "-m",
        "rsi.replay_worker",
        str(root / "policy.py"),
        str(root / "result.json"),
        *worlds,
        "--max-parallelism",
        str(max_parallelism),
        "--lambda",
        repr(float(lambda_parallel)),
        "--traces",
        str(root / "traces.jsonl"),
    ]
    run = {"ws": ws, "worker_args": worker_args, "evidence": ev, "use_bwrap": use_bwrap}
    first = _run_worker_once(**run, hash_seed="0")
    second = _run_worker_once(**run, hash_seed="1")
    if first != second:
        raise PolicyReplayError("non-deterministic across hash seeds")
    return first
