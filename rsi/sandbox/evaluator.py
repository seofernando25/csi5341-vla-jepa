"""The fixed `solution` evaluator run in the GPU sandbox on a read-only workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsi.gpu import GPU_SLOTS_DIR, gpu_slot
from rsi.paths import REPO_ROOT, SANDBOX_OUT, SANDBOX_REPO, SANDBOX_VENV, SANDBOX_WORK
from rsi.sandbox.bwrap import build_bwrap_argv, repo_source_binds, study_protocol
from rsi.sandbox.process import (
    SandboxOutputError,
    SandboxTimeoutError,
    read_agent_output,
    run_captured,
)

EVALUATOR_MODULE = "evaluation.compression.study"
EVALUATOR_PHASES = ("check", "smoke", "development")
EVALUATOR_RESULT = "result.json"
EVALUATOR_LOG = "log.txt"
MAX_RESULT_BYTES = 8 * 1024 * 1024


def _evaluator_failure(message: str) -> dict[str, Any]:
    return {"status": "evaluator_failure", "error": message[-2000:]}


def evaluator_argv(phase: str, protocol: str) -> list[str]:
    """Sandboxed `solution` evaluator argv for one phase, reading `protocol` under /tmp/repo."""
    return [
        f"{SANDBOX_VENV}/bin/python",
        "-P",
        "-m",
        EVALUATOR_MODULE,
        "solution",
        "--workspace",
        SANDBOX_WORK,
        "--phase",
        phase,
        "--out",
        SANDBOX_OUT,
        "--protocol",
        protocol,
    ]


def run_sandboxed_evaluator(
    workspace_dir: Path,
    phase: str,
    out_dir: Path,
    *,
    timeout: float,
    gpu_slots_dir: Path | None = None,
    gpu_slot_count: int = 1,
    study_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the fixed `solution` evaluator on a read-only workspace; return its parsed result.json.

    Network is off, the workspace is read-only at /tmp/work and `out_dir` is writable at /tmp/out.
    The run holds one GPU slot. `timeout` bounds the slot wait and the run separately. Any
    timeout, missing or invalid result becomes `{"status": "evaluator_failure"}`. `study_dir`
    selects the `protocol.json` the evaluator reads; without it the program-v1 study is used.
    """
    if phase not in EVALUATOR_PHASES:
        raise ValueError(f"phase must be one of {EVALUATOR_PHASES}; got {phase!r}")
    workspace = Path(workspace_dir).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in (EVALUATOR_RESULT, EVALUATOR_LOG):
        (out / stale).unlink(missing_ok=True)
    protocol = study_protocol(study_dir)
    if protocol is None:
        raise ValueError(
            f"study dir {study_dir} is not under the repo; its protocol cannot be read"
        )
    inner = evaluator_argv(phase, f"{SANDBOX_REPO}/{protocol}")
    argv = build_bwrap_argv(
        inner,
        workspace_dir=workspace,
        workspace_ro=True,
        out_dir=out,
        network=False,
        gpu=True,
        gpu_slots_dir=gpu_slots_dir,
        source_ro=repo_source_binds(REPO_ROOT, study_dir=study_dir),
    )
    try:
        with gpu_slot(Path(gpu_slots_dir or GPU_SLOTS_DIR), gpu_slot_count, timeout=timeout):
            returncode, _, stderr = run_captured(
                argv,
                workspace=out,
                timeout=timeout,
                cpu_seconds=None,
                memory_bytes=None,
            )
    except (TimeoutError, SandboxTimeoutError) as exc:
        return _evaluator_failure(str(exc))
    try:
        text = read_agent_output(out, EVALUATOR_RESULT, max_bytes=MAX_RESULT_BYTES)
        result = json.loads(text)
    except (SandboxOutputError, ValueError) as exc:
        return _evaluator_failure(f"no valid {EVALUATOR_RESULT} (rc={returncode}): {exc}\n{stderr}")
    if not isinstance(result, dict):
        return _evaluator_failure(f"{EVALUATOR_RESULT} is not a JSON object")
    return result
