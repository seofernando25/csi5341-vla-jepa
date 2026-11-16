"""One discovery or policy-development agent run in the allowlist sandbox, with its path-free records."""

from __future__ import annotations

import json
import shlex
import time
from pathlib import Path
from typing import Any

from rsi.agents.command import agent_cli_version, agent_command
from rsi.agents.result import AgentRunResult, agent_result_payload, write_agent_result
from rsi.agents.spec import AgentSpec, load_agent_spec
from rsi.paths import REPO_ROOT, SANDBOX_REPO, STUDY_DIR
from rsi.sandbox.bwrap import build_bwrap_argv, study_protocol
from rsi.sandbox.process import SandboxTimeoutError, host_path_env, read_all, read_tail, run_process

COMMAND_FIELDS = (
    "backend",
    "model",
    "effort",
    "tools",
    "cli_version",
    "returncode",
    "network",
    "use_bwrap",
)


def _write_command(logs: Path, meta: dict[str, Any]) -> None:
    """Write the path-free command record: backend identity, exit code and sandbox mode."""
    record = {name: meta.get(name) for name in COMMAND_FIELDS}
    (Path(logs) / "command.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _finish_run(
    logs: Path,
    meta: dict[str, Any],
    *,
    backend: str,
    spec: AgentSpec | None,
    returncode: int,
    out_path: Path,
) -> dict[str, Any]:
    """Record the exit code, then write and read back the parsed-result sidecar."""
    _write_command(logs, {**meta, "returncode": returncode})
    payload = agent_result_payload(
        backend,
        model=spec.model if spec else None,
        effort=spec.effort if spec else None,
        returncode=returncode,
        stdout=read_all(out_path),
    )
    return write_agent_result(logs, payload)


def run_sandboxed_agent(
    *,
    prompt_text: str,
    workspace_dir: Path,
    evidence_dir: Path,
    logs_dir: Path,
    agent_cmd: str | list[str] | None = None,
    agent: AgentSpec | None = None,
    timeout: float = 240.0,
    use_bwrap: bool = True,
    repo_root: Path = REPO_ROOT,
    extra_ro_binds: dict[Path, str] | None = None,
    writable_dirs: dict[Path, str] | None = None,
    env_overrides: dict[str, str] | None = None,
    network: bool = False,
    gpu: bool = False,
    source_ro: dict[Path, str] | None = None,
    gpu_slots_dir: Path | None = None,
    study_dir: Path | None = None,
) -> AgentRunResult:
    """Execute a discovery or policy-dev agent in an allowlist sandbox (`bwrap`).

    Built-in backends (`agent`, default: the protocol's agent) always get network access and
    require bwrap. A custom `agent_cmd` (tests, wrappers) gets network only with `network=True`
    and may run without bwrap via `use_bwrap=False`; it receives the prompt on stdin.
    `study_dir` selects the study whose protocol names the default agent and the protocol the
    workspace's own `tools/check.py` evaluates against.
    """
    custom = agent_cmd is not None
    if not custom and not use_bwrap:
        raise RuntimeError("the built-in agents require bwrap; use_bwrap=False needs agent_cmd")
    workspace = Path(workspace_dir).resolve()
    evidence = Path(evidence_dir).resolve()
    logs = Path(logs_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    (evidence / "TASK.md").write_text(prompt_text, encoding="utf-8")

    spec = None if custom else (agent or load_agent_spec(study_dir or STUDY_DIR))
    backend_binds: dict[Path, str] = {}
    backend_env: dict[str, str] = {}
    if spec is None:
        if isinstance(agent_cmd, str):
            inner_argv = shlex.split(agent_cmd)
        else:
            inner_argv = list(agent_cmd or [])
    else:
        inner_argv, backend_binds, backend_env = agent_command(spec)
    backend = spec.backend if spec else "custom"
    # The agent's `tools/check.py` runs the same evaluator; point it at this study's protocol.
    protocol = study_protocol(study_dir, repo_root)
    study_env = {"VLA_PROTOCOL": f"{SANDBOX_REPO}/{protocol}"} if protocol else {}

    if use_bwrap:
        argv = build_bwrap_argv(
            inner_argv,
            workspace_dir=workspace,
            evidence_dir=evidence,
            repo_root=repo_root,
            mount_venv=True,
            extra_ro_binds={**backend_binds, **(extra_ro_binds or {})},
            writable_dirs=writable_dirs,
            network=network or not custom,
            env_overrides={**backend_env, **study_env, **(env_overrides or {})},
            gpu=gpu,
            gpu_slots_dir=gpu_slots_dir,
            source_ro=source_ro,
        )
        env = host_path_env()
    else:
        argv = inner_argv
        env = {
            **host_path_env(),
            "HOME": str(workspace),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "RSI_WORKSPACE_DIR": str(workspace),
            "RSI_EVIDENCE_DIR": str(evidence),
            **study_env,
            **(env_overrides or {}),
        }

    # `network` and `use_bwrap` are the sandbox mode the watch page reports; no argv or paths.
    meta: dict[str, Any] = {
        "backend": backend,
        "model": spec.model if spec else None,
        "effort": spec.effort if spec else None,
        "tools": spec.tools if spec else None,
        "cli_version": agent_cli_version(spec.backend) if spec else None,
        "returncode": None,
        "network": network or not custom,
        "use_bwrap": use_bwrap,
    }
    _write_command(logs, meta)

    out_path = logs / "stdout.log"
    err_path = logs / "stderr.log"
    started = time.monotonic()
    try:
        with (
            out_path.open("w", encoding="utf-8") as fout,
            err_path.open("w", encoding="utf-8") as ferr,
        ):
            returncode = run_process(
                argv,
                cwd=workspace,
                env=env,
                stdout=fout,
                stderr=ferr,
                stdin_payload=prompt_text if custom else None,
                timeout=timeout,
                cpu_seconds=None,
                memory_bytes=None,
            )
    except SandboxTimeoutError:
        # The killed run still spent money; keep its partial usage for the loop's budget guard.
        _finish_run(
            logs,
            meta,
            backend=backend,
            spec=spec,
            returncode=-1,
            out_path=out_path,
        )
        raise
    elapsed = round(time.monotonic() - started, 3)
    side = _finish_run(
        logs, meta, backend=backend, spec=spec, returncode=returncode, out_path=out_path
    )
    return AgentRunResult(
        returncode=returncode,
        stdout=read_tail(out_path),
        stderr=read_tail(err_path),
        elapsed_seconds=elapsed,
        sandboxed=use_bwrap,
        logs_dir=str(logs),
        backend=backend,
        model=spec.model if spec else None,
        effort=spec.effort if spec else None,
        num_turns=side["num_turns"],
        cost_usd=side["cost_usd"],
        is_error=side["is_error"],
        text=str(side.get("text") or ""),
    )
