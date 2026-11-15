"""Bubblewrap logical-workspace sandbox and headless `pi` agent harness."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PI_MODEL = "opencode-go/deepseek-v4.1-flash"
SANDBOX_WORK = "/tmp/work"
SANDBOX_EVIDENCE = "/tmp/evidence"


class SandboxTimeoutError(TimeoutError):
    """Raised when a sandboxed agent exceeds its wall-clock timeout."""


@dataclass(frozen=True)
class AgentRunResult:
    """Outcome of one sandboxed discovery or policy-development agent invocation."""

    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    sandboxed: bool
    logs_dir: str


def resolve_pi_executable() -> str:
    """Return the `pi` CLI path or raise RuntimeError if unavailable."""
    found = shutil.which("pi")
    if found:
        return str(Path(found).resolve())
    fallback = Path.home() / ".bun/bin/pi"
    if fallback.exists():
        return str(fallback.resolve())
    raise RuntimeError("pi CLI not found in PATH or ~/.bun/bin/pi")


def pi_argv(
    *,
    model: str | None = None,
    tools: str = "read,write,edit,bash",
    thinking: str | None = None,
    prompt_arg: str = (
        f"Read {SANDBOX_EVIDENCE}/TASK.md in full, inspect all files in {SANDBOX_EVIDENCE} "
        f"and {SANDBOX_WORK}, and follow every instruction exactly."
    ),
) -> list[str]:
    """Build deterministic, extension-free headless `pi` arguments."""
    exe = resolve_pi_executable()
    chosen_model = (model or os.environ.get("RSI_PI_MODEL") or DEFAULT_PI_MODEL).strip()
    argv = [
        exe,
        "--no-extensions",
        "--no-mcp",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--no-session",
        "--print",
        "--model",
        chosen_model,
    ]
    if thinking:
        argv.extend(["--thinking", thinking])
    if tools:
        argv.extend(["--tools", tools])
    else:
        argv.append("--no-tools")
    argv.extend(["-p", prompt_arg])
    return argv


def build_bwrap_argv(
    inner_argv: list[str],
    *,
    workspace_dir: Path,
    evidence_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
    mount_pi_agent: bool = True,
    mount_venv: bool = True,
    extra_ro_binds: dict[Path, str] | None = None,
) -> list[str]:
    """Wrap `inner_argv` in bubblewrap isolating workspace + read-only evidence from the repo."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap (bwrap) is required for isolated agent execution")

    workspace = Path(workspace_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    repo = Path(repo_root).resolve()

    cmd = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        str(repo),
        "--bind",
        str(workspace),
        SANDBOX_WORK,
        "--chdir",
        SANDBOX_WORK,
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "CUDA_VISIBLE_DEVICES",
        "",
    ]

    if evidence_dir is not None:
        ev = Path(evidence_dir).resolve()
        cmd.extend(["--ro-bind", str(ev), SANDBOX_EVIDENCE])

    if extra_ro_binds:
        for src, dst in extra_ro_binds.items():
            cmd.extend(["--ro-bind", str(Path(src).resolve()), dst])

    path_prefix: list[str] = []
    resolved_inner = list(inner_argv)
    if mount_venv:
        venv = (repo / ".venv").resolve()
        if venv.is_dir():
            cmd.extend(
                [
                    "--ro-bind",
                    str(venv),
                    "/tmp/venv",
                    "--setenv",
                    "VIRTUAL_ENV",
                    "/tmp/venv",
                ]
            )
            path_prefix.append("/tmp/venv/bin")
            if resolved_inner:
                # Note: do not resolve symlinks on python binary (venv python -> /usr/bin/python3.12)
                raw_exe = Path(resolved_inner[0]).absolute()
                try:
                    rel = raw_exe.relative_to(venv.absolute())
                    resolved_inner[0] = f"/tmp/venv/{rel.as_posix()}"
                except ValueError:
                    pass

    if mount_pi_agent:
        pi_dir = Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi/agent")).resolve()
        cmd.extend(["--dir", "/tmp/pi-agent", "--setenv", "PI_CODING_AGENT_DIR", "/tmp/pi-agent"])
        for fname in ("auth.json", "models.json", "settings.json"):
            fpath = pi_dir / fname
            if fpath.is_file():
                cmd.extend(["--ro-bind", str(fpath), f"/tmp/pi-agent/{fname}"])

    if path_prefix:
        base_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        cmd.extend(["--setenv", "PATH", ":".join([*path_prefix, base_path])])

    cmd.append("--")
    cmd.extend(resolved_inner)
    return cmd


def run_sandboxed_agent(
    *,
    prompt_text: str,
    workspace_dir: Path,
    evidence_dir: Path,
    logs_dir: Path,
    agent_cmd: str | list[str] | None = None,
    model: str | None = None,
    tools: str = "read,write,edit,bash",
    thinking: str | None = None,
    timeout: float = 240.0,
    use_bwrap: bool = True,
    repo_root: Path = REPO_ROOT,
    extra_ro_binds: dict[Path, str] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> AgentRunResult:
    """Execute a discovery or policy-dev agent inside an isolated logical folder (`bwrap`)."""
    workspace = Path(workspace_dir).resolve()
    evidence = Path(evidence_dir).resolve()
    logs = Path(logs_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    task_file = evidence / "TASK.md"
    task_file.write_text(prompt_text, encoding="utf-8")

    is_pi = agent_cmd is None
    if agent_cmd is None:
        inner_argv = pi_argv(model=model, tools=tools, thinking=thinking)
    elif isinstance(agent_cmd, str):
        inner_argv = shlex.split(agent_cmd)
    else:
        inner_argv = list(agent_cmd)

    if use_bwrap:
        auto_binds: dict[Path, str] = dict(extra_ro_binds or {})
        for arg in inner_argv[1:]:
            try:
                cand = Path(arg)
            except (TypeError, ValueError):
                continue
            if not cand.is_absolute() or not cand.is_file():
                continue
            try:
                resolved = cand.resolve()
            except OSError:
                continue
            repo_resolved = Path(repo_root).resolve()
            if resolved == repo_resolved or repo_resolved in resolved.parents:
                continue
            if workspace in resolved.parents or evidence in resolved.parents:
                continue
            if resolved.parts[:2] == ("/", "tmp") and len(resolved.parts) > 3:
                top = Path(*resolved.parts[:3])
                if top.is_dir() and top not in auto_binds:
                    auto_binds[top] = str(top)
            elif resolved.parent.is_dir():
                parent = resolved.parent
                if parent not in auto_binds:
                    auto_binds[parent] = str(parent)
        argv = build_bwrap_argv(
            inner_argv,
            workspace_dir=workspace,
            evidence_dir=evidence,
            repo_root=repo_root,
            mount_pi_agent=is_pi,
            mount_venv=True,
            extra_ro_binds=auto_binds or None,
        )
        run_cwd = workspace
    else:
        argv = inner_argv
        run_cwd = workspace

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if not use_bwrap:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["RSI_WORKSPACE_DIR"] = str(workspace)
        env["RSI_EVIDENCE_DIR"] = str(evidence)
    if env_overrides:
        env.update(env_overrides)

    meta: dict[str, Any] = {
        "argv": argv,
        "cwd": str(run_cwd),
        "workspace_dir": str(workspace),
        "evidence_dir": str(evidence),
        "use_bwrap": use_bwrap,
        "is_pi": is_pi,
    }
    (logs / "command.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    started = time.monotonic()
    out_path = logs / "stdout.log"
    err_path = logs / "stderr.log"
    stdin_payload = None if is_pi else prompt_text

    with out_path.open("w", encoding="utf-8") as fout, err_path.open("w", encoding="utf-8") as ferr:
        proc = subprocess.Popen(
            argv,
            cwd=run_cwd,
            stdin=subprocess.PIPE if stdin_payload is not None else subprocess.DEVNULL,
            stdout=fout,
            stderr=ferr,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            if stdin_payload is not None and proc.stdin is not None:
                try:
                    proc.stdin.write(stdin_payload)
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            while proc.poll() is None:
                if time.monotonic() - started > timeout:
                    raise SandboxTimeoutError(
                        f"Sandboxed agent timed out after {timeout:.1f}s (logs: {logs})"
                    )
                time.sleep(0.1)
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=3)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
            else:
                proc.wait()

    elapsed = round(time.monotonic() - started, 3)
    return AgentRunResult(
        returncode=int(proc.returncode or 0),
        stdout=out_path.read_text(encoding="utf-8", errors="replace"),
        stderr=err_path.read_text(encoding="utf-8", errors="replace"),
        elapsed_seconds=elapsed,
        sandboxed=use_bwrap,
        logs_dir=str(logs),
    )
