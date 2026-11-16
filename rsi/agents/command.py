"""Backend dispatch: the argv, binds and env for one agent spec, and the CLI version it reports."""

from __future__ import annotations

import functools
import shutil
import subprocess

from rsi.agents.claude import claude_command
from rsi.agents.pi import pi_command, require_pi_auth, resolve_pi_executable
from rsi.agents.spec import TASK_PROMPT, AgentSpec, Command


def agent_command(spec: AgentSpec, prompt_arg: str = TASK_PROMPT) -> Command:
    """Dispatch to the backend named by `spec`."""
    if spec.backend == "claude":
        return claude_command(spec, prompt_arg)
    if spec.backend == "pi":
        require_pi_auth(spec)
        return pi_command(spec, prompt_arg)
    raise ValueError(f"unknown agent backend {spec.backend!r}")


@functools.cache
def agent_cli_version(backend: str) -> str | None:
    """Host-side `--version` string of the backend CLI, captured once per process."""
    try:
        exe = shutil.which("claude") if backend == "claude" else resolve_pi_executable()
        if not exe:
            return None
        out = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None
