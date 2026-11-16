"""The Claude Code backend: headless argv with the binary and credentials as the only host binds."""

from __future__ import annotations

import shutil
from pathlib import Path

from rsi.agents.spec import TASK_PROMPT, AgentSpec, Command
from rsi.paths import SANDBOX_CLAUDE_CREDENTIALS, SANDBOX_EVIDENCE

CLAUDE_TOOL_NAMES = {"read": "Read", "write": "Write", "edit": "Edit", "bash": "Bash"}


def _claude_tools(tools: str) -> str:
    names = [name.strip() for name in tools.split(",") if name.strip()]
    if not names or any(name not in CLAUDE_TOOL_NAMES for name in names):
        allowed = ",".join(CLAUDE_TOOL_NAMES)
        raise ValueError(f"claude backend allows only {allowed}; got {tools!r}")
    return ",".join(CLAUDE_TOOL_NAMES[name] for name in names)


def claude_command(spec: AgentSpec, prompt_arg: str = TASK_PROMPT) -> Command:
    """Return Claude Code headless argv; the binary and credentials are the only host binds."""
    found = shutil.which("claude")
    if not found:
        raise RuntimeError("claude CLI not found in PATH; required for the claude agent")
    exe = Path(found).resolve()
    creds = Path.home() / ".claude" / ".credentials.json"
    if not creds.is_file():
        raise RuntimeError(f"claude credentials file missing at {creds}; log in with claude")
    argv = [str(exe), "-p", prompt_arg, "--model", spec.model]
    if spec.effort:
        argv.extend(["--effort", spec.effort])
    claude_tools = _claude_tools(spec.tools)
    # acceptEdits covers file edits only; headless Bash needs an explicit allow rule.
    bash_allow = ["--allowedTools", "Bash(*)"] if "Bash" in claude_tools.split(",") else []
    argv.extend(
        [
            "--tools",
            claude_tools,
            "--permission-mode",
            "acceptEdits",
            *bash_allow,
            "--add-dir",
            SANDBOX_EVIDENCE,
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
    )
    binds = {exe: str(exe), creds: SANDBOX_CLAUDE_CREDENTIALS}
    return argv, binds, {}
