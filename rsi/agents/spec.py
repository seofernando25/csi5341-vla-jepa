"""Which headless agent runs: the backend, model, effort and tools a study's protocol names."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi.paths import SANDBOX_EVIDENCE, SANDBOX_WORK, STUDY_DIR
from rsi.runner import load_protocol

BACKENDS = ("claude", "pi")
DEFAULT_PI_MODEL = "mistral/deepseek-v4-1-flash-1017"
DEFAULT_BACKEND = "pi"
DEFAULT_CLAUDE_MODEL = "haiku"
DEFAULT_PI_EFFORT = "max"
DEFAULT_CLAUDE_EFFORT = "low"
DEFAULT_TOOLS = "read,write,edit,bash"
TASK_PROMPT = (
    f"Read {SANDBOX_EVIDENCE}/TASK.md in full, inspect all files in {SANDBOX_EVIDENCE} "
    f"and {SANDBOX_WORK}, and follow every instruction exactly."
)
Command = tuple[list[str], dict[Path, str], dict[str, str]]


@dataclass(frozen=True)
class AgentSpec:
    """Which headless agent runs, on which model, at which effort, with which tools."""

    backend: str
    model: str
    effort: str | None
    tools: str


def _default_model(backend: str) -> str:
    if backend == "pi":
        return os.environ.get("RSI_PI_MODEL") or DEFAULT_PI_MODEL
    return DEFAULT_CLAUDE_MODEL


def _default_effort(backend: str) -> str | None:
    if backend == "pi":
        return os.environ.get("RSI_PI_EFFORT") or DEFAULT_PI_EFFORT
    return DEFAULT_CLAUDE_EFFORT


def load_agent_spec(
    study_dir: Path = STUDY_DIR,
    *,
    backend: str | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> AgentSpec:
    """Read `dream_rsi.agent` from the protocol; overrides ignore it when the backend changes."""
    drsi = load_protocol(study_dir)["dream_rsi"]
    stored: dict[str, Any] = drsi.get("agent", {})
    proto_backend = str(stored.get("backend", DEFAULT_BACKEND))
    chosen = backend or proto_backend
    base = stored if chosen == proto_backend else {}
    return AgentSpec(
        backend=chosen,
        model=model or str(base.get("model") or _default_model(chosen)),
        effort=effort or base.get("effort") or _default_effort(chosen),
        tools=str(drsi.get("policy_dev_tools", DEFAULT_TOOLS)),
    )
