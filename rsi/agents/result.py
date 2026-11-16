"""The record of one agent run: its result, the `agent_result.json` sidecar and manifest totals."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi.agents.command import agent_cli_version
from rsi.agents.output import parse_agent_output
from rsi.agents.spec import AgentSpec
from rsi.sandbox.process import SandboxOutputError

MAX_AGENT_TEXT_CHARS = 65_536
AGENT_RESULT = "agent_result.json"


@dataclass(frozen=True)
class AgentRunResult:
    """Outcome of one sandboxed discovery or policy-development agent invocation."""

    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float | None
    sandboxed: bool
    logs_dir: str
    backend: str
    model: str | None
    effort: str | None
    num_turns: int | None
    cost_usd: float | None
    is_error: bool | None = None
    text: str = ""


def _clip(text: str, limit: int = MAX_AGENT_TEXT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit]


def agent_result_payload(
    backend: str,
    *,
    model: str | None,
    effort: str | None,
    returncode: int,
    stdout: str,
) -> dict[str, Any]:
    """Compact, path-free record of one agent run, parsed from the whole stdout stream.

    `stdout` must be the complete transcript: a pi stream puts its per-turn usage before the
    final (large) message, so parsing only the tail loses cost and turn counts.
    """
    parsed = parse_agent_output(backend, stdout)
    return {
        "backend": backend,
        "model": model,
        "effort": effort,
        "returncode": returncode,
        "cost_usd": parsed["cost_usd"],
        "num_turns": parsed["num_turns"],
        "is_error": parsed["is_error"],
        "text": _clip(parsed["text"]),
    }


def read_agent_result(logs_dir: Path) -> dict[str, Any] | None:
    """The `agent_result.json` sidecar of one run, or None when it is missing or malformed."""
    try:
        data = json.loads((Path(logs_dir) / AGENT_RESULT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("returncode"), int):
        return None
    for key in ("cost_usd", "num_turns"):
        value = data.get(key)
        if value is not None and (not isinstance(value, int | float) or isinstance(value, bool)):
            return None
    return data


def write_agent_result(logs_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Write the sidecar and return the values read back from it."""
    path = Path(logs_dir) / AGENT_RESULT
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    stored = read_agent_result(logs_dir)
    if stored is None:
        raise SandboxOutputError(f"{AGENT_RESULT} is unreadable after writing")
    return stored


def timed_out_result(logs_dir: Path, *, sandboxed: bool) -> AgentRunResult | None:
    """Cost and turns recovered from a timed-out run's sidecar; None when it wrote none."""
    data = read_agent_result(logs_dir)
    if data is None:
        return None
    return AgentRunResult(
        returncode=int(data["returncode"]),
        stdout="",
        stderr="",
        elapsed_seconds=None,
        sandboxed=sandboxed,
        logs_dir=str(logs_dir),
        backend=str(data.get("backend") or "custom"),
        model=data.get("model"),
        effort=data.get("effort"),
        num_turns=data.get("num_turns"),
        cost_usd=data.get("cost_usd"),
        is_error=data.get("is_error"),
        text=str(data.get("text") or ""),
    )


def agent_record(spec: AgentSpec | None, results: Sequence[AgentRunResult]) -> dict[str, Any]:
    """Manifest block: backend identity plus summed cost and turns that the backend reported."""
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    turns = [r.num_turns for r in results if r.num_turns is not None]
    return {
        "backend": spec.backend if spec else "custom",
        "model": spec.model if spec else None,
        "effort": spec.effort if spec else None,
        "tools": spec.tools if spec else None,
        "cli_version": agent_cli_version(spec.backend) if spec else None,
        "calls": len(results),
        "num_turns": sum(turns) if turns else None,
        "cost_usd": round(sum(costs), 6) if costs else None,
    }
