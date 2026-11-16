"""Watch events and the parse state of one agent log, shared by both transcript dialects."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rsi.agents.output import clean_text
from rsi.redact import redact_obj

TEXT_CAP = 20_000
OUTPUT_CAP = 8_000
INPUT_CAP = 4_000
CREDENTIAL_OUTPUT = "[redacted: credential file]"
CREDENTIAL_TARGET = re.compile(
    r"credentials|auth\.json|\.env\b|\.envrc|id_rsa|id_ed25519|id_ecdsa|\.pem\b|\.p12\b|\.pfx\b"
    r"|\.netrc|secrets\.ya?ml|token\.json|hosts\.yml|\.ssh\b",
    re.IGNORECASE,
)
_TOOL_NAMES = {"read": "Read", "write": "Write", "edit": "Edit", "bash": "Bash"}


def make_event(kind: str, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "i": -1,
        "t": None,
        "kind": kind,
        "text": None,
        "tool": None,
        "tool_use_id": None,
        "input": None,
        "output": None,
        "truncated": False,
        "is_error": False,
        "meta": None,
    }
    base.update(fields)
    return base


@dataclass
class LogStream:
    """Parsed prefix of one log: consumed byte offset, events so far and tool-call bookkeeping."""

    offset: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    calls: dict[str, tuple[str | None, bool]] = field(default_factory=dict)
    result_seen: bool = False
    session_seen: bool = False
    turns: int = 0
    cost_usd: float = 0.0
    final_text: str | None = None
    errored: bool = False

    def reset(self) -> None:
        self.offset = 0
        self.events = []
        self.calls = {}
        self.result_seen = False
        self.session_seen = False
        self.turns = 0
        self.cost_usd = 0.0
        self.final_text = None
        self.errored = False


def append_event(stream: LogStream, event: dict[str, Any]) -> None:
    """Redact and clean first, then cap, so a secret is never cut in half and no placeholder ships."""
    event = redact_obj(event)
    event["i"] = len(stream.events)
    for key, limit in (("text", TEXT_CAP), ("output", OUTPUT_CAP)):
        value = event.get(key)
        if isinstance(value, str):
            value = clean_text(value)
            if len(value) > limit:
                value = value[:limit]
                event["truncated"] = True if key == "output" else event["truncated"]
            event[key] = value
    if isinstance(event.get("input"), dict):
        event["input"] = {
            k: clean_text(v)[:INPUT_CAP] if isinstance(v, str) else v
            for k, v in event["input"].items()
        }
    stream.events.append(event)


def tool_call(stream: LogStream, tool_id: Any, name: str, inp: Any) -> None:
    """Append one `tool_call` event unless this call id was already reported."""
    key = tool_id if isinstance(tool_id, str) else None
    if key is not None and key in stream.calls:
        return
    args = inp if isinstance(inp, dict) else {}
    path = args.get("file_path") or args.get("path")
    path = path if isinstance(path, str) else ""
    command = args.get("command") if isinstance(args.get("command"), str) else ""
    secret = bool(CREDENTIAL_TARGET.search(path) or CREDENTIAL_TARGET.search(command))
    if key is not None:
        stream.calls[key] = (name, secret)
    compact: dict[str, Any] = {"summary": _summary(name, args, path, command)}
    if path:
        compact["file_path"] = path
    if command:
        compact["command"] = command
    append_event(
        stream,
        make_event("tool_call", tool=name, tool_use_id=tool_id, input=compact),
    )


def _summary(name: str, inp: dict[str, Any], path: str, command: str) -> str:
    kind = _TOOL_NAMES.get(name.lower(), name)
    if kind == "Write":
        content = inp.get("content")
        size = len(content) if isinstance(content, str) else 0
        return f"write {path} ({size} chars)"
    if kind in ("Edit", "Read"):
        return f"{kind.lower()} {path}"
    if kind == "Bash":
        return f"bash: {command.splitlines()[0] if command else ''}"
    return f"{kind} {','.join(sorted(map(str, inp)))}"


def result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        inner = content.get("content")
        if inner is not None:
            return result_text(inner)
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("text")
        )
    return ""
