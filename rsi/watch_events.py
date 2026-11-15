"""Agent stream logs to watch events: incremental reads, partial lines and redaction.

Two transcript dialects are understood: Claude Code `stream-json` (`system`, `assistant`,
`user`, `result`) and `pi --mode json` (`session`, `message_update`, `tool_execution_*`,
`turn_end`, `agent_end`). Any other event type is ignored, never rendered as raw JSON.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TEXT_CAP = 20_000
OUTPUT_CAP = 8_000
INPUT_CAP = 4_000
CREDENTIAL_OUTPUT = "[redacted: credential file]"
CREDENTIAL_TARGET = re.compile(
    r"credentials|auth\.json|\.env\b|\.envrc|id_rsa|id_ed25519|id_ecdsa|\.pem\b|\.p12\b|\.pfx\b"
    r"|\.netrc|secrets\.ya?ml|token\.json|hosts\.yml|\.ssh\b",
    re.IGNORECASE,
)
CACHE_LIMIT = 64
PLACEHOLDER = re.compile(r"(?:\[object Object\])+")
PI_TYPES = frozenset(
    {
        "session",
        "agent_start",
        "turn_start",
        "message_start",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
        "agent_settled",
        "tool_execution_start",
        "tool_execution_end",
    }
)
_TOOL_NAMES = {"read": "Read", "write": "Write", "edit": "Edit", "bash": "Bash"}
_KEY_VALUE = re.compile(
    r"""(?<![A-Za-z0-9_])((?:"|')?(?:accessToken|refreshToken|access_token|refresh_token"""
    r"""|api_key|apiKey|x-api-key|x_api_key|authorization|password|passwd|secret|token"""
    r"""|client_secret|private_key|session_token|aws_secret_access_key|anthropic_api_key"""
    r"""|openai_api_key|mistral_api_key|hf_token|github_token|gh_token)(?:"|')?\s*[:=]\s*)"""
    r"""(?:Bearer \[redacted\]|(?:Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=-]+"""
    r"""|"[^"]*"|'[^']*'|[^\s,}\]"']+)""",
    re.IGNORECASE,
)
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Key/value rules run first so a token rule cannot leave a half-matched value behind.
    (_KEY_VALUE, r'\1"[redacted]"'),
    (re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [redacted]"),
    (re.compile(r"sk-ant-[A-Za-z0-9_-]+"), "sk-ant-[redacted]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[redacted: aws key]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[redacted: github token]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "[redacted: hf token]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*"), "[redacted: jwt]"),
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
            re.DOTALL,
        ),
        "[redacted: private key]",
    ),
    (re.compile(r"(?<=://)[^\s/:@]+:[^\s/@]+(?=@)"), "[redacted]"),
)


def redact(text: str) -> str:
    """Mask provider keys, bearer tokens, JWTs, private keys and credential-style key values."""
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def redact_obj(obj: Any) -> Any:
    """Apply `redact` to every string inside a JSON-like object."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    return obj


def clean_text(text: str) -> str:
    """Drop the JS `[object Object]` placeholder pi emits for non-string content.

    Only the literal placeholder is removed, so legitimate prose around it survives.
    """
    return PLACEHOLDER.sub("", text)


def pi_text(message: Any) -> str:
    """Joined text blocks of one pi message, skipping placeholder-only harness artifacts."""
    content = message.get("content") if isinstance(message, dict) else None
    parts: list[str] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and PLACEHOLDER.fullmatch(text.strip()):
            continue
        if isinstance(text, str) and text.strip():
            parts.append(clean_text(text))
    return "".join(parts).strip()


def pi_cost(obj: Any) -> float | None:
    """USD cost reported on a pi `message_end`/`message_update`/`turn_end` object, if any."""
    if not isinstance(obj, dict):
        return None
    message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else message.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    total = cost.get("total") if isinstance(cost, dict) else None
    return float(total) if isinstance(total, int | float) and not isinstance(total, bool) else None


def _event(kind: str, **fields: Any) -> dict[str, Any]:
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
class _Stream:
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


_CACHE: dict[str, _Stream] = {}
_LOCK = threading.Lock()


def read_events(
    log_path: Path, after: int, *, finished: bool = False
) -> tuple[list[dict], int, bool]:
    """Return events with index >= `after`, the next index and whether the process is done.

    `finished` tells the reader that the producing process has exited, so a final line
    without a newline is consumed. The process also counts as done once its `result` line is seen.
    """
    key = str(Path(log_path))
    with _LOCK:
        stream = _CACHE.get(key)
        if stream is None:
            if len(_CACHE) >= CACHE_LIMIT:
                _CACHE.pop(next(iter(_CACHE)))
            stream = _CACHE[key] = _Stream()
        _advance(stream, Path(log_path), finished)
        events = stream.events[max(after, 0) :]
        return events, len(stream.events), finished or stream.result_seen


def _advance(stream: _Stream, path: Path, finished: bool) -> None:
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size < stream.offset:
        stream.reset()
    if size == stream.offset:
        return
    with path.open("rb") as fh:
        fh.seek(stream.offset)
        data = fh.read()
    parts = data.split(b"\n")
    tail = parts.pop()
    complete = parts
    stream.offset += sum(len(p) + 1 for p in complete)
    if finished and tail:
        complete.append(tail)
        stream.offset += len(tail)
    for raw in complete:
        _emit(stream, raw.decode("utf-8", errors="replace"))


def _append(stream: _Stream, event: dict[str, Any]) -> None:
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


def _emit(stream: _Stream, line: str) -> None:
    text = line.strip()
    if not text:
        return
    try:
        obj = json.loads(text)
    except ValueError:
        _append(stream, _event("text", text=text))
        return
    if not isinstance(obj, dict):
        _append(stream, _event("text", text=text))
        return
    kind = obj.get("type")
    if kind in PI_TYPES:
        _emit_pi(stream, obj)
    elif kind == "system":
        _emit_system(stream, obj)
    elif kind == "assistant":
        _emit_assistant(stream, obj)
    elif kind == "user":
        _emit_user(stream, obj)
    elif kind == "result" or (kind is None and "result" in obj):
        stream.result_seen = True
        _append(stream, _result_event(obj))
    # Unknown event types carry no renderable content; dropping them keeps the transcript clean.


def _emit_system(stream: _Stream, obj: dict[str, Any]) -> None:
    subtype = obj.get("subtype")
    if subtype == "init":
        tools = obj.get("tools") if isinstance(obj.get("tools"), list) else []
        text = f"session start model={obj.get('model')} tools={','.join(map(str, tools))}"
        _append(stream, _event("system", text=text))
    elif subtype not in ("thinking_tokens", None):
        _append(stream, _event("system", text=str(subtype)))


def _blocks(obj: dict[str, Any]) -> list[dict[str, Any]]:
    message = obj.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _emit_assistant(stream: _Stream, obj: dict[str, Any]) -> None:
    for block in _blocks(obj):
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            _append(stream, _event("text", text=block["text"]))
        elif btype == "thinking" and block.get("thinking"):
            _append(stream, _event("thinking", text=str(block["thinking"])))
        elif btype == "tool_use":
            _tool_call(stream, block.get("id"), str(block.get("name") or ""), block.get("input"))


def _tool_call(stream: _Stream, tool_id: Any, name: str, inp: Any) -> None:
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
    _append(
        stream,
        _event("tool_call", tool=name, tool_use_id=tool_id, input=compact),
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


def _emit_user(stream: _Stream, obj: dict[str, Any]) -> None:
    for block in _blocks(obj):
        if block.get("type") != "tool_result":
            continue
        tool_id = block.get("tool_use_id")
        name, secret = stream.calls.get(str(tool_id), (None, False))
        output = CREDENTIAL_OUTPUT if secret else _result_text(block.get("content"))
        _append(
            stream,
            _event(
                "tool_result",
                tool=name,
                tool_use_id=tool_id,
                output=output,
                is_error=block.get("is_error") is True,
            ),
        )


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        inner = content.get("content")
        if inner is not None:
            return _result_text(inner)
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("text")
        )
    return ""


def _result_event(obj: dict[str, Any]) -> dict[str, Any]:
    turns = obj.get("num_turns")
    cost = obj.get("total_cost_usd")
    duration = obj.get("duration_ms")
    meta: dict[str, Any] = {}
    if isinstance(turns, int) and not isinstance(turns, bool):
        meta["num_turns"] = turns
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        meta["cost_usd"] = float(cost)
    if isinstance(duration, int | float) and not isinstance(duration, bool):
        meta["duration_s"] = round(float(duration) / 1000.0, 3)
    text = obj.get("result")
    return _event(
        "result",
        text=str(text) if text is not None else None,
        is_error=obj.get("is_error") is True,
        meta=meta,
    )


def _emit_pi(stream: _Stream, obj: dict[str, Any]) -> None:
    """Dispatch one `pi --mode json` event; only content-bearing types produce watch events."""
    kind = obj.get("type")
    if kind in ("session", "agent_start"):
        _pi_session(stream, obj)
    elif kind == "message_update":
        _pi_update(stream, obj)
    elif kind == "tool_execution_start":
        _tool_call(stream, obj.get("toolCallId"), str(obj.get("toolName") or ""), obj.get("args"))
    elif kind == "tool_execution_end":
        _pi_tool_end(stream, obj)
    elif kind == "message_end":
        _pi_message_end(stream, obj)
    elif kind == "turn_end":
        _pi_turn_end(stream, obj)
    elif kind == "agent_end":
        _pi_agent_end(stream, obj)


def _pi_session(stream: _Stream, obj: dict[str, Any]) -> None:
    """One system event for the whole run, not one per turn."""
    if stream.session_seen:
        return
    stream.session_seen = True
    cwd = obj.get("cwd")
    text = f"session start cwd={cwd}" if obj.get("type") == "session" else "agent start"
    _append(stream, _event("system", text=text))


def _pi_update(stream: _Stream, obj: dict[str, Any]) -> None:
    event = obj.get("assistantMessageEvent")
    if not isinstance(event, dict):
        return
    etype = event.get("type")
    if etype in ("text_delta", "thinking_delta"):
        _pi_delta(stream, str(etype), event.get("delta"))
    elif etype in ("toolcall_end", "tool_call_end"):
        call = event.get("toolCall") if isinstance(event.get("toolCall"), dict) else {}
        _tool_call(stream, call.get("id"), str(call.get("name") or ""), call.get("arguments"))


def _pi_delta(stream: _Stream, etype: str, delta: Any) -> None:
    """A string delta is text; pi passes provider reasoning blocks as a list on `text_delta`."""
    if isinstance(delta, str):
        text = clean_text(delta)
        if text:
            _append(stream, _event("thinking" if etype == "thinking_delta" else "text", text=text))
        return
    for block in delta if isinstance(delta, list) else []:
        if not isinstance(block, dict) or block.get("type") != "thinking":
            continue
        inner = block.get("thinking")
        if isinstance(inner, list):
            text = "".join(str(b.get("text", "")) for b in inner if isinstance(b, dict))
        else:
            text = str(inner or "")
        text = clean_text(text)
        if text:
            _append(stream, _event("thinking", text=text))


def _pi_tool_end(stream: _Stream, obj: dict[str, Any]) -> None:
    tool_id = obj.get("toolCallId")
    name, secret = stream.calls.get(str(tool_id), (obj.get("toolName"), False))
    output = CREDENTIAL_OUTPUT if secret else _result_text(obj.get("result"))
    _append(
        stream,
        _event(
            "tool_result",
            tool=name,
            tool_use_id=tool_id,
            output=output,
            is_error=obj.get("isError") is True,
        ),
    )


def _pi_message_end(stream: _Stream, obj: dict[str, Any]) -> None:
    """Assistant messages carry the final text and the per-message cost; others are noise."""
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return
    text = pi_text(message)
    if text:
        stream.final_text = text
    cost = pi_cost(obj)
    if cost is not None:
        stream.cost_usd += cost
    if message.get("stopReason") in ("error", "aborted"):
        stream.errored = True


def _pi_turn_end(stream: _Stream, obj: dict[str, Any]) -> None:
    stream.turns += 1
    message = obj.get("message")
    if isinstance(message, dict):
        text = pi_text(message)
        if text:
            stream.final_text = text
        if message.get("stopReason") in ("error", "aborted"):
            stream.errored = True


def _pi_agent_end(stream: _Stream, obj: dict[str, Any]) -> None:
    """Close the transcript with one `result` event carrying text, turns and cost."""
    if stream.final_text is None:
        messages = obj.get("messages")
        for message in reversed(messages if isinstance(messages, list) else []):
            if isinstance(message, dict) and message.get("role") == "assistant":
                stream.final_text = pi_text(message) or None
                break
    meta: dict[str, Any] = {}
    if stream.turns:
        meta["num_turns"] = stream.turns
    if stream.cost_usd:
        meta["cost_usd"] = round(stream.cost_usd, 6)
    stream.result_seen = True
    _append(stream, _event("result", text=stream.final_text, is_error=stream.errored, meta=meta))
