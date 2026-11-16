"""Agent stream logs to watch events: incremental, cached reads of partial lines.

Two transcript dialects are understood: Claude Code `stream-json` (`system`, `assistant`,
`user`, `result`, handled here) and `pi --mode json` (`pi_dialect`). Any other event type is
ignored, never rendered as raw JSON.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from rsi.watch.events import (
    CREDENTIAL_OUTPUT,
    LogStream,
    append_event,
    make_event,
    result_text,
    tool_call,
)
from rsi.watch.pi_dialect import PI_TYPES, emit_pi

CACHE_LIMIT = 64
_CACHE: dict[str, LogStream] = {}
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
            stream = _CACHE[key] = LogStream()
        _advance(stream, Path(log_path), finished)
        events = stream.events[max(after, 0) :]
        return events, len(stream.events), finished or stream.result_seen


def _advance(stream: LogStream, path: Path, finished: bool) -> None:
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


def _emit(stream: LogStream, line: str) -> None:
    text = line.strip()
    if not text:
        return
    try:
        obj = json.loads(text)
    except ValueError:
        append_event(stream, make_event("text", text=text))
        return
    if not isinstance(obj, dict):
        append_event(stream, make_event("text", text=text))
        return
    kind = obj.get("type")
    if kind in PI_TYPES:
        emit_pi(stream, obj)
    elif kind == "system":
        _emit_system(stream, obj)
    elif kind == "assistant":
        _emit_assistant(stream, obj)
    elif kind == "user":
        _emit_user(stream, obj)
    elif kind == "result" or (kind is None and "result" in obj):
        stream.result_seen = True
        append_event(stream, _result_event(obj))


def _emit_system(stream: LogStream, obj: dict[str, Any]) -> None:
    subtype = obj.get("subtype")
    if subtype == "init":
        tools = obj.get("tools") if isinstance(obj.get("tools"), list) else []
        text = f"session start model={obj.get('model')} tools={','.join(map(str, tools))}"
        append_event(stream, make_event("system", text=text))
    elif subtype not in ("thinking_tokens", None):
        append_event(stream, make_event("system", text=str(subtype)))


def _blocks(obj: dict[str, Any]) -> list[dict[str, Any]]:
    message = obj.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _emit_assistant(stream: LogStream, obj: dict[str, Any]) -> None:
    for block in _blocks(obj):
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            append_event(stream, make_event("text", text=block["text"]))
        elif btype == "thinking" and block.get("thinking"):
            append_event(stream, make_event("thinking", text=str(block["thinking"])))
        elif btype == "tool_use":
            tool_call(stream, block.get("id"), str(block.get("name") or ""), block.get("input"))


def _emit_user(stream: LogStream, obj: dict[str, Any]) -> None:
    for block in _blocks(obj):
        if block.get("type") != "tool_result":
            continue
        tool_id = block.get("tool_use_id")
        name, secret = stream.calls.get(str(tool_id), (None, False))
        output = CREDENTIAL_OUTPUT if secret else result_text(block.get("content"))
        append_event(
            stream,
            make_event(
                "tool_result",
                tool=name,
                tool_use_id=tool_id,
                output=output,
                is_error=block.get("is_error") is True,
            ),
        )


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
    return make_event(
        "result",
        text=str(text) if text is not None else None,
        is_error=obj.get("is_error") is True,
        meta=meta,
    )
