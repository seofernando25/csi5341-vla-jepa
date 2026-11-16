"""The `pi --mode json` transcript dialect: sessions, deltas, tool executions, turns and the final result."""

from __future__ import annotations

from typing import Any

from rsi.agents.output import clean_text, pi_cost, pi_text
from rsi.watch.events import (
    CREDENTIAL_OUTPUT,
    LogStream,
    append_event,
    make_event,
    result_text,
    tool_call,
)

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


def emit_pi(stream: LogStream, obj: dict[str, Any]) -> None:
    """Dispatch one `pi --mode json` event; only content-bearing types produce watch events."""
    kind = obj.get("type")
    if kind in ("session", "agent_start"):
        _pi_session(stream, obj)
    elif kind == "message_update":
        _pi_update(stream, obj)
    elif kind == "tool_execution_start":
        tool_call(stream, obj.get("toolCallId"), str(obj.get("toolName") or ""), obj.get("args"))
    elif kind == "tool_execution_end":
        _pi_tool_end(stream, obj)
    elif kind == "message_end":
        _pi_message_end(stream, obj)
    elif kind == "turn_end":
        _pi_turn_end(stream, obj)
    elif kind == "agent_end":
        _pi_agent_end(stream, obj)


def _pi_session(stream: LogStream, obj: dict[str, Any]) -> None:
    """One system event for the whole run, not one per turn."""
    if stream.session_seen:
        return
    stream.session_seen = True
    cwd = obj.get("cwd")
    text = f"session start cwd={cwd}" if obj.get("type") == "session" else "agent start"
    append_event(stream, make_event("system", text=text))


def _pi_update(stream: LogStream, obj: dict[str, Any]) -> None:
    event = obj.get("assistantMessageEvent")
    if not isinstance(event, dict):
        return
    etype = event.get("type")
    if etype in ("text_delta", "thinking_delta"):
        _pi_delta(stream, str(etype), event.get("delta"))
    elif etype in ("toolcall_end", "tool_call_end"):
        call = event.get("toolCall") if isinstance(event.get("toolCall"), dict) else {}
        tool_call(stream, call.get("id"), str(call.get("name") or ""), call.get("arguments"))


def _pi_delta(stream: LogStream, etype: str, delta: Any) -> None:
    """A string delta is text; pi passes provider reasoning blocks as a list on `text_delta`."""
    if isinstance(delta, str):
        text = clean_text(delta)
        if text:
            append_event(
                stream, make_event("thinking" if etype == "thinking_delta" else "text", text=text)
            )
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
            append_event(stream, make_event("thinking", text=text))


def _pi_tool_end(stream: LogStream, obj: dict[str, Any]) -> None:
    tool_id = obj.get("toolCallId")
    name, secret = stream.calls.get(str(tool_id), (obj.get("toolName"), False))
    output = CREDENTIAL_OUTPUT if secret else result_text(obj.get("result"))
    append_event(
        stream,
        make_event(
            "tool_result",
            tool=name,
            tool_use_id=tool_id,
            output=output,
            is_error=obj.get("isError") is True,
        ),
    )


def _pi_message_end(stream: LogStream, obj: dict[str, Any]) -> None:
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


def _pi_turn_end(stream: LogStream, obj: dict[str, Any]) -> None:
    stream.turns += 1
    message = obj.get("message")
    if isinstance(message, dict):
        text = pi_text(message)
        if text:
            stream.final_text = text
        if message.get("stopReason") in ("error", "aborted"):
            stream.errored = True


def _pi_agent_end(stream: LogStream, obj: dict[str, Any]) -> None:
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
    append_event(
        stream, make_event("result", text=stream.final_text, is_error=stream.errored, meta=meta)
    )
