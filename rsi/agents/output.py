"""Agent transcript parsing: final text, turn count, cost and error flag from one run's stdout.

Claude emits stream-json (one object per line, final `result` line). pi emits `--mode json`
events; its cost and turn count are summed from the stream.
"""

from __future__ import annotations

import json
import re
from typing import Any

PLACEHOLDER = re.compile(r"(?:\[object Object\])+")


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


def _number(value: Any) -> float | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def _result_object(stdout: str) -> dict[str, Any] | None:
    """Final `result` object of a stream-json transcript, or the whole stdout of the old single-object form."""
    try:
        whole = json.loads(stdout)
    except ValueError:
        whole = None
    if isinstance(whole, dict) and ("result" in whole or whole.get("type") == "result"):
        return whole
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            obj = json.loads(text)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            return obj
    return None


def _pi_stream_result(stdout: str) -> dict[str, Any] | None:
    """Final text, turn count and summed cost of a `pi --mode json` transcript, or None."""
    turns = 0
    cost = 0.0
    text: str | None = None
    errored = False
    finished = False
    seen = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            obj = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        kind = obj.get("type")
        if kind == "turn_end":
            seen = True
            turns += 1
            message = obj.get("message")
            if isinstance(message, dict):
                body = pi_text(message)
                if body:
                    text = body
                if message.get("stopReason") in ("error", "aborted"):
                    errored = True
        elif kind == "message_end":
            message = obj.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                seen = True
                body = pi_text(message)
                if body:
                    text = body
                if message.get("stopReason") in ("error", "aborted"):
                    errored = True
                message_cost = pi_cost(obj)
                if message_cost is not None:
                    cost += message_cost
        elif kind == "agent_end":
            seen = finished = True
            if text is None:
                messages = obj.get("messages")
                for message in reversed(messages if isinstance(messages, list) else []):
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        text = pi_text(message) or None
                        break
    if not seen:
        return None
    return {
        "text": text or "",
        "num_turns": turns or None,
        "cost_usd": round(cost, 6) if cost else None,
        "is_error": errored if finished else None,
    }


def parse_agent_output(backend: str, stdout: str) -> dict[str, Any]:
    """Return text, num_turns, cost_usd and is_error from one agent's stdout.

    Claude emits stream-json (one object per line, final `result` line) or the old single object.
    pi emits `--mode json` events; its cost and turn count are summed from the stream.
    """
    plain: dict[str, Any] = {"text": stdout, "num_turns": None, "cost_usd": None, "is_error": None}
    if backend == "pi":
        return _pi_stream_result(stdout) or plain
    if backend != "claude":
        return plain
    obj = _result_object(stdout)
    if obj is None:
        return plain
    turns = _number(obj.get("num_turns"))
    is_error = obj.get("is_error")
    return {
        "text": str(obj.get("result") or ""),
        "num_turns": int(turns) if turns is not None else None,
        "cost_usd": _number(obj.get("total_cost_usd")),
        "is_error": is_error if isinstance(is_error, bool) else None,
    }
