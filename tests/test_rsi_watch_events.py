"""Tests for `pi --mode json` transcript parsing in the watch event reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsi.watch_events import clean_text, read_events

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PI_STREAM = FIXTURES / "pi_stream.jsonl"


def _log(tmp_path: Path, *objects: dict) -> Path:
    path = tmp_path / "stdout.log"
    path.write_text("\n".join(json.dumps(o) for o in objects) + "\n", encoding="utf-8")
    return path


def test_pi_fixture_becomes_streamed_events(tmp_path: Path) -> None:
    log = tmp_path / "pi.jsonl"
    log.write_text(PI_STREAM.read_text(encoding="utf-8"), encoding="utf-8")
    events, nxt, done = read_events(log, 0)
    assert done is True and nxt == len(events)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "system" and events[0]["text"].startswith("session start")
    assert "thinking" in kinds and "tool_call" in kinds and "tool_result" in kinds
    assert kinds[-1] == "result"
    calls = [e for e in events if e["kind"] == "tool_call"]
    assert [c["tool"] for c in calls] == ["read", "write"]
    assert calls[0]["input"]["file_path"] == "/tmp/work/TASK.md"
    assert calls[0]["input"]["summary"] == "read /tmp/work/TASK.md"
    assert calls[1]["input"]["summary"].endswith("(4 chars)")
    results = [e for e in events if e["kind"] == "tool_result"]
    assert results[0]["tool"] == "read" and "PONG" in results[0]["output"]
    assert not any(e["is_error"] for e in results)
    assert events[-1]["text"] == "DONE"
    assert events[-1]["meta"] == {"num_turns": 3, "cost_usd": 0.000744}


def test_pi_reasoning_arrives_as_list_delta(tmp_path: Path) -> None:
    """pi passes provider reasoning blocks through `text_delta`; they must not become `[object Object]`."""
    delta = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "weigh it"}], "closed": True}
    ]
    log = _log(
        tmp_path,
        {"type": "session", "cwd": "/tmp/work"},
        {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": delta}},
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "thinking_delta", "delta": "plain thought"},
        },
    )
    events, _, _ = read_events(log, 0)
    assert [e["kind"] for e in events] == ["system", "thinking", "thinking"]
    assert [e["text"] for e in events[1:]] == ["weigh it", "plain thought"]
    assert "[object Object]" not in json.dumps(events)


def test_pi_system_message_end_does_not_dump_the_prompt(tmp_path: Path) -> None:
    log = _log(
        tmp_path,
        {"type": "session", "cwd": "/tmp/work"},
        {
            "type": "message_end",
            "message": {"role": "system", "content": "", "sections": {"preamble": "secret prompt"}},
        },
        {
            "type": "message_end",
            "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        },
        {
            "type": "message_end",
            "message": {"role": "toolResult", "content": [{"type": "text", "text": "x"}]},
        },
    )
    events, _, done = read_events(log, 0)
    assert [e["kind"] for e in events] == ["system"]
    assert done is False


def test_pi_unknown_event_types_are_ignored(tmp_path: Path) -> None:
    log = _log(
        tmp_path,
        {"type": "session", "cwd": "/tmp/work"},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        {"type": "agent_settled", "aborted": False},
        {"type": "something_new", "payload": {"raw": True}},
        {"type": "rate_limit_event", "rate_limit_info": {}},
    )
    events, nxt, _ = read_events(log, 0)
    assert [e["kind"] for e in events] == ["system"] and nxt == 1


def test_pi_tool_call_is_reported_once(tmp_path: Path) -> None:
    call = {"id": "c1", "name": "read", "arguments": {"path": "/tmp/evidence/TASK.md"}}
    log = _log(
        tmp_path,
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "toolcall_end", "toolCall": call},
        },
        {
            "type": "tool_execution_start",
            "toolCallId": "c1",
            "toolName": "read",
            "args": call["arguments"],
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "c1",
            "toolName": "read",
            "result": {"content": [{"type": "text", "text": "ok"}]},
            "isError": False,
        },
    )
    events, _, _ = read_events(log, 0)
    assert [e["kind"] for e in events] == ["tool_call", "tool_result"]
    assert events[0]["tool_use_id"] == "c1" and events[1]["output"] == "ok"


def test_pi_tool_error_and_credential_reads_are_marked(tmp_path: Path) -> None:
    log = _log(
        tmp_path,
        {
            "type": "message_update",
            "assistantMessageEvent": {
                "type": "toolcall_end",
                "toolCall": {
                    "id": "c2",
                    "name": "read",
                    "arguments": {"path": "/home/u/.pi/agent/auth.json"},
                },
            },
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "c2",
            "toolName": "read",
            "result": {"content": [{"type": "text", "text": '{"mistral": {"key": "abc"}}'}]},
            "isError": True,
        },
        {
            "type": "message_update",
            "assistantMessageEvent": {
                "type": "toolcall_end",
                "toolCall": {
                    "id": "c3",
                    "name": "read",
                    "arguments": {"path": "/home/u/.ssh/id_rsa"},
                },
            },
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "c3",
            "toolName": "read",
            "result": {
                "content": [{"type": "text", "text": "-----BEGIN OPENSSH PRIVATE KEY-----"}]
            },
            "isError": False,
        },
    )
    events, _, _ = read_events(log, 0)
    results = [e for e in events if e["kind"] == "tool_result"]
    assert results[0]["is_error"] is True
    assert results[0]["output"] == "[redacted: credential file]"
    assert results[1]["output"] == "[redacted: credential file]"
    assert "abc" not in json.dumps(events)


def test_pi_agent_end_closes_the_stream_without_a_result_line(tmp_path: Path) -> None:
    log = _log(
        tmp_path,
        {
            "type": "turn_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        },
        {"type": "agent_end", "messages": [], "willRetry": False},
    )
    events, _, done = read_events(log, 0)
    assert done is True and events[-1]["kind"] == "result"
    assert events[-1]["text"] == "done" and events[-1]["meta"]["num_turns"] == 1


def test_pi_aborted_run_is_flagged(tmp_path: Path) -> None:
    log = _log(
        tmp_path,
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [], "stopReason": "aborted"},
        },
        {"type": "agent_end", "messages": []},
    )
    events, _, _ = read_events(log, 0)
    assert events[-1]["kind"] == "result" and events[-1]["is_error"] is True


def test_pi_string_delta_placeholder_is_dropped(tmp_path: Path) -> None:
    """A JS-coerced delta must not reach the transcript as `[object Object]`."""
    log = _log(
        tmp_path,
        {"type": "session", "cwd": "/tmp/work"},
        {
            "type": "message_update",
            "assistantMessageEvent": {
                "type": "text_delta",
                "delta": "[object Object][object Object]",
            },
        },
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "hello [object Object] world"},
        },
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "thinking_delta", "delta": "[object Object]"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "c9",
            "toolName": "read",
            "result": {"content": [{"type": "text", "text": "out [object Object] put"}]},
            "isError": False,
        },
    )
    events, _, _ = read_events(log, 0)
    texts = [e["text"] for e in events if e["kind"] in ("text", "thinking")]
    assert texts == ["hello  world"]
    outputs = [e["output"] for e in events if e["kind"] == "tool_result"]
    assert outputs == ["out  put"]
    assert "[object Object]" not in json.dumps(events)


def test_pi_placeholder_only_message_end_text_is_skipped(tmp_path: Path) -> None:
    log = _log(
        tmp_path,
        {"type": "session", "cwd": "/tmp/work"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "[object Object][object Object][object Object]"}
                ],
            },
        },
        {"type": "agent_end", "messages": []},
    )
    events, _, done = read_events(log, 0)
    assert events[-1]["kind"] == "result" and events[-1]["text"] is None
    assert "[object Object]" not in json.dumps(events)
    assert done is True


def test_clean_text_keeps_legitimate_text() -> None:
    assert clean_text("plain") == "plain"
    assert clean_text("[object Object]") == ""
    assert clean_text("a[object Object]b") == "ab"
    assert clean_text("[object Object][object Object]") == ""
    assert clean_text("[object] [Object]") == "[object] [Object]"


@pytest.mark.parametrize("text", ["plain text line", ""])
def test_pi_plain_and_blank_lines_stay_text(tmp_path: Path, text: str) -> None:
    log = tmp_path / "plain.log"
    log.write_text(f"{text}\n\n", encoding="utf-8")
    events, _, _ = read_events(log, 0)
    assert [e["kind"] for e in events] == (["text"] if text else [])
