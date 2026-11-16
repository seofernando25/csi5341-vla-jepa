"""Server-Sent Events for one source: `events` as its log grows, `state` every 2 s, `done` on exit."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from rsi.watch.files import log_path
from rsi.watch.source import Source, events_payload, source_finished

STREAM_TAIL_SECONDS = 0.25
STREAM_STATE_SECONDS = 2.0
STREAM_PING_SECONDS = 15.0
STREAM_MAX_SECONDS = 1800.0
STREAM_EVENT_CAP = 200


def _log_token(path: Path) -> str:
    """Short stable token for one log path, so a client can tell a rotated log from a truncated one."""
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def _sse(event: str, payload: Any, ident: int | None = None) -> bytes:
    """One Server-Sent Events frame; JSON bodies are single-line so one `data:` field suffices."""
    lines: list[str] = []
    if ident is not None:
        lines.append(f"id: {ident}")
    lines.append(f"event: {event}")
    lines.extend(f"data: {part}" for part in json.dumps(payload).splitlines() or [""])
    return ("\n".join(lines) + "\n\n").encode("utf-8")


class StreamMixin:
    """The SSE loop of the watch request handler; the handler supplies `server` and `wfile`."""

    server: Any
    wfile: Any
    _streaming: bool
    close_connection: bool

    def _stream_loop(self, study: Path, source: Source, after: int) -> None:
        token = _log_token(log_path(study, source))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self._streaming = True
        self.close_connection = True
        self._write(
            _sse(
                "hello",
                {
                    "v": 1,
                    "source": f"{source.kind}:{source.cycle_id}:{source.name}",
                    "log": token,
                    "cursor": after,
                    "interval_ms": int(STREAM_TAIL_SECONDS * 1000),
                },
                ident=after,
            )
        )
        index = after
        reason: str | None = None
        started = last_ping = time.monotonic()
        last_state = started - STREAM_STATE_SECONDS  # first state goes out immediately
        try:
            while True:
                path = log_path(study, source)
                new_token = _log_token(path)
                if new_token != token:
                    token = new_token
                    index = 0
                    self._write(_sse("reset", {"reason": "log_rotated", "cursor": 0}))
                payload = events_payload(study, source, index)
                if payload["next"] < index:
                    index = 0
                    self._write(_sse("reset", {"reason": "log_truncated", "cursor": 0}))
                    payload = events_payload(study, source, 0)
                if payload["events"]:
                    batch = payload["events"][:STREAM_EVENT_CAP]
                    index = payload["next"] - (len(payload["events"]) - len(batch))
                    self._write(_sse("events", {"events": batch, "next": index}, ident=index))
                now = time.monotonic()
                if now - last_state >= STREAM_STATE_SECONDS:
                    last_state = now
                    seq, state = self.server.snapshot()
                    self._write(_sse("state", {"seq": seq, "state": state}))
                if reason is None:
                    reason = self._done_reason(study, source, payload)
                    if reason is not None:
                        self._write(_sse("done", {"next": index, "reason": reason}))
                if now - started >= STREAM_MAX_SECONDS:
                    return self._write(_sse("done", {"next": index, "reason": "timeout"}))
                if now - last_ping >= STREAM_PING_SECONDS:
                    last_ping = now
                    self._write(b": ping\n\n")
                time.sleep(STREAM_TAIL_SECONDS)
        except OSError:  # the client is gone; stop the loop without a traceback
            return

    def _done_reason(self, study: Path, source: Source, payload: dict[str, Any]) -> str | None:
        """Why the source is finished: result line, score/verdict file, or a dead loop process."""
        if payload["done"]:
            if source_finished(study, source):
                return "score" if source.kind == "node" else "verdict"
            return "settled"
        return None if self.server.running() else "gone"

    def _write(self, data: bytes) -> None:
        self.wfile.write(data)
        self.wfile.flush()
