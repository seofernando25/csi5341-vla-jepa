"""The read-only watch server: a stdlib HTTP server on loopback with JSON endpoints and an SSE stream."""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from rsi.redact import redact_obj
from rsi.watch import state
from rsi.watch.assets import static_response
from rsi.watch.context import context_payload
from rsi.watch.dream import dream_payload
from rsi.watch.files import PART
from rsi.watch.source import (
    ARTIFACTS,
    SourceError,
    events_payload,
    parse_source,
    read_artifact,
)
from rsi.watch.stream import STREAM_STATE_SECONDS, StreamMixin

AFTER = re.compile(r"[0-9]{1,9}")
MAX_STREAMS = 4
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    study_dir: Path
    verbose: bool = False
    streams: threading.BoundedSemaphore
    _state: dict[str, Any] | None = None
    _state_at: float = 0.0
    _seq: int = 0
    _running: bool | None = None
    _running_at: float = 0.0
    _lock = threading.Lock()

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A client that vanished mid-response is not a server error; never print its traceback."""
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)

    def snapshot(self) -> tuple[int, dict[str, Any]]:
        """Monotonic sequence plus a state payload rebuilt at most once per state interval."""
        now = time.monotonic()
        with self._lock:
            if self._state is None or now - self._state_at >= STREAM_STATE_SECONDS:
                self._seq += 1
                self._state = redact_obj(state.build_state(self.study_dir))
                self._state_at = now
            return self._seq, self._state

    def running(self) -> bool:
        """Whether the loop process still holds the study lock, memoized for one state interval."""
        now = time.monotonic()
        with self._lock:
            if self._running is None or now - self._running_at >= STREAM_STATE_SECONDS:
                self._running = state.lock_owner(self.study_dir) is not None
                self._running_at = now
            return self._running


class _Handler(StreamMixin, BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"
    timeout = 10
    _streaming = False

    def do_GET(self) -> None:
        try:
            self._route()
        except Exception:  # noqa: BLE001 - last resort: never leak a traceback to the client
            if self._streaming:
                return
            try:
                self._json(500, {"error": "internal error"})
            except OSError:  # the client is gone; nothing to send and nothing to print
                self.close_connection = True

    def do_POST(self) -> None:
        self.close_connection = True
        self._json(405, {"error": "GET only"})

    do_PUT = do_DELETE = do_PATCH = do_POST

    def _route(self) -> None:
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        study = self.server.study_dir
        if not self._host_ok():
            return self._json(403, {"error": "forbidden host"})
        if not url.path.startswith("/api/"):
            page = static_response(url.path)
            if page is None:
                return self._json(404, {"error": "not found"})
            return self._send(*page)
        if url.path == "/api/state":
            return self._json(200, self.server.snapshot()[1])
        if url.path == "/api/events":
            return self._events(study, query)
        if url.path == "/api/stream":
            return self._stream(study, query)
        if url.path == "/api/context":
            return self._context(study, query)
        if url.path == "/api/artifact":
            return self._artifact(study, query)
        if url.path == "/api/dream":
            return self._dream(study, query)
        return self._json(404, {"error": "not found"})

    def _host_ok(self) -> bool:
        """Reject a non-loopback `Host`, which closes DNS-rebinding reads of agent transcripts."""
        host = (self.headers.get("Host") or "").strip().lower()
        port = self.server.server_address[1]
        return host in {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    def _after(self, query: dict[str, list[str]]) -> int | None:
        """`after` query value, falling back to the SSE `Last-Event-ID` of a reconnect."""
        raw = _first(query, "after") or self.headers.get("Last-Event-ID") or "0"
        return int(raw) if AFTER.fullmatch(raw) else None

    def _events(self, study: Path, query: dict[str, list[str]]) -> None:
        try:
            source = parse_source(_first(query, "source"))
        except SourceError as exc:
            return self._json(400, {"error": str(exc)})
        after = self._after(query)
        if after is None:
            return self._json(400, {"error": "after must be a non-negative integer"})
        return self._json(200, redact_obj(events_payload(study, source, after)))

    def _context(self, study: Path, query: dict[str, list[str]]) -> None:
        try:
            source = parse_source(_first(query, "source"))
        except SourceError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, redact_obj(context_payload(study, source)))

    def _stream(self, study: Path, query: dict[str, list[str]]) -> None:
        """Server-Sent Events: `events` as the log grows, `state` every 2 s, `done` on exit."""
        try:
            source = parse_source(_first(query, "source"))
        except SourceError as exc:
            return self._json(400, {"error": str(exc)})
        after = self._after(query)
        if after is None:
            return self._json(400, {"error": "after must be a non-negative integer"})
        if not self.server.streams.acquire(blocking=False):
            return self._json(503, {"error": "too many streams"})
        try:
            self._stream_loop(study, source, after)
        finally:
            self.server.streams.release()

    def _artifact(self, study: Path, query: dict[str, list[str]]) -> None:
        try:
            source = parse_source(_first(query, "source"))
        except SourceError as exc:
            return self._json(400, {"error": str(exc)})
        name = _first(query, "name") or ""
        if name not in ARTIFACTS:
            return self._json(400, {"error": f"name must be one of {', '.join(ARTIFACTS)}"})
        text = read_artifact(study, source, name)
        if text is None:
            return self._json(404, {"error": "artifact not found"})
        return self._json(200, {"name": name, "text": redact_obj(text)})

    def _dream(self, study: Path, query: dict[str, list[str]]) -> None:
        """One cycle's policy-development record; an unknown cycle is a 404, not an empty page."""
        cycle = (_first(query, "cycle") or "").strip()
        if not cycle or not PART.fullmatch(cycle) or cycle in (".", ".."):
            return self._json(400, {"error": "cycle must be a cycle id"})
        world = _first(query, "world") or None
        if world is not None and (len(world) > 200 or not PART.fullmatch(world)):
            return self._json(400, {"error": "world must be a world id"})
        payload = dream_payload(study, cycle, world)
        if payload["status"] == "missing":
            return self._json(404, {"error": "unknown cycle"})
        return self._json(200, redact_obj(payload))

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(format, *args)


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def make_server(study_dir: Path, host: str = "127.0.0.1", port: int = 8787) -> _Server:
    """Bind the read-only watch server for one study directory (port 0 picks a free port)."""
    server = _Server((host, port), _Handler)
    server.study_dir = Path(study_dir).resolve()
    server.streams = threading.BoundedSemaphore(MAX_STREAMS)
    return server


def run_watch(
    study_dir: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    allow_remote: bool = False,
) -> int:
    """Serve the watch page until Ctrl-C; refuse a non-loopback bind unless explicitly allowed."""
    if host not in LOCAL_HOSTS and not allow_remote:
        raise SystemExit(
            f"--host {host} is not loopback and the watch page has no authentication; "
            "bind 127.0.0.1 or localhost"
        )
    server = make_server(study_dir, host, port)
    bound_host, bound_port = server.server_address[:2]
    print(
        f"watching {server.study_dir} at http://{bound_host}:{bound_port}/ (Ctrl-C to stop)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
