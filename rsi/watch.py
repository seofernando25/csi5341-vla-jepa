"""Read-only live watch of a Dream-RSI study: state builder, artifacts and a stdlib HTTP server."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from rsi.watch_context import context_payload
from rsi.watch_events import read_events, redact_obj
from rsi.watch_files import (
    ARTIFACT_CAP,
    HEX,
    PART,
    STEP_RE,
    inside,
    log_path,
    read_json,
    read_text,
    step_dir,
)

__all__ = [
    "Source",
    "SourceError",
    "build_state",
    "context_payload",
    "events_payload",
    "make_server",
    "parse_source",
    "read_artifact",
    "run_watch",
    "static_path",
]

ARTIFACTS = ("proposal", "diff", "score", "error", "solution", "policy", "verdict", "curve", "seed")
NODE_FILES = {
    "proposal": "proposal.md",
    "diff": "diff.patch",
    "score": "score.json",
    "error": "error.txt",
}
AFTER = re.compile(r"[0-9]{1,9}")
STREAM_TAIL_SECONDS = 0.25
STREAM_STATE_SECONDS = 2.0
STREAM_PING_SECONDS = 15.0
STREAM_MAX_SECONDS = 1800.0
STREAM_EVENT_CAP = 200
MAX_STREAMS = 4
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
STATIC_INDEX = Path(__file__).resolve().parent / "watch_static" / "index.html"
STATIC_DIRS = ("css", "js")
STATIC_FILES = ("index.html", "fixture.json", "fixture_events.json")
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}
_GPU: list[Any] = [0.0, None]


class SourceError(ValueError):
    """A source id that is not `node:<cycle>:<cell>` or `policy:<cycle>:<step>`."""


@dataclass(frozen=True)
class Source:
    """A parsed, validated source id; `name` is a cell id (node) or a step number (policy)."""

    kind: str
    cycle_id: str
    name: str


def parse_source(text: str | None) -> Source:
    """Parse a source id strictly; path components are validated before any filesystem use."""
    parts = (text or "").split(":")
    if len(parts) != 3 or parts[0] not in ("node", "policy"):
        raise SourceError(f"bad source id: {text!r}")
    cycle, name = parts[1], parts[2]
    for part in (cycle, name):
        if not PART.match(part) or part in (".", ".."):
            raise SourceError(f"bad source id component: {part!r}")
    if parts[0] == "policy" and (not name.isdigit() or int(name) < 1):
        raise SourceError(f"policy step must be an integer >= 1: {name!r}")
    return Source(parts[0], cycle, name)


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _branches(value: Any) -> list[int] | None:
    """Branch indices listed in a cycle manifest, or None when the field is absent."""
    if not isinstance(value, list):
        return None
    return [b for b in value if isinstance(b, int) and not isinstance(b, bool)]


def _str(value: Any) -> str | None:
    return None if value is None else str(value)


def _iso(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _gpu_block() -> dict[str, Any] | None:
    """nvidia-smi summary, cached for 2 s; None when the query fails or no GPU is present."""
    if time.monotonic() - _GPU[0] < 2.0:
        return _GPU[1]
    value: dict[str, Any] | None = None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        fields = (
            [f.strip() for f in out.stdout.splitlines()[0].split(",")]
            if out.returncode == 0
            else []
        )
        if len(fields) == 4 and all(f.isdigit() for f in fields[1:]):
            value = {
                "name": fields[0],
                "memory_used_mb": int(fields[1]),
                "memory_total_mb": int(fields[2]),
                "utilization_pct": int(fields[3]),
            }
    except (OSError, subprocess.SubprocessError, IndexError):
        value = None
    _GPU[0], _GPU[1] = time.monotonic(), value
    return value


def _blank(
    cell_id: str, parent: str | None, branch: int | None, attempt: int | None
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "parent_id": parent,
        "branch": branch,
        "attempt": attempt,
        "round": None,
        "status": "pending",
        "fail_class": None,
        "score": None,
        "size_reduction": None,
        "successes": None,
        "episodes": None,
        "resident_mb": None,
        "commit": None,
        "agent_seconds": None,
        "eval_seconds": None,
        "agent_cost_usd": None,
        "agent_turns": None,
        "agent_timed_out": False,
        "error": None,
        "headline": None,
        "promoted": None,
        "started_utc": None,
        "elapsed_s": None,
    }


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        return None


def _mark_stopped(node: dict[str, Any]) -> None:
    """A node cannot be in flight once no loop process holds the study lock.

    The filesystem probe cannot tell a live agent from one whose loop died, so `build_state`
    demotes in-flight nodes to `stopped` and drops the live clock. Pending and finished nodes
    keep their status.
    """
    if node["status"] in ("agent", "evaluating"):
        node["status"] = "stopped"
        node["elapsed_s"] = None


def _headline(study: Path, path: Path) -> str | None:
    text = read_text(study, path, cap=20_000) or ""
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()[:200] or None
    return None


def _node(
    study: Path,
    ndir: Path,
    cell_id: str,
    parent: str | None,
    branch: int | None,
    attempt: int | None,
) -> dict[str, Any]:
    rec = _blank(cell_id, parent, branch, attempt)
    score = read_json(study, ndir / "score.json")
    if score is not None:
        sizes = score.get("sizes") if isinstance(score.get("sizes"), dict) else {}
        resident = _num(sizes.get("resident_bytes"))
        rec.update(
            status="done",
            fail_class=_str(score.get("fail_class")),
            score=_num(score.get("score")),
            size_reduction=_num(sizes.get("size_reduction")),
            successes=_int(score.get("successes")),
            episodes=_int(score.get("episodes")),
            resident_mb=round(resident / 1e6, 2) if resident is not None else None,
            commit=_str(score.get("commit")),
            agent_seconds=_num(score.get("agent_seconds")),
            eval_seconds=_num(score.get("eval_seconds")),
            agent_cost_usd=_num(score.get("agent_cost_usd")),
            agent_turns=_int(score.get("agent_turns")),
            agent_timed_out=score.get("agent_timed_out") is True,
            error=_str(score.get("error")),
            round=_int(score.get("round")),
        )
    elif (ndir / "eval").is_dir():
        rec["status"] = "evaluating"
    elif (ndir / "logs" / "stdout.log").is_file():
        rec["status"] = "agent"
    if rec["status"] in ("agent", "evaluating"):
        marker = ndir / "logs" / "stdout.log" if rec["status"] == "agent" else ndir / "eval"
        started = _mtime(marker)
        if started is not None:
            rec["started_utc"] = started.isoformat()
            rec["elapsed_s"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    if ndir.is_dir():
        rec["headline"] = _headline(study, ndir / "proposal.md")
        dev = read_json(study, ndir / "development.json")
        if dev is not None:
            rec["promoted"] = {
                "successes": _int(dev.get("successes")) or 0,
                "episodes": _int(dev.get("episodes")) or 0,
            }
    return rec


def _nodes(study: Path, cdir: Path) -> list[dict[str, Any]]:
    out = [{**_blank("root", None, None, None), "status": "done", "fail_class": "ok", "score": 0.0}]
    grid = read_json(study, cdir / "grid_plan.json") or {}
    branches, attempts = _int(grid.get("max_branches")), _int(grid.get("max_attempts_per_branch"))
    slots: list[tuple[str, str, int, int]] = []
    if branches is not None and attempts is not None:
        for b in range(branches):
            parent = "root"
            for a in range(attempts + 1):
                cell = f"{cdir.name}-b{b}-a{a}"
                slots.append((cell, parent, b, a))
                parent = cell
    ndir_root = cdir / "nodes"
    for cell, parent, b, a in slots:
        out.append(_node(study, ndir_root / cell, cell, parent, b, a))
    known = {cell for cell, *_ in slots}
    if ndir_root.is_dir():
        for path in sorted(ndir_root.iterdir()):
            if path.is_dir() and path.name not in known and PART.match(path.name):
                out.append(_node(study, path, path.name, None, None, None))
    return out


def _policy_dev(study: Path, cid: str, raw: dict[str, Any]) -> dict[str, Any]:
    pdir = study / "policies" / cid
    selection = read_json(study, pdir / "selection.json") or {}
    steps: list[dict[str, Any]] = []
    selected_step = None
    if pdir.is_dir():
        for sdir in sorted(pdir.iterdir()):
            match = STEP_RE.match(sdir.name)
            if not match or not sdir.is_dir() or int(match.group(1)) < 1:
                continue
            verdict = read_json(study, pdir / "evidence" / "history" / sdir.name / "verdict.json")
            step = int(match.group(1))
            if verdict is None:
                status, mean_v, delta, reason = "running", None, None, None
            else:
                status = str(verdict.get("status"))
                status = status if status in ("accepted", "not_better", "rejected") else "rejected"
                mean_v, delta = _num(verdict.get("mean_v")), _num(verdict.get("delta_vs_best"))
                reason = _str(verdict.get("error"))
                if (
                    selection.get("selected_slot")
                    and verdict.get("slot") == selection["selected_slot"]
                ):
                    selected_step = step
            steps.append(
                {
                    "step": step,
                    "status": status,
                    "mean_v": mean_v,
                    "delta_vs_best": delta,
                    "reason": reason,
                    "attempts": _int((verdict or {}).get("attempts")),
                    "slot": _str((verdict or {}).get("slot")),
                }
            )
    pd_status = raw.get("status")
    return {
        "status": pd_status
        if pd_status in ("pending", "running", "done", "skipped")
        else "pending",
        "selected_step": selected_step,
        "incumbent_v": _num(selection.get("incumbent_in_sample_V")),
        "selected_v": _num(selection.get("selected_in_sample_V")),
        "heldout_delta": _num(selection.get("heldout_delta_vs_incumbent")),
        "steps": steps,
    }


def _cycle(study: Path, cid: str, iteration: int | None, it: dict[str, Any]) -> dict[str, Any]:
    cdir = study / "cycles" / cid
    cyc = it.get("cycle") if isinstance(it.get("cycle"), dict) else {}
    manifest = read_json(study, cdir / "live_cycle_manifest.json")
    done = manifest is not None or cyc.get("status") == "done"
    policy_in = it.get("policy_in") if isinstance(it.get("policy_in"), dict) else {}
    return {
        "cycle_id": cid,
        "iteration": iteration,
        "status": "done" if done else "running",
        "decision_rounds": _int(manifest.get("decision_rounds")) if manifest else None,
        "opened_branches": _branches(manifest.get("opened_branches")) if manifest else None,
        "probes_used": _int(manifest.get("probes_used")) if manifest else None,
        "effective_sequential_rounds": (
            _int(manifest.get("effective_sequential_rounds")) if manifest else None
        ),
        "agent_timeout_s": _num(manifest.get("agent_timeout_s")) if manifest else None,
        "policy_sha256": _str(policy_in.get("sha256")),
        "nodes": _nodes(study, cdir),
        "policy_dev": _policy_dev(
            study, cid, it.get("policy_dev") if isinstance(it.get("policy_dev"), dict) else {}
        ),
    }


def _iteration_done(it: dict[str, Any]) -> bool:
    cyc = it.get("cycle") if isinstance(it.get("cycle"), dict) else {}
    pdev = it.get("policy_dev") if isinstance(it.get("policy_dev"), dict) else {}
    return cyc.get("status") == "done" and pdev.get("status") in ("done", "skipped")


def _elapsed(loop_raw: dict[str, Any] | None, running: bool) -> float | None:
    if loop_raw is None:
        return None
    start = _iso(loop_raw.get("first_started_utc") or loop_raw.get("started_utc"))
    end = datetime.now(UTC) if running else _iso(loop_raw.get("updated_utc"))
    if start is None or end is None:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    return round((end - start).total_seconds(), 1) if end >= start else None


def _holder(study: Path) -> str | None:
    from rsi.service import lock_holder

    try:
        return lock_holder(study)
    except OSError:
        return None


def _protocol_block(study: Path) -> dict[str, Any]:
    proto = read_json(study, study / "protocol.json") or {}
    drsi = proto.get("dream_rsi") if isinstance(proto.get("dream_rsi"), dict) else {}
    gates = proto.get("gates") if isinstance(proto.get("gates"), dict) else {}
    phases = proto.get("phases") if isinstance(proto.get("phases"), dict) else {}
    grid = drsi.get("grid_plan") if isinstance(drsi.get("grid_plan"), dict) else {}
    agent = drsi.get("agent") if isinstance(drsi.get("agent"), dict) else {}
    return {
        "smoke_gate": _int(gates.get("smoke_min_successes")),
        "smoke_trials": _int((phases.get("smoke") or {}).get("trials")),
        "dev_gate": _int(gates.get("development_min_successes")),
        "dev_trials": _int((phases.get("development") or {}).get("trials")),
        "max_parallelism": _int(grid.get("max_parallelism")),
        "rounds_cap": _int(grid.get("rounds_cap")),
        "max_branches": _int(grid.get("max_branches")),
        "max_attempts_per_branch": _int(grid.get("max_attempts_per_branch")),
        "gpu_slots": _int(drsi.get("gpu_slots")),
        "agent": {
            "backend": _str(agent.get("backend")),
            "model": _str(agent.get("model")),
            "effort": _str(agent.get("effort")),
        },
    }


def build_state(study_dir: Path) -> dict[str, Any]:
    """Watch state per the watch_api contract; missing or half-written files are skipped."""
    study = Path(study_dir)
    now = datetime.now(UTC)
    raw = read_json(study, study / "loop_state.json")
    running = _holder(study) is not None
    iterations = [it for it in (raw or {}).get("iterations", []) if isinstance(it, dict)]
    cycles = [
        _cycle(study, str(it.get("cycle_id")), _int(it.get("iteration")), it)
        for it in iterations
        if it.get("cycle_id") and PART.match(str(it.get("cycle_id")))
    ]
    known = {c["cycle_id"] for c in cycles}
    cdir_root = study / "cycles"
    if cdir_root.is_dir():
        for path in sorted(cdir_root.iterdir()):
            if path.is_dir() and path.name not in known and PART.match(path.name):
                cycles.append(_cycle(study, path.name, None, {}))
    nodes = [n for c in cycles for n in c["nodes"]]
    if not running:
        for node in nodes:
            _mark_stopped(node)
    scores = [n["score"] for n in nodes if n["status"] == "done" and n["score"] is not None]
    cost = sum(n["agent_cost_usd"] or 0.0 for n in nodes)
    cost += sum(
        _num((it.get("policy_dev") or {}).get("agent_cost_usd")) or 0.0 for it in iterations
    )
    current = next((it for it in iterations if not _iteration_done(it)), None)
    stage = None
    if current is not None:
        cyc = current.get("cycle") if isinstance(current.get("cycle"), dict) else {}
        stage = "cycle" if cyc.get("status") != "done" else "policy_dev"
    iteration = None
    if current is not None:
        iteration = _int(current.get("iteration"))
    elif iterations:
        iteration = _int(iterations[-1].get("iteration"))
    state_status = (raw or {}).get("status")
    if running:
        status = "running"
    elif state_status == "running":
        status = "stopped"
    else:
        status = state_status if state_status in ("stopped", "done", "error") else "idle"
    run = (raw or {}).get("run") if isinstance((raw or {}).get("run"), dict) else {}
    first = (raw or {}).get("first_run") if isinstance((raw or {}).get("first_run"), dict) else {}
    probe_budget = _int(run.get("probe_budget"))
    if probe_budget is None:
        probe_budget = _int(first.get("probe_budget"))
    loop = {
        "status": status,
        "stop_reason": (raw or {}).get("stop_reason"),
        "iteration": iteration,
        "stage": stage if running else None,
        "cycles_target": _int((raw or {}).get("cycles_target")),
        "elapsed_s": _elapsed(raw, running),
        "agent_cost_usd": round(cost, 6),
        "best_score": max(scores) if scores else None,
        "running": running,
        "probe_budget": probe_budget,
    }
    return {
        "study_dir": study.name,
        "now": now.isoformat(),
        "loop": loop,
        "protocol": _protocol_block(study),
        "gpu": _gpu_block(),
        "cycles": cycles,
    }


def _solution(study: Path, src: Source, nodes_commit: str | None) -> str | None:
    if not nodes_commit or not HEX.match(nodes_commit):
        return None
    store = study / "cycles" / src.cycle_id / "snapshots.git"
    try:
        out = subprocess.run(
            ["git", "--git-dir", str(store), "show", f"{nodes_commit}:solution.py"],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout[: ARTIFACT_CAP * 4].decode("utf-8", errors="replace")[:ARTIFACT_CAP]


def read_artifact(study_dir: Path, source: Source, name: str) -> str | None:
    """Text of one artifact of a validated source, or None when it does not exist."""
    study = Path(study_dir)
    if source.kind == "node":
        ndir = study / "cycles" / source.cycle_id / "nodes" / source.name
        if name in NODE_FILES:
            return read_text(study, ndir / NODE_FILES[name])
        if name == "solution":
            score = read_json(study, ndir / "score.json") or {}
            return _solution(study, source, _str(score.get("commit")))
        return None
    sdir = step_dir(study, source)
    if name == "policy":
        return read_text(study, sdir / "candidate.py")
    if name == "seed":
        return read_text(study, sdir / "workspace" / "method.py")
    history = study / "policies" / source.cycle_id / "evidence" / "history" / sdir.name
    if name == "verdict":
        return read_text(study, history / "verdict.json")
    if name == "curve":
        return read_text(study, history / "scaling_curve.json")
    return None


def _finished(study: Path, source: Source) -> bool:
    if source.kind == "node":
        return (study / "cycles" / source.cycle_id / "nodes" / source.name / "score.json").is_file()
    return (
        study
        / "policies"
        / source.cycle_id
        / "evidence"
        / "history"
        / f"step_{int(source.name):02d}"
        / "verdict.json"
    ).is_file()


def events_payload(study_dir: Path, source: Source, after: int) -> dict[str, Any]:
    """Events from `after` for one source, with the next index and done flag."""
    study = Path(study_dir)
    log = log_path(study, source)
    if log.exists() and inside(study, log) is None:
        return {"events": [], "next": after, "done": True, "error": "log path outside study"}
    finished = _finished(study, source)
    events, nxt, done = read_events(log, after, finished=finished)
    return {"events": events, "next": nxt, "done": done}


def _static_root() -> Path:
    """The directory the page is served from; tests patch STATIC_INDEX to a temp copy."""
    return STATIC_INDEX.parent.resolve()


def static_path(url_path: str) -> Path | None:
    """Resolve one allowlisted static path, refusing traversal, dotfiles and symlinks.

    The page is a set of ES modules under `css/` and `js/` plus three root files. Anything
    else is not served, even if it exists: the static tree is not a general file server.
    """
    relative = url_path.lstrip("/")
    if not relative or relative.endswith("/"):
        return None
    parts = relative.split("/")
    if len(parts) > 3 or any(
        not PART.match(p) or p in (".", "..") or p.startswith(".") for p in parts
    ):
        return None
    if len(parts) == 1 and parts[0] not in STATIC_FILES:
        return None
    if len(parts) > 1 and parts[0] not in STATIC_DIRS:
        return None
    root = _static_root()
    candidate = root / relative
    try:
        real = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not real.is_relative_to(root) or not real.is_file() or candidate.is_symlink():
        return None
    return real


def _static(url_path: str) -> tuple[int, bytes, str] | None:
    """(status, body, content type) for one static request, or None when it is not served."""
    if url_path in ("/", "/index.html"):
        if STATIC_INDEX.is_file():
            return 200, STATIC_INDEX.read_bytes(), STATIC_TYPES[".html"]
        hint = "<p>Watch page not built: rsi/watch_static/index.html is missing.</p>"
        return 404, f"<!doctype html><title>watch</title>{hint}".encode(), STATIC_TYPES[".html"]
    path = static_path(url_path)
    if path is None:
        return None
    ctype = STATIC_TYPES.get(path.suffix)
    if ctype is None:
        return None
    try:
        return 200, path.read_bytes(), ctype
    except OSError:
        return None


def _log_token(path: Path) -> str:
    """Short stable token for one log path, so a client can tell a rotated log from a truncated one."""
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


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
                self._state = redact_obj(build_state(self.study_dir))
                self._state_at = now
            return self._seq, self._state

    def running(self) -> bool:
        """Whether the loop process still holds the study lock, memoized for one state interval."""
        now = time.monotonic()
        with self._lock:
            if self._running is None or now - self._running_at >= STREAM_STATE_SECONDS:
                self._running = _holder(self.study_dir) is not None
                self._running_at = now
            return self._running


class _Handler(BaseHTTPRequestHandler):
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
            page = _static(url.path)
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
            if _finished(study, source):
                return "score" if source.kind == "node" else "verdict"
            return "settled"
        return None if self.server.running() else "gone"

    def _write(self, data: bytes) -> None:
        self.wfile.write(data)
        self.wfile.flush()

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


def _sse(event: str, payload: Any, ident: int | None = None) -> bytes:
    """One Server-Sent Events frame; JSON bodies are single-line so one `data:` field suffices."""
    lines: list[str] = []
    if ident is not None:
        lines.append(f"id: {ident}")
    lines.append(f"event: {event}")
    lines.extend(f"data: {part}" for part in json.dumps(payload).splitlines() or [""])
    return ("\n".join(lines) + "\n\n").encode("utf-8")


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
