"""Tests for the live watch: state builder, incremental events, redaction, artifacts and server."""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from rsi import watch
from rsi.redact import redact
from rsi.watch import SourceError, build_state, make_server, parse_source, read_artifact
from rsi.watch import assets as watch_assets
from rsi.watch import policy as watch_policy
from rsi.watch import server as watch_server
from rsi.watch import state as watch_state
from rsi.watch.transcript import read_events

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NODE_CELL = "iter_000-b0-a0"


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watch_state, "_gpu_block", lambda: None)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json(path: Path, obj: Any) -> None:
    _write(path, json.dumps(obj, indent=2) + "\n")


def _git_store(store: Path, solution: str) -> str:
    """Create a bare snapshot store holding one commit whose tree has solution.py."""
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": "/usr/bin:/bin",
    }
    subprocess.run(["git", "init", "-q", "--bare", str(store)], check=True, env=env)
    blob = (
        subprocess.run(
            ["git", "--git-dir", str(store), "hash-object", "-w", "--stdin"],
            input=solution.encode(),
            capture_output=True,
            check=True,
            env=env,
        )
        .stdout.decode()
        .strip()
    )
    tree = (
        subprocess.run(
            ["git", "--git-dir", str(store), "mktree"],
            input=f"100644 blob {blob}\tsolution.py\n".encode(),
            capture_output=True,
            check=True,
            env=env,
        )
        .stdout.decode()
        .strip()
    )
    commit = (
        subprocess.run(
            ["git", "--git-dir", str(store), "commit-tree", tree, "-m", "root"],
            capture_output=True,
            check=True,
            env=env,
        )
        .stdout.decode()
        .strip()
    )
    return commit


@pytest.fixture
def study(tmp_path: Path) -> Path:
    """A study with one finished node, one evaluating, one agent-running (partial line), one pending
    slot, and a policy_dev with two steps."""
    root = tmp_path / "study"
    cdir = root / "cycles" / "iter_000"
    _json(
        root / "protocol.json",
        {
            "gates": {"smoke_min_successes": 8, "development_min_successes": 77},
            "phases": {"smoke": {"trials": 10}, "development": {"trials": 100}},
            "dream_rsi": {
                "grid_plan": {
                    "max_branches": 1,
                    "max_attempts_per_branch": 3,
                    "max_parallelism": 2,
                    "rounds_cap": 6,
                },
                "agent": {"backend": "claude", "model": "haiku", "effort": "low"},
                "gpu_slots": 2,
            },
        },
    )
    _json(
        root / "loop_state.json",
        {
            "status": "done",
            "stop_reason": "cycles reached",
            "cycles_target": 1,
            "first_started_utc": "2026-11-15T10:00:00+00:00",
            "updated_utc": "2026-11-15T10:10:00+00:00",
            "iterations": [
                {
                    "iteration": 0,
                    "cycle_id": "iter_000",
                    "policy_in": {"path": "p", "sha256": "abc"},
                    "cycle": {"status": "done"},
                    "policy_dev": {
                        "status": "done",
                        "agent_cost_usd": 0.5,
                        "selected_policy_path": "x",
                        "sha256": "d",
                    },
                }
            ],
        },
    )
    _json(cdir / "grid_plan.json", {"max_branches": 1, "max_attempts_per_branch": 3})
    _json(cdir / "live_cycle_manifest.json", {"decision_rounds": 2})
    commit = _git_store(
        cdir / "snapshots.git", "def compress(policy, calibration):\n    return policy\n"
    )

    done = cdir / "nodes" / NODE_CELL
    _json(
        done / "score.json",
        {
            "cell_id": NODE_CELL,
            "commit": commit,
            "fail_class": "ok",
            "score": 0.4,
            "successes": 8,
            "episodes": 10,
            "sizes": {"resident_bytes": 2_000_000, "size_reduction": 0.4},
            "agent_cost_usd": 0.01,
            "agent_turns": 7,
            "agent_timed_out": False,
            "error": None,
        },
    )
    _write(done / "proposal.md", "# Int8 all linears\n\nBody.\n")
    _write(done / "diff.patch", "--- a\n+++ b\n")
    _write(done / "logs" / "stdout.log", "")

    evaluating = cdir / "nodes" / "iter_000-b0-a1"
    (evaluating / "eval").mkdir(parents=True)
    _write(evaluating / "logs" / "stdout.log", "")

    running = cdir / "nodes" / "iter_000-b0-a2"
    good = json.dumps(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "working"}]}}
    )
    _write(running / "logs" / "stdout.log", good + "\n" + good + "\n" + '{"type":"assis')

    pdir = root / "policies" / "iter_000"
    _write(pdir / "step_00" / "candidate.py", "incumbent\n")
    for step, status, mean_v, slot in (
        (1, "accepted", 0.61, "m1_a"),
        (2, "rejected", None, "m2_b"),
    ):
        sdir = pdir / f"step_{step:02d}"
        _write(sdir / "candidate.py", f"step {step}\n")
        _write(sdir / "logs" / "attempt_0" / "stdout.log", good + "\n")
        _json(
            pdir / "evidence" / "history" / f"step_{step:02d}" / "verdict.json",
            {
                "step": step,
                "slot": slot,
                "status": status,
                "mean_v": mean_v,
                "delta_vs_best": 0.05 if mean_v else None,
                "error": None if mean_v else "compile failed",
            },
        )
    _json(
        pdir / "selection.json",
        {
            "selected_slot": "m1_a",
            "incumbent_in_sample_V": 0.5,
            "selected_in_sample_V": 0.61,
            "heldout_delta_vs_incumbent": 0.02,
        },
    )
    return root


def test_empty_study_gives_idle_state(tmp_path: Path) -> None:
    state = build_state(tmp_path / "missing")
    assert state["loop"]["status"] == "idle"
    assert state["loop"]["running"] is False
    assert state["cycles"] == []
    assert state["protocol"]["smoke_gate"] is None
    assert state["gpu"] is None


def test_state_node_statuses(study: Path) -> None:
    state = build_state(study)
    cycle = state["cycles"][0]
    assert cycle["status"] == "done"
    assert cycle["decision_rounds"] == 2
    assert cycle["policy_sha256"] == "abc"
    nodes = {n["cell_id"]: n for n in cycle["nodes"]}
    assert nodes["root"]["status"] == "done" and nodes["root"]["score"] == 0.0
    done = nodes[NODE_CELL]
    assert done["status"] == "done" and done["score"] == 0.4
    assert done["resident_mb"] == 2.0 and done["size_reduction"] == 0.4
    assert done["headline"] == "Int8 all linears"
    assert done["parent_id"] == "root" and done["branch"] == 0 and done["attempt"] == 0
    # No loop process holds the lock, so in-flight nodes are demoted to `stopped`.
    assert nodes["iter_000-b0-a1"]["status"] == "stopped"
    assert nodes["iter_000-b0-a2"]["status"] == "stopped"
    assert nodes["iter_000-b0-a3"]["status"] == "pending"
    assert nodes["iter_000-b0-a2"]["parent_id"] == "iter_000-b0-a1"


def test_state_loop_and_policy_dev(study: Path) -> None:
    state = build_state(study)
    loop = state["loop"]
    assert loop["status"] == "done" and loop["running"] is False and loop["stage"] is None
    assert loop["best_score"] == 0.4
    assert loop["agent_cost_usd"] == pytest.approx(0.51)
    assert loop["elapsed_s"] == 600.0
    pdev = state["cycles"][0]["policy_dev"]
    assert pdev["status"] == "done"
    assert pdev["selected_step"] == 1
    assert pdev["incumbent_v"] == 0.5 and pdev["selected_v"] == 0.61
    assert pdev["heldout_delta"] == 0.02
    assert [(s["step"], s["status"]) for s in pdev["steps"]] == [(1, "accepted"), (2, "rejected")]
    assert pdev["steps"][1]["reason"] == "compile failed"


def test_state_survives_half_written_json(study: Path) -> None:
    _write(
        study / "cycles" / "iter_000" / "nodes" / NODE_CELL / "score.json", '{"score": 0.4, "sizes"'
    )
    node = {n["cell_id"]: n for n in build_state(study)["cycles"][0]["nodes"]}[NODE_CELL]
    assert node["status"] == "stopped"


def test_read_events_incremental_and_partial_line(tmp_path: Path) -> None:
    log = tmp_path / "stdout.log"
    line_a = json.dumps(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "a"}]}}
    )
    line_b = json.dumps(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "b"}]}}
    )
    _write(log, line_a + "\n" + '{"type":"assistant","mess')
    events, nxt, done = read_events(log, 0)
    assert [e["text"] for e in events] == ["a"] and nxt == 1 and done is False
    assert events[0]["i"] == 0
    events, nxt, _ = read_events(log, nxt)
    assert events == [] and nxt == 1
    with log.open("a", encoding="utf-8") as fh:
        fh.write('age":{"content":[]}}\n' + line_b + "\n")
    events, nxt, _ = read_events(log, 1)
    assert [e["text"] for e in events] == ["b"] and events[0]["i"] == 1 and nxt == 2


def test_read_events_finished_consumes_unterminated_line_and_result_completes(
    tmp_path: Path,
) -> None:
    log = tmp_path / "stdout.log"
    result = json.dumps({"type": "result", "num_turns": 3, "total_cost_usd": 0.02, "result": "ok"})
    _write(log, result + "\n")
    events, nxt, done = read_events(log, 0)
    assert done is True and nxt == 1 and events[0]["kind"] == "result"
    assert events[0]["meta"]["num_turns"] == 3 and events[0]["meta"]["cost_usd"] == 0.02
    partial = tmp_path / "partial.log"
    _write(partial, '{"type":"system","subtype":"init","model":"m"}')
    events, nxt, done = read_events(partial, 0, finished=True)
    assert nxt == 1 and done is True


def test_read_events_old_single_json_format(tmp_path: Path) -> None:
    log = tmp_path / "old.log"
    _write(log, json.dumps({"type": "result", "num_turns": 9, "result": "old"}) + "\n")
    events, _, done = read_events(log, 0)
    assert done is True and events[0]["text"] == "old" and events[0]["meta"]["num_turns"] == 9


def test_read_events_plain_text_and_missing(tmp_path: Path) -> None:
    log = tmp_path / "pi.log"
    _write(log, "plain line\n")
    events, _, _ = read_events(log, 0)
    assert events[0]["kind"] == "text" and events[0]["text"] == "plain line"
    assert read_events(tmp_path / "absent.log", 0) == ([], 0, False)


def test_read_events_fixture_stream(tmp_path: Path) -> None:
    log = tmp_path / "stream.jsonl"
    log.write_text((FIXTURES / "claude_stream.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    events, nxt, done = read_events(log, 0)
    kinds = [e["kind"] for e in events]
    assert done is True and nxt == len(events)
    assert kinds[0] == "system" and "tool_call" in kinds
    calls = [e for e in events if e["kind"] == "tool_call"]
    assert calls[0]["tool"] == "Read" and calls[0]["input"]["file_path"] == "/tmp/evidence/TASK.md"
    results = [e for e in events if e["kind"] == "tool_result"]
    assert results[0]["tool"] == "Read" and "Write the single word" in results[0]["output"]
    assert any(e["is_error"] for e in results)
    assert events[-1]["kind"] == "result" and events[-1]["meta"]["num_turns"] == 5


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("key sk-ant-api03-AbC_def-123 end", "key sk-ant-[redacted] end"),
        ('{"accessToken": "tok123"}', '{"accessToken": "[redacted]"}'),
        ('{"refreshToken":"r-1"}', '{"refreshToken":"[redacted]"}'),
        ("refresh_token=abc", 'refresh_token="[redacted]"'),
        ('{"access_token": "zz"}', '{"access_token": "[redacted]"}'),
        ('"api_key": "k"', '"api_key": "[redacted]"'),
        ('{"apiKey": "k2"}', '{"apiKey": "[redacted]"}'),
        ('"authorization": "Bearer abc.def"', '"authorization": "[redacted]"'),
        ("Authorization: Bearer xyz123", 'Authorization: "[redacted]"'),
        ("Bearer eyJhbGciOi.payload", "Bearer [redacted]"),
        ('{"apiKeySource": "none"}', '{"apiKeySource": "none"}'),
    ],
)
def test_redaction_patterns(raw: str, expected: str) -> None:
    assert redact(raw) == expected


def test_credential_file_reads_are_replaced(tmp_path: Path) -> None:
    log = tmp_path / "creds.log"
    call = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Read",
                    "input": {"file_path": "/home/u/.claude/.credentials.json"},
                }
            ]
        },
    }
    result = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": '{"accessToken": "sk-ant-secretsecret"}',
                }
            ]
        },
    }
    bash = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Bash",
                    "input": {"command": "cat ~/.pi/agent/auth.json"},
                }
            ]
        },
    }
    bash_out = {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "tool_use_id": "t2", "content": "x" * 60}]},
    }
    _write(log, "\n".join(json.dumps(o) for o in (call, result, bash, bash_out)) + "\n")
    events, _, _ = read_events(log, 0)
    outputs = [e["output"] for e in events if e["kind"] == "tool_result"]
    assert outputs == ["[redacted: credential file]", "[redacted: credential file]"]
    assert "secretsecret" not in json.dumps(events)


def test_long_output_is_capped_and_flagged(tmp_path: Path) -> None:
    log = tmp_path / "big.log"
    big = {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "tool_use_id": "z", "content": "y" * 9000}]
        },
    }
    _write(log, json.dumps(big) + "\n")
    events, _, _ = read_events(log, 0)
    assert len(events[0]["output"]) == 8000 and events[0]["truncated"] is True


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "node",
        "node:iter_000",
        "node:iter_000:root:x",
        "node:../etc:root",
        "node:iter_000:..",
        "node:iter_000:a/b",
        "other:iter_000:root",
        "policy:iter_000:-1",
        "policy:iter_000:x",
        "node:iter 0:root",
    ],
)
def test_bad_source_ids_rejected(text: str | None) -> None:
    with pytest.raises(SourceError):
        parse_source(text)


def test_good_source_ids_parsed() -> None:
    assert parse_source("node:iter_000:iter_000-b0-a0") == watch.Source(
        "node", "iter_000", "iter_000-b0-a0"
    )
    assert parse_source("policy:iter_000:3").name == "3"


def test_artifacts_node_and_missing(study: Path) -> None:
    node = parse_source(f"node:iter_000:{NODE_CELL}")
    assert read_artifact(study, node, "proposal").startswith("# Int8")
    assert read_artifact(study, node, "diff").startswith("--- a")
    assert read_artifact(study, node, "error") is None
    assert "return policy" in read_artifact(study, node, "solution")
    assert read_artifact(study, parse_source("node:iter_000:iter_000-b0-a3"), "proposal") is None


def test_artifacts_policy_step(study: Path) -> None:
    step = parse_source("policy:iter_000:1")
    assert read_artifact(study, step, "policy") == "step 1\n"
    assert '"status": "accepted"' in read_artifact(study, step, "verdict")
    assert read_artifact(study, step, "curve") is None


def test_policy_step_timeout_is_not_shown_as_rejected(tmp_path: Path) -> None:
    root = tmp_path / "policies" / "iter_000"
    _json(root / "selection.json", {"selected_slot": "m0_incumbent"})
    _json(
        root / "evidence" / "history" / "step_01" / "verdict.json",
        {
            "step": 1,
            "slot": "m1_llm",
            "status": "timeout",
            "mean_v": None,
            "delta_vs_best": None,
            "error": "Sandboxed process timed out after 1200.0s",
            "attempts": 1,
        },
    )
    (root / "step_01").mkdir(parents=True, exist_ok=True)
    dev = watch_policy.policy_dev_block(tmp_path, root, {"status": "done"})
    assert [step["status"] for step in dev["steps"]] == ["timeout"]
    assert dev["steps"][0]["reason"].startswith("Sandboxed process timed out")


@pytest.fixture
def server(study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    index = tmp_path / "index.html"
    index.write_text("<!doctype html><title>t</title>", encoding="utf-8")
    monkeypatch.setattr(watch_assets, "STATIC_INDEX", index)
    srv = make_server(study, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _get(srv, path: str, method: str = "GET") -> tuple[int, bytes, dict[str, str]]:
    port = srv.server_address[1]
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def test_server_routes(server, study: Path) -> None:
    status, body, headers = _get(server, "/api/state")
    assert status == 200 and headers["Cache-Control"] == "no-store"
    assert json.loads(body)["loop"]["status"] == "done"

    status, body, _ = _get(server, f"/api/events?source=node:iter_000:{NODE_CELL}&after=0")
    payload = json.loads(body)
    assert status == 200 and payload["done"] is True and payload["next"] == 0

    status, body, _ = _get(server, "/api/events?source=policy:iter_000:1&after=0")
    assert status == 200 and json.loads(body)["next"] == 1 and json.loads(body)["done"] is True

    status, body, _ = _get(
        server, "/api/artifact?source=node:iter_000:" + NODE_CELL + "&name=proposal"
    )
    assert status == 200 and json.loads(body)["name"] == "proposal"
    status, body, _ = _get(
        server, "/api/artifact?source=node:iter_000:" + NODE_CELL + "&name=error"
    )
    assert status == 404 and "error" in json.loads(body)

    status, body, headers = _get(server, "/")
    assert status == 200 and b"<title>t</title>" in body and "text/html" in headers["Content-Type"]


def test_server_rejects_bad_requests(server) -> None:
    assert _get(server, "/api/events?source=node:..:x")[0] == 400
    assert _get(server, "/api/events?source=node:iter_000:root&after=-1")[0] == 400
    assert _get(server, "/api/events?source=node:iter_000:root&after=abc")[0] == 400
    assert _get(server, "/api/artifact?source=node:iter_000:root&name=passwords")[0] == 400
    assert _get(server, "/api/artifact?source=node:iter_000:a%2Fb&name=proposal")[0] == 400
    assert _get(server, "/api/nope")[0] == 404
    status, body, _ = _get(server, "/api/state", method="POST")
    assert status == 405 and json.loads(body)["error"] == "GET only"


def test_server_index_hint_when_page_missing(
    study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watch_assets, "STATIC_INDEX", tmp_path / "absent.html")
    srv = make_server(study, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        status, body, _ = _get(srv, "/")
        assert status == 404 and b"not built" in body
        assert str(tmp_path).encode() not in body
    finally:
        srv.shutdown()
        srv.server_close()


def _raw(srv, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=10) as sock:
        sock.sendall(request)
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    return b"".join(chunks)


def _frames(srv, path: str, stop: str = "done", timeout: float = 10.0) -> list[tuple]:
    """Read SSE frames until `stop` arrives, returning (event, data, id) tuples."""
    port = srv.server_address[1]
    frames: list[tuple] = []
    deadline = time.monotonic() + timeout
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as resp:
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        assert "Content-Length" not in resp.headers
        name, data, ident = None, [], None
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data.append(line[len("data: ") :])
            elif line.startswith("id: "):
                ident = int(line[len("id: ") :])
            elif not line:
                if name:
                    frames.append((name, json.loads("\n".join(data)), ident))
                name, data, ident = None, [], None
                if frames and frames[-1][0] == stop:
                    return frames
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"timed out waiting for {stop!r}; got {[f[0] for f in frames]}"
                )
    return frames


def test_stream_frames_for_a_finished_node(server) -> None:
    frames = _frames(server, f"/api/stream?source=node:iter_000:{NODE_CELL}&after=0")
    names = [f[0] for f in frames]
    assert names[0] == "hello" and names[-1] == "done"
    assert "state" in names
    hello = frames[0][1]
    assert hello["v"] == 1 and hello["cursor"] == 0 and hello["interval_ms"] == 250
    assert hello["source"] == f"node:iter_000:{NODE_CELL}" and len(hello["log"]) == 12
    state = next(f[1] for f in frames if f[0] == "state")
    assert state["seq"] >= 1 and state["state"]["loop"]["status"] == "done"
    done = frames[-1][1]
    assert done["reason"] == "score" and done["next"] == 0


def test_stream_reports_gone_when_the_loop_process_is_dead(study: Path) -> None:
    """A crashed agent never writes score.json; the stream must still close with a reason."""
    srv = make_server(study, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        frames = _frames(srv, "/api/stream?source=node:iter_000:iter_000-b0-a2&after=0")
        assert frames[-1][0] == "done" and frames[-1][1]["reason"] == "gone"
        assert frames[-1][1]["next"] == 2
    finally:
        srv.shutdown()
        srv.server_close()


def test_stream_tails_growth_and_resets_on_truncation(
    study: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watch_state, "lock_owner", lambda _study: "pid 1")
    log = study / "cycles" / "iter_000" / "nodes" / "iter_000-b0-a2" / "logs" / "stdout.log"
    log.write_text("", encoding="utf-8")
    srv = make_server(study, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    seen: list[tuple] = []
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/stream?source=node:iter_000:iter_000-b0-a2&after=0",
            timeout=20,
        ) as resp:

            def read_until(predicate, limit: float = 10.0) -> None:
                deadline = time.monotonic() + limit
                name, data = None, []
                while time.monotonic() < deadline:
                    line = resp.readline().decode("utf-8").rstrip("\n")
                    if line.startswith("event: "):
                        name = line[7:]
                    elif line.startswith("data: "):
                        data.append(line[6:])
                    elif not line and name:
                        seen.append((name, json.loads("\n".join(data))))
                        name, data = None, []
                        if predicate(seen):
                            return
                raise AssertionError(f"timed out; frames={[f[0] for f in seen]}")

            read_until(lambda f: f[0][0] == "hello")
            line = json.dumps(
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "a"}]}}
            )
            log.write_text(line + "\n", encoding="utf-8")
            read_until(lambda f: any(n == "events" and d["events"] for n, d in f))
            assert [e["text"] for n, d in seen if n == "events" for e in d["events"]] == ["a"]
            log.write_text("", encoding="utf-8")
            read_until(lambda f: any(n == "reset" for n, _ in f))
            reset = next(d for n, d in seen if n == "reset")
            assert reset == {"reason": "log_truncated", "cursor": 0}
    finally:
        srv.shutdown()
        srv.server_close()


def test_stream_resets_when_the_policy_attempt_rotates(
    study: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watch_state, "lock_owner", lambda _study: "pid 1")
    srv = make_server(study, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/stream?source=policy:iter_000:1&after=0", timeout=20
        ) as resp:
            hello = None
            while hello is None:
                line = resp.readline().decode("utf-8").rstrip("\n")
                if line.startswith("event: hello"):
                    hello = json.loads(resp.readline().decode("utf-8")[len("data: ") :])
            attempt = study / "policies" / "iter_000" / "step_01" / "logs" / "attempt_1"
            _write(attempt / "stdout.log", "")
            deadline = time.monotonic() + 10
            reset = None
            while reset is None and time.monotonic() < deadline:
                line = resp.readline().decode("utf-8").rstrip("\n")
                if line == "event: reset":
                    reset = json.loads(resp.readline().decode("utf-8")[len("data: ") :])
            assert reset == {"reason": "log_rotated", "cursor": 0}
    finally:
        srv.shutdown()
        srv.server_close()


def test_stream_and_events_reject_bad_after(server) -> None:
    for bad in ("-1", "abc", "%C2%B2", "9" * 40, "1.5"):
        assert _get(server, f"/api/events?source=node:iter_000:root&after={bad}")[0] == 400
        assert _get(server, f"/api/stream?source=node:iter_000:root&after={bad}")[0] == 400
    assert _get(server, "/api/stream?source=node:..:x")[0] == 400


def test_stream_cap_refuses_extra_connections(server) -> None:
    held = [server.streams.acquire(blocking=False) for _ in range(watch_server.MAX_STREAMS)]
    assert all(held)
    try:
        status, body, _ = _get(server, "/api/stream?source=node:iter_000:root&after=0")
        assert status == 503 and "too many streams" in json.loads(body)["error"]
    finally:
        for _ in held:
            server.streams.release()


def test_non_loopback_host_is_rejected(server) -> None:
    port = server.server_address[1]
    response = _raw(
        server,
        f"GET /api/state HTTP/1.1\r\nHost: evil.example:{port}\r\nConnection: close\r\n\r\n".encode(),
    )
    assert b" 403 " in response.split(b"\r\n")[0] and b"forbidden host" in response
    ok = _raw(
        server,
        f"GET /api/state HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode(),
    )
    assert b" 200 " in ok.split(b"\r\n")[0]


def test_context_payload_lists_evidence_and_command(study: Path) -> None:
    ndir = study / "cycles" / "iter_000" / "nodes" / NODE_CELL
    _write(ndir / "evidence" / "TASK.md", "Read the evidence and write a proposal.\n")
    _json(ndir / "evidence" / "tree.json", {"nodes": [{"cell_id": NODE_CELL}]})
    _json(
        ndir / "logs" / "command.json",
        {
            "backend": "pi",
            "model": "mistral/deepseek-v4-1-flash-1017",
            "effort": "max",
            "tools": "read,write,edit",
            "cli_version": "1.5.2",
            "network": True,
            "use_bwrap": True,
            "argv": ["/usr/bin/bwrap", "--ro-bind", "/home/u/.pi", "/tmp/pi-agent", "-p", "PROMPT"],
        },
    )
    payload = watch.context_payload(study, parse_source(f"node:iter_000:{NODE_CELL}"))
    assert [f["name"] for f in payload["files"]] == ["TASK.md", "tree.json", "proposal.md"]
    assert payload["files"][0]["path"] == f"cycles/iter_000/nodes/{NODE_CELL}/evidence/TASK.md"
    assert payload["files"][2]["text"].startswith("# Int8")
    assert payload["command"] == {
        "backend": "pi",
        "model": "mistral/deepseek-v4-1-flash-1017",
        "effort": "max",
        "tools": "read,write,edit",
        "cli_version": "1.5.2",
        "network": True,
        "sandboxed": True,
    }
    assert str(study) not in json.dumps(payload)


def test_context_endpoint_ships_no_host_paths(server, study: Path) -> None:
    """Decision 7: the context payload must not leak host paths from the sandbox command line."""
    ndir = study / "cycles" / "iter_000" / "nodes" / NODE_CELL
    _write(ndir / "evidence" / "TASK.md", "Read the evidence and write a proposal.\n")
    _json(ndir / "evidence" / "tree.json", {"nodes": []})
    _json(
        ndir / "logs" / "command.json",
        {
            "backend": "pi",
            "model": "mistral/deepseek-v4-1-flash-1017",
            "effort": "max",
            "tools": "read,write,edit,bash",
            "cli_version": "1.5.2",
            "network": True,
            "use_bwrap": True,
            "argv": [
                "/usr/bin/bwrap",
                "--ro-bind",
                "/home/u/.pi/agent/auth.json",
                "/tmp/pi-agent/auth.json",
                "--bind",
                "/tmp/work",
                "/tmp/work",
                "-p",
                "PROMPT",
            ],
        },
    )
    status, body, _ = _get(server, f"/api/context?source=node:iter_000:{NODE_CELL}")
    assert status == 200
    payload = json.loads(body)
    text = json.dumps(payload)
    assert "argv" not in payload["command"]
    for marker in ("/home/", "/tmp/", "/usr/"):
        assert marker not in text, marker


def test_context_payload_for_a_policy_step(study: Path) -> None:
    pdir = study / "policies" / "iter_000"
    _write(pdir / "evidence" / "TASK.md", "Rewrite method.py.\n")
    _json(pdir / "step_01" / "logs" / "attempt_0" / "command.json", {"backend": "pi"})
    payload = watch.context_payload(study, parse_source("policy:iter_000:1"))
    assert [f["name"] for f in payload["files"]] == ["TASK.md", "candidate.py"]
    assert payload["files"][1]["text"] == "step 1\n"
    assert payload["command"]["backend"] == "pi"


def test_symlinks_outside_the_study_are_not_served(study: Path) -> None:
    """H1: an agent-authored symlink must not turn the watch server into a file read primitive."""
    outside = study.parent / "outside.txt"
    outside.write_text("SECRET\n", encoding="utf-8")
    ndir = study / "cycles" / "iter_000" / "nodes" / NODE_CELL
    (ndir / "proposal.md").unlink()
    (ndir / "proposal.md").symlink_to(outside)
    (ndir / "evidence").mkdir(parents=True, exist_ok=True)
    (ndir / "evidence" / "TASK.md").symlink_to(outside)
    assert read_artifact(study, parse_source(f"node:iter_000:{NODE_CELL}"), "proposal") is None
    payload = watch.context_payload(study, parse_source(f"node:iter_000:{NODE_CELL}"))
    assert payload["files"] == []
    log = ndir / "logs" / "stdout.log"
    log.unlink()
    log.symlink_to(outside)
    result = watch.events_payload(study, parse_source(f"node:iter_000:{NODE_CELL}"), 0)
    assert result["events"] == [] and result["error"] == "log path outside study"


def test_symlinked_cycle_directory_is_skipped(study: Path) -> None:
    outside = study.parent / "outside-cycle"
    (outside / "nodes").mkdir(parents=True)
    (study / "cycles" / "iter_999").symlink_to(outside)
    state = build_state(study)
    cycle = next(c for c in state["cycles"] if c["cycle_id"] == "iter_999")
    assert [(n["cell_id"], n["status"]) for n in cycle["nodes"]] == [("root", "done")]


def test_agent_failure_fail_class_passes_through(study: Path) -> None:
    ndir = study / "cycles" / "iter_000" / "nodes" / "iter_000-b0-a1"
    (ndir / "eval").rmdir()
    _json(
        ndir / "score.json",
        {
            "cell_id": "iter_000-b0-a1",
            "fail_class": "agent_failure",
            "score": None,
            "successes": 0,
            "episodes": 0,
            "error": "Agent exited 1: No API key found for mistral.",
            "agent_timed_out": False,
        },
    )
    node = {n["cell_id"]: n for n in build_state(study)["cycles"][0]["nodes"]}["iter_000-b0-a1"]
    assert node["status"] == "done" and node["fail_class"] == "agent_failure"
    assert "No API key" in node["error"] and node["score"] is None


def test_state_reports_batch_composition_and_live_timer(
    study: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watch_state, "lock_owner", lambda _study: "pid 1")
    _json(
        study / "cycles" / "iter_000" / "live_cycle_manifest.json",
        {
            "decision_rounds": 2,
            "opened_branches": [0, 1],
            "probes_used": 5,
            "effective_sequential_rounds": 3,
            "agent_timeout_s": 900.0,
        },
    )
    cycle = build_state(study)["cycles"][0]
    assert cycle["opened_branches"] == [0, 1]
    assert cycle["probes_used"] == 5
    assert cycle["effective_sequential_rounds"] == 3
    assert cycle["agent_timeout_s"] == 900.0
    nodes = {n["cell_id"]: n for n in cycle["nodes"]}
    live = nodes["iter_000-b0-a2"]
    assert live["started_utc"] is not None and live["elapsed_s"] is not None
    assert live["elapsed_s"] >= 0
    assert nodes[NODE_CELL]["started_utc"] is None and nodes[NODE_CELL]["elapsed_s"] is None


def test_dead_loop_demotes_an_in_flight_node_to_stopped(study: Path) -> None:
    """A node with a log but no score cannot still be `agent` once the lock holder is gone."""
    node = {n["cell_id"]: n for n in build_state(study)["cycles"][0]["nodes"]}["iter_000-b0-a2"]
    assert node["status"] == "stopped"
    assert node["started_utc"] is not None and node["elapsed_s"] is None
    assert build_state(study)["loop"]["running"] is False


def test_running_loop_keeps_an_in_flight_node_as_agent(
    study: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watch_state, "lock_owner", lambda _study: "pid 1")
    state = build_state(study)
    assert state["loop"]["running"] is True and state["loop"]["status"] == "running"
    nodes = {n["cell_id"]: n for n in state["cycles"][0]["nodes"]}
    assert nodes["iter_000-b0-a2"]["status"] == "agent"
    assert nodes["iter_000-b0-a1"]["status"] == "evaluating"
    assert nodes["iter_000-b0-a2"]["elapsed_s"] is not None
    assert nodes[NODE_CELL]["status"] == "done"
    assert nodes["iter_000-b0-a3"]["status"] == "pending"


def test_loop_state_error_is_reported_as_error(study: Path) -> None:
    _write(
        study / "loop_state.json",
        json.dumps(
            {
                "status": "error",
                "stop_reason": "agent exited 1",
                "cycles_target": 1,
                "iterations": [],
            }
        )
        + "\n",
    )
    loop = build_state(study)["loop"]
    assert loop["status"] == "error" and loop["running"] is False
    assert loop["stop_reason"] == "agent exited 1"


def test_run_watch_refuses_a_non_loopback_bind() -> None:
    with pytest.raises(SystemExit, match="not loopback"):
        watch.run_watch("/tmp/does-not-matter", host="0.0.0.0", port=0)


def test_broken_pipe_during_the_error_response_is_silent(
    server, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """V3 check 7: a dead client must not print a traceback with absolute paths to stderr."""
    calls = {"n": 0}

    def failing_send(self, status, body, ctype):
        calls["n"] += 1
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(watch_server._Handler, "_send", failing_send)
    port = server.server_address[1]
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(f"GET /api/state HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
        while sock.recv(65536):
            pass
    assert calls["n"] == 2, "the 500 fallback must have been attempted and swallowed"
    err = capsys.readouterr().err
    assert err == "", err
    assert "/home/" not in err and "/tmp/" not in err


def test_server_handle_error_silences_only_a_vanished_client(
    server, capsys: pytest.CaptureFixture[str]
) -> None:
    try:
        raise ConnectionResetError(104, "Connection reset by peer")
    except ConnectionResetError:
        server.handle_error(None, ("127.0.0.1", 1))
    assert capsys.readouterr().err == ""
    try:
        raise ValueError("a real bug")
    except ValueError:
        server.handle_error(None, ("127.0.0.1", 1))
    assert "ValueError" in capsys.readouterr().err


def test_static_assets_are_allowlisted(
    study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page is a module tree: css/js are served, everything else is not."""
    root = tmp_path / "static"
    (root / "css").mkdir(parents=True)
    (root / "js" / "views").mkdir(parents=True)
    _write(root / "index.html", "<!doctype html><title>t</title>")
    _write(root / "css" / "x.css", "body { color: red; }")
    _write(root / "js" / "main.js", "export const x = 1;\n")
    _write(root / "js" / "views" / "v.js", "export const y = 2;\n")
    _write(root / "fixture.json", "{}")
    _write(root / "secret.txt", "SECRET")
    _write(root / "package.json", "{}")
    outside = tmp_path / "outside.js"
    _write(outside, "SECRET")
    (root / "js" / "escape.js").symlink_to(outside)
    monkeypatch.setattr(watch_assets, "STATIC_INDEX", root / "index.html")
    srv = make_server(study, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        status, body, headers = _get(srv, "/js/main.js")
        assert status == 200 and body == b"export const x = 1;\n"
        assert headers["Content-Type"].startswith("text/javascript")
        status, _, headers = _get(srv, "/js/views/v.js")
        assert status == 200 and headers["Content-Type"].startswith("text/javascript")
        status, _, headers = _get(srv, "/css/x.css")
        assert status == 200 and headers["Content-Type"].startswith("text/css")
        status, _, headers = _get(srv, "/fixture.json")
        assert status == 200 and headers["Content-Type"].startswith("application/json")
        assert _get(srv, "/secret.txt")[0] == 404
        assert _get(srv, "/package.json")[0] == 404
        assert _get(srv, "/js/escape.js")[0] == 404
        port = srv.server_address[1]
        for path in ("/js/../watch.py", "/js/%2e%2e/watch.py", "/js/..%2fwatch.py", "/js//main.js"):
            raw = _raw(
                srv,
                f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode(),
            )
            assert b" 404 " in raw.split(b"\r\n")[0], path
    finally:
        srv.shutdown()
        srv.server_close()


def test_static_index_is_the_module_page() -> None:
    """The committed page is served as ES modules, not one inline script."""
    srv = make_server(Path("."), "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        status, body, headers = _get(srv, "/")
        assert status == 200 and headers["Content-Type"].startswith("text/html")
        assert b'type="module"' in body and b"js/main.js" in body
        assert b"css/tokens.css" in body
        assert b"<script>" not in body
    finally:
        srv.shutdown()
        srv.server_close()


def test_context_sections_for_a_node(study: Path) -> None:
    """Context is split into the sections the agent actually received, with sizes."""
    ndir = study / "cycles" / "iter_000" / "nodes" / NODE_CELL
    task = "Read the evidence and write a proposal.\n"
    _write(ndir / "evidence" / "TASK.md", task)
    _json(ndir / "evidence" / "tree.json", {"nodes": [{"cell_id": "root"}]})
    _write(ndir / "evidence" / "nodes" / "root" / "proposal.md", "# Baseline\n")
    _json(ndir / "logs" / "command.json", {"backend": "pi", "model": "m", "use_bwrap": True})
    payload = watch.context_payload(study, parse_source(f"node:iter_000:{NODE_CELL}"))
    sections = {section["id"]: section for section in payload["sections"]}
    assert set(sections) == {"system", "message", "tree", "earlier", "command"}
    assert sections["system"]["recorded"] is False
    assert sections["system"]["text"] is None and sections["system"]["note"]
    assert sections["message"]["text"] == task
    assert sections["message"]["chars"] == len(task)
    assert sections["message"]["tokens"] == (len(task) + 3) // 4
    assert sections["tree"]["path"].endswith("evidence/tree.json")
    earlier = sections["earlier"]["items"]
    assert earlier[0]["cell_id"] == "root"
    assert earlier[0]["files"][0]["name"] == "proposal.md"
    assert sections["command"]["fields"]["backend"] == "pi"
    assert sections["command"]["fields"]["sandboxed"] is True
    assert str(study) not in json.dumps(payload)


def test_context_sections_for_a_policy_step(study: Path) -> None:
    pdir = study / "policies" / "iter_000"
    _write(pdir / "evidence" / "TASK.md", "Rewrite method.py.\n")
    _write(pdir / "step_01" / "workspace" / "method.py", "class OptimalPolicy: pass\n")
    _write(pdir / "step_01" / "candidate.py", "class OptimalPolicy:\n    pass\n")
    _json(pdir / "evidence" / "history" / "step_01" / "verdict.json", {"status": "accepted"})
    payload = watch.context_payload(study, parse_source("policy:iter_000:1"))
    sections = {section["id"]: section for section in payload["sections"]}
    assert sections["seed"]["text"].startswith("class OptimalPolicy")
    assert sections["candidate"]["text"].startswith("class OptimalPolicy")
    assert sections["system"]["recorded"] is False
    assert {item["name"] for item in sections["evidence"]["items"]} >= {"verdict.json"}


def test_policy_seed_artifact(study: Path) -> None:
    pdir = study / "policies" / "iter_000"
    _write(pdir / "step_01" / "workspace" / "method.py", "seed\n")
    step = parse_source("policy:iter_000:1")
    assert read_artifact(study, step, "seed") == "seed\n"
    assert read_artifact(study, parse_source(f"node:iter_000:{NODE_CELL}"), "seed") is None


def test_loop_reports_the_probe_budget(study: Path) -> None:
    """The title bar shows the run's probe budget, not the grid maximum."""
    raw = json.loads((study / "loop_state.json").read_text(encoding="utf-8"))
    raw["run"] = {"probe_budget": 4, "max_parallelism": 2}
    _json(study / "loop_state.json", raw)
    assert build_state(study)["loop"]["probe_budget"] == 4
    raw.pop("run")
    raw["first_run"] = {"probe_budget": 6}
    _json(study / "loop_state.json", raw)
    assert build_state(study)["loop"]["probe_budget"] == 6
    raw.pop("first_run")
    _json(study / "loop_state.json", raw)
    assert build_state(study)["loop"]["probe_budget"] is None
