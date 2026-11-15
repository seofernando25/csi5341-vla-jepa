"""Static checks for the live watch page: one offline file, fixtures that match watch_api.md."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "rsi" / "watch_static"
INDEX = STATIC / "index.html"
FIXTURE = STATIC / "fixture.json"
FIXTURE_EVENTS = STATIC / "fixture_events.json"
PROTOCOL = ROOT / "studies" / "program-v1" / "protocol.json"

STATE_FIELDS = {"study_dir", "now", "loop", "protocol", "gpu", "cycles"}
LOOP_FIELDS = {
    "status",
    "stop_reason",
    "iteration",
    "stage",
    "cycles_target",
    "elapsed_s",
    "agent_cost_usd",
    "best_score",
    "running",
}
PROTOCOL_FIELDS = {
    "smoke_gate",
    "smoke_trials",
    "dev_gate",
    "dev_trials",
    "max_parallelism",
    "rounds_cap",
    "max_branches",
    "max_attempts_per_branch",
    "gpu_slots",
    "agent",
}
CYCLE_FIELDS = {
    "cycle_id",
    "iteration",
    "status",
    "decision_rounds",
    "policy_sha256",
    "nodes",
    "policy_dev",
}
POLICY_DEV_FIELDS = {
    "status",
    "selected_step",
    "incumbent_v",
    "selected_v",
    "heldout_delta",
    "steps",
}
STEP_FIELDS = {"step", "status", "mean_v", "delta_vs_best", "reason"}
NODE_FIELDS = {
    "cell_id",
    "parent_id",
    "branch",
    "attempt",
    "round",
    "status",
    "fail_class",
    "score",
    "size_reduction",
    "successes",
    "episodes",
    "resident_mb",
    "commit",
    "agent_seconds",
    "eval_seconds",
    "agent_cost_usd",
    "agent_turns",
    "agent_timed_out",
    "error",
    "headline",
    "promoted",
}
EVENT_FIELDS = {
    "i",
    "t",
    "kind",
    "text",
    "tool",
    "tool_use_id",
    "input",
    "output",
    "truncated",
    "is_error",
    "meta",
}
EVENT_KINDS = {"text", "thinking", "tool_call", "tool_result", "result", "system"}
LOOP_STATUSES = {"idle", "running", "stopped", "done", "error"}
NODE_STATUSES = {"pending", "agent", "evaluating", "stopped", "done"}
STEP_STATUSES = {"running", "accepted", "not_better", "rejected"}
ARTIFACT_NAMES = {"proposal", "diff", "score", "error", "solution", "policy", "verdict", "curve"}


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def _script_body(html: str) -> str:
    match = re.search(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    assert match, "index.html has no inline script"
    return match.group(1)


def _strip_comments(html: str) -> str:
    return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)


def _check_keys(obj: object, required: set[str], where: str) -> None:
    assert isinstance(obj, dict), f"{where} is not an object"
    missing = required - set(obj)
    assert not missing, f"{where} is missing {sorted(missing)}"


def test_index_is_one_offline_file_under_200kb() -> None:
    assert INDEX.is_file()
    assert INDEX.stat().st_size < 200 * 1024


def test_index_makes_no_remote_requests() -> None:
    html = _html()
    outside = _strip_comments(html)
    assert not re.search(r"https?://", outside), "remote URL outside comments"
    assert not re.search(r"<script[^>]+\bsrc\s*=", html, flags=re.IGNORECASE)
    assert not re.search(r"<link[^>]+\bhref\s*=\s*[\"']?https?:", html, flags=re.IGNORECASE)
    assert not re.search(r"@import|url\(\s*[\"']?https?:", html)


def test_index_never_assigns_markup_from_strings() -> None:
    assert "innerHTML" not in _html()
    assert "outerHTML" not in _html()
    assert "insertAdjacentHTML" not in _html()


def test_fixture_state_has_every_contract_field() -> None:
    state = json.loads(FIXTURE.read_text(encoding="utf-8"))
    _check_keys(state, STATE_FIELDS, "State")
    _check_keys(state["loop"], LOOP_FIELDS, "State.loop")
    assert state["loop"]["status"] in LOOP_STATUSES
    _check_keys(state["protocol"], PROTOCOL_FIELDS, "State.protocol")
    _check_keys(state["protocol"]["agent"], {"backend", "model", "effort"}, "State.protocol.agent")
    assert isinstance(state["cycles"], list) and state["cycles"]
    for cycle in state["cycles"]:
        _check_keys(cycle, CYCLE_FIELDS, "Cycle")
        _check_keys(cycle["policy_dev"], POLICY_DEV_FIELDS, "Cycle.policy_dev")
        for step in cycle["policy_dev"]["steps"]:
            _check_keys(step, STEP_FIELDS, "policy_dev step")
            assert step["status"] in STEP_STATUSES
        assert any(n["cell_id"] == "root" and n["parent_id"] is None for n in cycle["nodes"])
        for node in cycle["nodes"]:
            _check_keys(node, NODE_FIELDS, "Node")
            assert node["status"] in NODE_STATUSES


def test_fixture_protocol_matches_the_frozen_protocol() -> None:
    """The demo renders these numbers; they must be the frozen protocol's, not a stale copy."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))["protocol"]
    frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    plan = frozen["dream_rsi"]["grid_plan"]
    shared = {
        "smoke_gate": frozen["gates"]["smoke_min_successes"],
        "smoke_trials": frozen["phases"]["smoke"]["trials"],
        "dev_gate": frozen["gates"]["development_min_successes"],
        "dev_trials": frozen["phases"]["development"]["trials"],
        "max_parallelism": plan["max_parallelism"],
        "rounds_cap": plan["rounds_cap"],
        "max_branches": plan["max_branches"],
        "max_attempts_per_branch": plan["max_attempts_per_branch"],
        "gpu_slots": frozen["dream_rsi"]["gpu_slots"],
    }
    for field, expected in shared.items():
        assert fixture[field] == expected, f"fixture protocol {field} disagrees with protocol.json"
    agent = frozen["dream_rsi"]["agent"]
    assert fixture["agent"] == {key: agent[key] for key in ("backend", "model", "effort")}


def test_fixture_is_free_of_machine_specific_paths() -> None:
    """The demo is committed: no host, temp or home paths may leak into it."""
    markers = ("/home/", "/tmp/opencode/", "/Users/")
    for path in (FIXTURE, FIXTURE_EVENTS):
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            assert marker not in text, f"{path.name} contains a machine path: {marker}"
    state = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert not Path(state["study_dir"]).is_absolute(), state["study_dir"]


def test_fixture_demo_scores_are_not_pilot_results() -> None:
    """The demo must not reuse the 4-node pilot's 25.4/38.6/44.2 % scores."""
    pilot = {0.254, 0.386, 0.442}
    state = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert state["loop"]["best_score"] not in pilot
    scores = {
        n["score"]
        for cycle in state["cycles"]
        for n in cycle["nodes"]
        if n["score"] not in (None, 0.0)
    }
    assert scores, "the demo has no scored node"
    assert not scores & pilot, f"demo reuses pilot scores {sorted(scores & pilot)}"
    # Event text quotes the same scores as decimals and as percentages.
    tokens = {f"{v:.3f}" for v in pilot} | {f"{v * 100:.1f}" for v in pilot}
    for path in (FIXTURE, FIXTURE_EVENTS):
        text = path.read_text(encoding="utf-8")
        found = sorted(t for t in tokens if t in text)
        assert not found, f"{path.name} quotes pilot scores {found}"


def test_fixture_covers_the_brief_scenario() -> None:
    state = json.loads(FIXTURE.read_text(encoding="utf-8"))
    done_cycle, running_cycle = state["cycles"][0], state["cycles"][1]
    assert done_cycle["status"] == "done"
    assert running_cycle["status"] == "running"
    passed = [n for n in done_cycle["nodes"] if n["fail_class"] == "ok" and n["cell_id"] != "root"]
    assert len(passed) == 3
    assert any(n["fail_class"] == "implementation_failure" for n in done_cycle["nodes"])
    statuses = sorted(n["status"] for n in running_cycle["nodes"] if n["cell_id"] != "root")
    assert statuses == ["agent", "evaluating", "pending", "pending"]
    assert done_cycle["policy_dev"]["status"] == "done"
    assert [s["status"] for s in done_cycle["policy_dev"]["steps"]] == [
        "accepted",
        "not_better",
        "rejected",
    ]


def test_fixture_events_have_every_contract_field() -> None:
    doc = json.loads(FIXTURE_EVENTS.read_text(encoding="utf-8"))
    assert set(doc) >= {"live", "events", "artifacts"}
    live_events = doc["events"][doc["live"][0]]
    assert len(live_events) >= 35
    assert any(e["is_error"] for e in live_events if e["kind"] == "tool_result")
    for source, events in doc["events"].items():
        assert re.fullmatch(r"(node|policy):[^:]+:.+", source), source
        for index, ev in enumerate(events):
            _check_keys(ev, EVENT_FIELDS, f"Event in {source}")
            assert ev["i"] == index, f"event index not monotonic in {source}"
            assert ev["kind"] in EVENT_KINDS
    for source, arts in doc["artifacts"].items():
        for name in arts:
            assert name in ARTIFACT_NAMES, name


def test_fixture_events_reference_live_sources() -> None:
    doc = json.loads(FIXTURE_EVENTS.read_text(encoding="utf-8"))
    state = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cycle = state["cycles"][-1]
    running = {n["cell_id"] for n in cycle["nodes"] if n["status"] in {"agent", "evaluating"}}
    assert doc["live"] == [f"node:{cycle['cycle_id']}:c2-b1a1"]
    assert doc["live"][0].split(":")[2] in running


def test_transcript_coalesces_streamed_deltas() -> None:
    """One block per delta run: the parser emits an event per token, the page must not."""
    body = _script_body(_html())
    assert "function flushDeltas" in body
    assert "DELTA_FLUSH_MS" in body
    # Deltas have one renderer; eventEl must not build a block per text or thinking event.
    el = body[body.index("function eventEl") :]
    el = el[: el.index("function attachResult")]
    assert "case 'thinking'" not in el
    assert "case 'text'" not in el
    # The demo fixture carries a delta run, so the offline page exercises the coalescer.
    doc = json.loads(FIXTURE_EVENTS.read_text(encoding="utf-8"))
    live = doc["events"][doc["live"][0]]
    longest = run = 0
    for ev in live:
        run = run + 1 if ev["kind"] == "thinking" else 0
        longest = max(longest, run)
    assert longest >= 20, "the demo transcript has no delta run to coalesce"


def test_display_strips_object_stringification() -> None:
    """A structured value must never reach a text node as `[object Object]`."""
    body = _script_body(_html())
    assert "function textOf" in body
    for call in (
        "queueDelta(ev.kind, ev.text)",
        "textOf(ev.text)",
        "textOf(ev.output)",
        "e.textContent = textOf(text)",
        "var clean = textOf(text)",
        "body.textContent = clean",
        "textOf(task.text)",
    ):
        assert call in body, call
    doc = json.loads(FIXTURE_EVENTS.read_text(encoding="utf-8"))
    live = doc["events"][doc["live"][0]]
    assert any("[object Object]" in (e["text"] or "") for e in live), (
        "the demo transcript has no stringified-object case"
    )
    assert any(
        "[object Object]" in (text or "")
        for arts in doc["artifacts"].values()
        for text in arts.values()
    ), "the demo artifacts have no stringified-object case"


def test_state_selectors_are_single_sourced() -> None:
    """One reader for (loop, node), one writer for the footer, no raw status comparisons."""
    body = _script_body(_html())
    for name in ("function loopLive", "function nodePhase", "function footLabel"):
        assert name in body, name
    assert "status === 'agent'" not in body
    assert "status === 'evaluating'" not in body
    assert "setTxEnd" not in body
    assert body.count("$('tx-end')") == 1, "the footer has more than one writer"
    assert "ev msg think live" not in body, "a thinking block must not be born live"
    assert "row.classList.add('live')" in body
    hello = body[body.index("on('hello'") : body.index("on('events'")]
    assert "txFrozen = false" not in hello, "a reconnect must not un-freeze the transcript"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_state_block_runs_under_node(tmp_path: Path) -> None:
    """The selectors are DOM-free: run them and check the audit's table and labels."""
    body = _script_body(_html())
    start = body.index("/* state:begin")
    end = body.index("/* state:end */") + len("/* state:end */")
    block = body[start:end]
    assert "document" not in block and "S." not in block
    script = (
        block
        + """
const stopped = {loop: {status: 'stopped', running: false}};
const running = {loop: {status: 'running', running: true}};
const idle = {loop: {status: 'idle', running: false}};
const failed = {loop: {status: 'error', running: false}};
const out = {
  liveRunning: loopLive(running),
  liveStopped: loopLive(stopped),
  liveIdle: loopLive(idle),
  liveError: loopLive(failed),
  endedError: loopEnded(failed),
  endedRunning: loopEnded(running),
  agentStopped: nodePhase({status: 'agent'}, stopped),
  agentRunning: nodePhase({status: 'agent'}, running),
  evaluatingIdle: nodePhase({status: 'evaluating'}, idle),
  doneStays: nodePhase({status: 'done', fail_class: 'ok'}, stopped),
  pendingStays: nodePhase({status: 'pending'}, stopped),
  stoppedStays: nodePhase({status: 'stopped'}, running),
  gone: footLabel({frozen: true, status: 'gone', live: false, empty: false}),
  goneAfterHello: footLabel({frozen: true, status: 'gone', live: false, empty: false}),
  closed: footLabel({frozen: true, status: 'open', live: false, ended: true, empty: false}),
  settled: footLabel({frozen: true, status: 'open', live: false, ended: false, phase: 'done', empty: false}),
  liveNode: footLabel({frozen: true, status: 'open', live: false, ended: false, phase: 'agent', empty: false}),
  live: footLabel({frozen: false, status: 'open', live: true, empty: false}),
  empty: footLabel({frozen: true, status: 'gone', live: false, empty: true}),
  quiet: footLabel({frozen: false, status: 'open', live: false, empty: false}),
  closedLoop: footLabel({frozen: false, status: 'open', live: false, ended: true, empty: false}),
};
console.log(JSON.stringify(out));
"""
    )
    path = tmp_path / "state_block.js"
    path.write_text(script, encoding="utf-8")
    # Hermetic: no ambient NODE_OPTIONS, no repo cwd, a generous timeout, and no retry.
    env = {key: value for key, value in os.environ.items() if key != "NODE_OPTIONS"}
    try:
        result = subprocess.run(
            ["node", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            cwd=tmp_path,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"node did not finish the state block within 30 s: {path}")
    except OSError as exc:
        pytest.fail(f"node could not run the state block {path}: {exc}")
    detail = (
        f"node exited {result.returncode} for {path}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0, detail
    assert result.stdout.strip(), f"node printed no result\n{detail}"
    out = json.loads(result.stdout)
    assert out["liveRunning"] is True and out["liveStopped"] is False
    assert out["liveIdle"] is False and out["liveError"] is False
    assert out["endedError"] is True and out["endedRunning"] is False
    assert out["agentStopped"] == "stopped"
    assert out["agentRunning"] == "agent"
    assert out["evaluatingIdle"] == "stopped"
    assert out["doneStays"] == "done" and out["pendingStays"] == "pending"
    assert out["stoppedStays"] == "stopped"
    assert out["gone"] == out["goneAfterHello"]
    assert out["gone"].startswith("Stream ended: the loop process is gone.")
    assert out["closed"] == "Transcript closed: the loop is no longer running."
    assert out["settled"] == "Transcript ended: this node has finished."
    assert out["liveNode"] == ""
    assert out["live"] == "Live. New events appear as the agent works."
    assert out["empty"] == "No transcript was recorded for this source."
    assert out["quiet"] == ""
    assert out["closedLoop"] == "Transcript closed: the loop is no longer running."


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_inline_script_parses_with_node(tmp_path: Path) -> None:
    path = tmp_path / "watch_inline.js"
    path.write_text(_script_body(_html()), encoding="utf-8")
    result = subprocess.run(
        ["node", "--check", str(path)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
