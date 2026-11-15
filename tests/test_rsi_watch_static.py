"""Static checks for the multi-file watch workbench: offline assets, fixtures, pure modules."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from rsi import watch

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "rsi" / "watch_static"
INDEX = STATIC / "index.html"
JS = STATIC / "js"
CSS = STATIC / "css"
FIXTURE = STATIC / "fixture.json"
FIXTURE_EVENTS = STATIC / "fixture_events.json"
PROTOCOL = ROOT / "studies" / "program-v1" / "protocol.json"


def _fields(names: str) -> set[str]:
    return set(names.split())


STATE_FIELDS = _fields("study_dir now loop protocol gpu cycles")
LOOP_FIELDS = _fields(
    "status stop_reason iteration stage cycles_target elapsed_s agent_cost_usd best_score running"
)
PROTOCOL_FIELDS = _fields(
    "smoke_gate smoke_trials dev_gate dev_trials max_parallelism rounds_cap max_branches "
    "max_attempts_per_branch gpu_slots agent"
)
CYCLE_FIELDS = _fields("cycle_id iteration status decision_rounds policy_sha256 nodes policy_dev")
POLICY_DEV_FIELDS = _fields("status selected_step incumbent_v selected_v heldout_delta steps")
STEP_FIELDS = _fields("step status mean_v delta_vs_best reason")
NODE_FIELDS = _fields(
    "cell_id parent_id branch attempt round status fail_class score size_reduction successes "
    "episodes resident_mb commit agent_seconds eval_seconds agent_cost_usd agent_turns "
    "agent_timed_out error headline promoted"
)
EVENT_FIELDS = _fields("i t kind text tool tool_use_id input output truncated is_error meta")
SECTION_FIELDS = _fields("id title recorded text chars tokens path note items fields")
EVENT_KINDS = {"text", "thinking", "tool_call", "tool_result", "result", "system"}
LOOP_STATUSES = {"idle", "running", "stopped", "done", "error"}
NODE_STATUSES = {"pending", "agent", "evaluating", "stopped", "done"}
STEP_STATUSES = {"running", "accepted", "not_better", "rejected"}
PURE_MODULES = (
    "store.js",
    "reducer.js",
    "selectors.js",
    "actions.js",
    "format.js",
    "views/meta.js",
)
FORBIDDEN_IN_PURE = ("document", "window", "localStorage", "fetch")
FORBIDDEN_IN_REDUCER = ("Date.now", "Math.random")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_comments(text: str, style: str) -> str:
    """Drop block comments; for JS also whole-line `//` comments. Strings stay visible."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    if style == "html":
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    elif style == "js":
        text = re.sub(r"(?m)^\s*//.*$", "", text)
    return text


def _remote_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s\"'<>)]+", text)
    return [url for url in urls if not url.startswith(SVG_NAMESPACE)]


def _asset_refs(html: str) -> list[str]:
    refs = re.findall(r"<script\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)", html, flags=re.IGNORECASE)
    for attrs in re.findall(r"<link\b([^>]*)>", html, flags=re.IGNORECASE):
        if re.search(r"\brel\s*=\s*[\"']?stylesheet", attrs, flags=re.IGNORECASE):
            refs += re.findall(r"\bhref\s*=\s*[\"']([^\"']+)", attrs, flags=re.IGNORECASE)
    return refs


def _imports(path: Path) -> list[str]:
    code = _strip_comments(_read(path), "js")
    return re.findall(r"""(?:\bfrom\s+|\bimport\s*\(\s*|\bimport\s+)["']([^"']+)["']""", code)


def _check_keys(obj: object, required: set[str], where: str) -> None:
    assert isinstance(obj, dict), f"{where} is not an object"
    missing = required - set(obj)
    assert not missing, f"{where} is missing {sorted(missing)}"


def _leaves(section: dict, where: str) -> list[dict]:
    out = []
    for item in section.get("items") or []:
        for leaf in item.get("files") or [item]:
            label = f"{where}/{leaf.get('name')}"
            assert isinstance(leaf.get("text"), str), f"{label}: no text"
            assert leaf["chars"] == len(leaf["text"]), f"{label}: chars != text length"
            assert leaf["tokens"] == (leaf["chars"] + 3) // 4, f"{label}: tokens != ceil(chars/4)"
            out.append(leaf)
    return out


def _check_stats(obj: dict, where: str) -> None:
    leaves = _leaves(obj, where)
    chars, tokens = obj["chars"], obj["tokens"]
    assert isinstance(chars, int) and isinstance(tokens, int), f"{where}: counts are not integers"
    text = obj.get("text")
    if text is not None:
        assert isinstance(text, str), f"{where}: text is not a string"
        assert chars == len(text), f"{where}: chars {chars} != text length {len(text)}"
        assert tokens == (chars + 3) // 4, f"{where}: tokens != ceil(chars/4)"
    elif obj["recorded"]:
        assert chars == sum(leaf["chars"] for leaf in leaves), f"{where}: chars != sum(items)"
        assert tokens == sum(leaf["tokens"] for leaf in leaves), f"{where}: tokens != sum(items)"
    else:
        assert chars == 0 and tokens == 0, f"{where}: an unrecorded section has counts"


def test_index_is_a_small_offline_page() -> None:
    assert INDEX.is_file() and INDEX.stat().st_size < 200 * 1024
    html = _read(INDEX)
    scripts = re.findall(
        r"<script\b([^>]*)>(.*?)</script\s*>", html, flags=re.IGNORECASE | re.DOTALL
    )
    assert scripts, "index.html loads no script"
    for attrs, body in scripts:
        assert re.search(r"\bsrc\s*=", attrs, flags=re.IGNORECASE), f"inline <script{attrs}>"
        assert not body.strip(), f"inline script body: {body.strip()[:60]!r}"
        kind = re.search(r"\btype\s*=\s*[\"']([^\"']+)", attrs, flags=re.IGNORECASE)
        if kind:
            assert kind.group(1).lower() == "module", f"unexpected script type {kind.group(1)}"
    assert any(re.search(r"\btype\s*=\s*[\"']module", attrs, re.IGNORECASE) for attrs, _ in scripts)


def test_page_makes_no_remote_requests() -> None:
    html = re.sub(r"data:[^\"<>]*", "", _strip_comments(_read(INDEX), "html"))
    urls = _remote_urls(html)
    assert not urls, f"remote URLs outside comments: {urls}"
    assert not re.search(r"@import", html, flags=re.IGNORECASE)
    assert not re.search(r"url\(\s*[\"']?https?:", html, flags=re.IGNORECASE)
    for href in re.findall(r"<link\b[^>]*\bhref\s*=\s*[\"']([^\"']+)", html, flags=re.IGNORECASE):
        assert not re.match(r"(?:https?:)?//", href, flags=re.IGNORECASE), f"external link {href}"
    for path in sorted(JS.rglob("*.js")) + sorted(CSS.rglob("*.css")):
        text = _strip_comments(_read(path), "js" if path.suffix == ".js" else "css")
        urls = _remote_urls(text)
        assert not urls, f"{path.relative_to(STATIC)} references {urls}"
        assert "@import" not in text, f"{path.relative_to(STATIC)} uses @import"
        assert not re.search(r"url\(\s*[\"']?https?:", text, flags=re.IGNORECASE), path


def test_index_assets_are_relative_and_allowlisted() -> None:
    refs = _asset_refs(_read(INDEX))
    assert "js/main.js" in refs and "js/theme-boot.js" in refs
    for ref in refs:
        assert not re.match(r"^[a-z][a-z0-9+.-]*:|^//", ref, flags=re.IGNORECASE), (
            f"remote asset {ref}"
        )
        assert ".." not in Path(ref).parts, f"asset escapes the static root: {ref}"
        target = STATIC / ref
        assert target.is_file(), f"{ref} does not exist"
        served = watch.static_path("/" + ref)
        assert served is not None, f"{ref} is not served by static_path"
        assert served.resolve() == target.resolve(), f"{ref} resolves outside the static tree"


def test_no_markup_injection_helpers() -> None:
    bad = re.compile(
        r"\binnerHTML\b|\bouterHTML\b|\binsertAdjacentHTML\b|document\s*\.\s*write\s*\("
        r"|\beval\s*\(|new\s+Function\b"
    )
    for path in sorted(JS.rglob("*.js")):
        match = bad.search(_strip_comments(_read(path), "js"))
        assert not match, f"{path.relative_to(STATIC)} uses {match.group(0)}"
    assert not bad.search(_strip_comments(_read(INDEX), "html"))


def test_every_module_is_referenced() -> None:
    modules = sorted(JS.rglob("*.js"))
    assert modules
    referenced = set()
    for path in modules:
        for spec in _imports(path):
            target = (path.parent / spec).resolve()
            assert spec.startswith(".") and target.is_file(), (
                f"{path.relative_to(STATIC)}: bad import {spec}"
            )
            referenced.add(target)
    for path in sorted(JS.glob("*.js")):
        if path.name not in {"main.js", "theme-boot.js", "demo.js"}:
            assert path.resolve() in referenced, f"orphan top-level module {path.name}"
    linked = {ref for ref in _asset_refs(_read(INDEX)) if ref.endswith(".css")}
    assert {f"css/{path.name}" for path in CSS.glob("*.css")} <= linked


def test_fixture_state_has_every_contract_field() -> None:
    state = json.loads(_read(FIXTURE))
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
    fixture = json.loads(_read(FIXTURE))["protocol"]
    frozen = json.loads(_read(PROTOCOL))
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
    for path in (FIXTURE, FIXTURE_EVENTS):
        text = _read(path)
        for marker in ("/home/", "/tmp/opencode/", "/Users/"):
            assert marker not in text, f"{path.name} contains a machine path: {marker}"
    state = json.loads(_read(FIXTURE))
    assert not Path(state["study_dir"]).is_absolute(), state["study_dir"]


def test_fixture_demo_scores_are_not_pilot_results() -> None:
    """The demo must not reuse the 4-node pilot's 25.4/38.6/44.2 % scores."""
    pilot = {0.254, 0.386, 0.442}
    state = json.loads(_read(FIXTURE))
    assert state["loop"]["best_score"] not in pilot
    scores = {
        n["score"]
        for cycle in state["cycles"]
        for n in cycle["nodes"]
        if n["score"] not in (None, 0.0)
    }
    assert scores, "the demo has no scored node"
    assert not scores & pilot, f"demo reuses pilot scores {sorted(scores & pilot)}"
    tokens = {f"{value:.3f}" for value in pilot} | {f"{value * 100:.1f}" for value in pilot}
    for path in (FIXTURE, FIXTURE_EVENTS):
        found = sorted(token for token in tokens if token in _read(path))
        assert not found, f"{path.name} quotes pilot scores {found}"


def test_fixture_covers_the_brief_scenario() -> None:
    state = json.loads(_read(FIXTURE))
    done = [
        cycle
        for cycle in state["cycles"]
        if cycle["status"] == "done"
        and [step["status"] for step in cycle["policy_dev"]["steps"]]
        == ["accepted", "not_better", "rejected"]
    ]
    assert done, "the demo has no finished cycle with a policy decision"
    passed = [n for n in done[0]["nodes"] if n["fail_class"] == "ok" and n["cell_id"] != "root"]
    assert len(passed) == 3
    assert any(n["fail_class"] == "implementation_failure" for n in done[0]["nodes"])
    running = [cycle for cycle in state["cycles"] if cycle["status"] == "running"]
    assert running, "the demo has no running cycle"
    statuses = sorted(n["status"] for n in running[0]["nodes"] if n["cell_id"] != "root")
    assert statuses == ["agent", "evaluating", "pending", "pending"]
    assert running[0]["policy_dev"]["status"] != "done"


def test_fixture_events_have_every_contract_field() -> None:
    doc = json.loads(_read(FIXTURE_EVENTS))
    state = json.loads(_read(FIXTURE))
    assert set(doc) >= {"live", "events", "artifacts"}
    cycle = next((c for c in state["cycles"] if c["status"] == "running"), state["cycles"][-1])
    running = {n["cell_id"] for n in cycle["nodes"] if n["status"] in {"agent", "evaluating"}}
    assert doc["live"], "the demo has no live source"
    for source in doc["live"]:
        kind, cycle_id, cell = source.split(":", 2)
        assert kind == "node" and cycle_id == cycle["cycle_id"] and cell in running, source
        assert source in doc["events"], f"{source} has no events"
    live_events = doc["events"][doc["live"][0]]
    assert len(live_events) >= 35
    assert any(event["is_error"] for event in live_events if event["kind"] == "tool_result")
    for source, events in doc["events"].items():
        assert re.fullmatch(r"(node|policy):[^:]+:.+", source), source
        for index, event in enumerate(events):
            _check_keys(event, EVENT_FIELDS, f"Event in {source}")
            assert event["i"] == index, f"event index not monotonic in {source}"
            assert event["kind"] in EVENT_KINDS
    for source, arts in doc["artifacts"].items():
        for name, text in arts.items():
            assert name in watch.ARTIFACTS, f"{source}: unknown artifact {name}"
            assert text is None or isinstance(text, str), f"{source}/{name} is not text"
    lengths = [
        len(event.get("text") or "")
        for events in doc["events"].values()
        for event in events
        if event["kind"] == "thinking"
    ]
    assert lengths and max(lengths) >= 10_000, "the demo has no 10k+ reasoning event"


def test_fixture_context_payloads_have_sections() -> None:
    doc = json.loads(_read(FIXTURE_EVENTS))
    contexts = doc.get("context") or {}
    assert contexts, "the demo has no context payload"
    for source, payload in contexts.items():
        _check_keys(
            payload, {"source", "kind", "log", "files", "command", "sections"}, f"context {source}"
        )
        assert payload["source"] == source
        assert payload["kind"] in {"node", "policy"}
        assert isinstance(payload["files"], list) and isinstance(payload["command"], dict)
        sections = payload["sections"]
        assert isinstance(sections, list) and sections, f"context {source} has no sections"
        ids = [section["id"] for section in sections]
        assert len(ids) == len(set(ids)), f"duplicate section ids in {source}"
        for section in sections:
            _check_keys(section, SECTION_FIELDS, f"section {source}/{section.get('id')}")
            assert isinstance(section["recorded"], bool), (
                f"{source}/{section['id']}: recorded is not a bool"
            )
            _check_stats(section, f"{source}/{section['id']}")
        by_id = {section["id"]: section for section in sections}
        system = by_id.get("system")
        assert system is not None and system["recorded"] is False, (
            f"{source}: system prompt is not unrecorded"
        )
        message = by_id.get("message")
        assert message is not None and message["recorded"] is True, (
            f"{source}: task prompt is not recorded"
        )


def test_pure_modules_are_dom_free() -> None:
    violations = []
    for name in PURE_MODULES:
        path = JS / name
        assert path.is_file(), f"missing pure module {name}"
        code = _strip_comments(_read(path), "js")
        violations.extend(f"{name}: {token}" for token in FORBIDDEN_IN_PURE if token in code)
        if name == "reducer.js":
            violations.extend(
                f"reducer.js: {token}" for token in FORBIDDEN_IN_REDUCER if token in code
            )
    assert not violations, (
        "pure modules must not touch the DOM, storage or the clock: " + ", ".join(violations)
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_module_parses_with_node(tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if key != "NODE_OPTIONS"}
    files = sorted(JS.rglob("*.js"))
    assert files
    for path in files:
        try:
            result = subprocess.run(
                ["node", "--input-type=module", "--check"],
                input=_read(path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                cwd=tmp_path,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"node did not parse {path.relative_to(STATIC)} within 30 s")
        assert result.returncode == 0, (
            f"{path.relative_to(STATIC)} does not parse:\n{result.stderr}"
        )
