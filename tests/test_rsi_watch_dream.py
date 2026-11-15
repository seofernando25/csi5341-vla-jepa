"""Tests for the Dream-RSI policy-development payload and the calibration fallback of `/api/state`."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from rsi import watch
from rsi.watch import build_state, make_server, parse_source, read_artifact
from rsi.watch_dream import dream_payload

SHA_A = "a" * 64
SHA_B = "b" * 64


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watch, "_gpu_block", lambda: None)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json(path: Path, obj: Any) -> None:
    _write(path, json.dumps(obj, indent=2) + "\n")


def _trace(world: str, beta: float, steps: list[dict[str, Any]], gain: float) -> str:
    row = {"world_id": world, "beta": beta, "max_gain": gain, "steps": steps}
    return json.dumps(row) + "\n"


def _selection() -> dict[str, Any]:
    return {
        "protocol": "dream-rsi-policy-dev",
        "cycle_id": "iter_000",
        "recorded_utc": "2026-11-15T10:00:00+00:00",
        "num_worlds": 2,
        "world_ids": ["w0", "w1"],
        "max_parallelism": 2,
        "lambda_parallel": 0.05,
        "selected_slot": "m1_llm",
        "selected_policy_path": "/home/fer/secret/selected_policy.py",
        "incumbent_in_sample_V": 0.4,
        "selected_in_sample_V": 0.5,
        "heldout": {"num_worlds": 24, "mean_V_by_slot": {"m0_incumbent": 0.4, "m1_llm": 0.5}},
        "heldout_delta_vs_incumbent": 0.02,
        "variants": [
            {
                "slot": "m0_incumbent",
                "step": 0,
                "status": "incumbent",
                "mean_v": 0.4,
                "delta_vs_best": 0.0,
                "error": None,
                "attempts": 0,
                "policy_path": "/home/fer/secret/step_00/candidate.py",
                "policy_sha256": SHA_A,
            },
            {
                "slot": "m1_llm",
                "step": 1,
                "status": "accepted",
                "mean_v": 0.5,
                "delta_vs_best": 0.1,
                "error": None,
                "attempts": 1,
                "policy_path": "/home/fer/secret/step_01/candidate.py",
                "policy_sha256": SHA_B,
            },
        ],
        "agent": {
            "backend": "pi",
            "model": "mistral/deepseek-v4-1-flash-1017",
            "effort": "max",
            "calls": 1,
            "num_turns": 10,
            "cost_usd": 0.5,
        },
    }


def _result(sha: str, betas: list[float]) -> dict[str, Any]:
    points = [
        {"beta": b, "probes": i + 2, "attainment": 0.2 * (i + 1), "rounds": 2}
        for i, b in enumerate(betas)
    ]
    return {
        "lambda_parallel": 0.05,
        "max_parallelism": 2,
        "mean_v": 0.5,
        "policy_sha256": sha,
        "worlds": {
            "w0": {
                "degenerate": False,
                "hypervolume": 0.2,
                "penalty": 0.1,
                "v": 0.2,
                "points": points,
            },
            "w1": {
                "degenerate": True,
                "hypervolume": 0.0,
                "penalty": 0.0,
                "v": 0.0,
                "points": [{"beta": 0.0, "probes": 1, "attainment": 0.0, "rounds": 1}],
            },
        },
    }


@pytest.fixture
def study(tmp_path: Path) -> Path:
    """A study with one finished policy-development cycle over two synthetic worlds."""
    root = tmp_path / "study"
    pdir = root / "policies" / "iter_000"
    _json(pdir / "selection.json", _selection())
    for step, status, sha, betas in (
        (0, "incumbent", SHA_A, [0.0]),
        (1, "accepted", SHA_B, [0.0, 0.5]),
    ):
        name = f"step_{step:02d}"
        _json(
            pdir / "evidence" / "history" / name / "verdict.json",
            {
                "step": step,
                "slot": "m0_incumbent" if step == 0 else "m1_llm",
                "status": status,
                "mean_v": 0.4 if step == 0 else 0.5,
                "delta_vs_best": 0.0 if step == 0 else 0.1,
                "error": None,
                "attempts": step,
            },
        )
        _json(pdir / name / "replay" / "result.json", _result(sha, betas))
    _write(
        pdir / "step_01" / "replay" / "traces.jsonl",
        _trace(
            "w0",
            0.0,
            [
                {
                    "round": 1,
                    "prefix_size": 0,
                    "batch": ["b0_a0"],
                    "outcomes": [{"cell_id": "b0_a0", "score": 0.2, "fail_class": "ok"}],
                }
            ],
            0.3,
        )
        + _trace(
            "w0",
            0.5,
            [
                {
                    "round": 1,
                    "prefix_size": 0,
                    "batch": ["b0_a0", "b1_a0"],
                    "outcomes": [
                        {"cell_id": "b0_a0", "score": 0.2, "fail_class": "ok"},
                        {"cell_id": "b1_a0", "score": 0.1, "fail_class": "timeout"},
                    ],
                }
            ],
            0.3,
        )
        + _trace(
            "w1",
            0.0,
            [
                {
                    "round": 1,
                    "prefix_size": 0,
                    "batch": ["b0_a0"],
                    "outcomes": [{"cell_id": "b0_a0", "score": 0.0, "fail_class": "ok"}],
                }
            ],
            0.1,
        ),
    )
    _write(
        pdir / "step_00" / "replay" / "traces.jsonl",
        _trace("w0", 0.0, [{"round": 1, "prefix_size": 0, "batch": ["b0_a0"], "outcomes": []}], 0.3)
        + _trace(
            "w1", 0.0, [{"round": 1, "prefix_size": 0, "batch": ["b0_a0"], "outcomes": []}], 0.1
        ),
    )
    _json(
        pdir / "evidence" / "worlds" / "00_w0.json",
        {
            "world_id": "w0",
            "baseline_score": 0.0,
            "nodes": [
                {
                    "cell_id": "b0_a0",
                    "parent_id": "root",
                    "branch": 0,
                    "attempt": 0,
                    "seq": 0,
                    "score": 0.2,
                    "evaluated": True,
                    "fail_class": "ok",
                    "summary": "/home/fer/secret/leak.py",
                },
                {
                    "cell_id": "b1_a0",
                    "parent_id": "b0_a0",
                    "branch": 1,
                    "attempt": 0,
                    "seq": 1,
                    "score": None,
                    "evaluated": False,
                    "fail_class": "timeout",
                    "summary": "b1",
                },
            ],
        },
    )
    _json(
        pdir / "evidence" / "worlds" / "01_w1.json",
        {
            "world_id": "w1",
            "baseline_score": 0.0,
            "nodes": [
                {
                    "cell_id": "b0_a0",
                    "parent_id": "root",
                    "branch": 0,
                    "attempt": 0,
                    "seq": 0,
                    "score": 0.0,
                    "evaluated": True,
                    "fail_class": "ok",
                    "summary": "b0",
                },
            ],
        },
    )
    return root


def test_pending_before_the_cycle_ends(tmp_path: Path) -> None:
    study = tmp_path / "study"
    (study / "cycles" / "iter_000").mkdir(parents=True)
    payload = dream_payload(study, "iter_000", None)
    assert payload["status"] == "pending"
    assert payload["source_dir"] is None
    assert payload["worlds"] is None and payload["candidates"] is None
    assert payload["rounds"] is None and payload["verdict"] is None


def test_missing_cycle_has_no_source(tmp_path: Path) -> None:
    study = tmp_path / "study"
    study.mkdir()
    payload = dream_payload(study, "iter_000", None)
    assert payload["status"] == "missing" and payload["source_dir"] is None


def test_ready_payload_reads_the_written_tree(study: Path) -> None:
    payload = dream_payload(study, "iter_000", None)
    assert payload["status"] == "ready"
    assert payload["source_dir"] == "policies/iter_000"
    assert payload["recorded_utc"] == "2026-11-15T10:00:00+00:00"
    assert payload["protocol"] == {
        "m": 2,
        "lambda_parallel": 0.05,
        "beta_grid": [0.0, 0.5],
        "max_parallelism": 2,
    }
    assert [c["slot"] for c in payload["candidates"]] == ["m0_incumbent", "m1_llm"]
    assert payload["candidates"][1]["sha256"] == SHA_B
    assert payload["candidates"][1]["status"] == "accepted"
    assert payload["worlds"] == [
        {"id": "w0", "nodes": 2, "baseline_score": 0.0, "max_gain": 0.3, "degenerate": False},
        {"id": "w1", "nodes": 1, "baseline_score": 0.0, "max_gain": 0.1, "degenerate": True},
    ]
    assert payload["tree"]["world_id"] == "w0"
    assert payload["tree"]["nodes"][0]["cell_id"] == "b0_a0"
    assert payload["tree"]["nodes"][1]["evaluated"] is False
    assert payload["curves"]["m1_llm"]["w0"][1] == {
        "beta": 0.5,
        "probes": 3,
        "attainment": 0.4,
        "rounds": 2,
    }
    assert payload["heldout"]["num_worlds"] == 24
    assert payload["verdict"] == {
        "selected_slot": "m1_llm",
        "incumbent_in_sample_V": 0.4,
        "selected_in_sample_V": 0.5,
    }
    assert payload["agent"]["cost_usd"] == 0.5
    assert payload["rounds"] is None, "rounds are per world"


def test_world_filter_selects_one_tree_and_its_rounds(study: Path) -> None:
    payload = dream_payload(study, "iter_000", "w1")
    assert payload["tree"]["world_id"] == "w1"
    assert list(payload["rounds"]) == ["m0_incumbent", "m1_llm"]
    assert list(payload["rounds"]["m1_llm"]) == ["w1"]
    assert list(payload["rounds"]["m1_llm"]["w1"]) == ["0"]
    assert payload["rounds"]["m1_llm"]["w1"]["0"][0]["batch"] == ["b0_a0"]
    both = dream_payload(study, "iter_000", "w0")
    assert list(both["rounds"]["m1_llm"]["w0"]) == ["0", "0.5"]
    assert both["rounds"]["m1_llm"]["w0"]["0.5"][0]["outcomes"][1]["fail_class"] == "timeout"
    assert "w1" not in both["rounds"]["m1_llm"]


def test_unknown_world_is_null_not_a_substitute(study: Path) -> None:
    payload = dream_payload(study, "iter_000", "nope")
    assert payload["status"] == "ready"
    assert payload["tree"] is None and payload["rounds"] is None


def test_absent_files_are_null(tmp_path: Path) -> None:
    root = tmp_path / "bare"
    _json(root / "policies" / "iter_000" / "selection.json", {"selected_slot": "m0_incumbent"})
    payload = dream_payload(root, "iter_000", None)
    assert payload["status"] == "ready"
    assert payload["worlds"] is None and payload["tree"] is None
    assert payload["candidates"] is None and payload["curves"] is None
    assert payload["heldout"] is None and payload["verdict"]["selected_slot"] == "m0_incumbent"
    assert payload["agent"] is None
    assert payload["protocol"] == {
        "m": None,
        "lambda_parallel": None,
        "beta_grid": None,
        "max_parallelism": None,
    }


def test_no_machine_path_leaves_the_payload(study: Path) -> None:
    payload = dream_payload(study, "iter_000", "w0")
    text = json.dumps(payload)
    assert "/home/" not in text and "/tmp/" not in text
    assert payload["tree"]["nodes"][0]["summary"] == "<path>"
    assert payload["source_dir"] == "policies/iter_000"


def test_calibration_root_is_one_finished_cycle(tmp_path: Path) -> None:
    """No loop_state.json, but a selection: the study is the finished policy-dev root."""
    root = tmp_path / "cleanroom-v2"
    selection = _selection() | {"cycle_id": "cleanroom-v2"}
    _json(root / "selection.json", selection)
    (root / "evidence" / "history" / "step_00").mkdir(parents=True)
    state = build_state(root)
    assert state["study_dir"] == "cleanroom-v2"
    assert state["loop"]["status"] == "done" and state["loop"]["running"] is False
    assert state["loop"]["agent_cost_usd"] == 0.5
    assert state["loop"]["baseline"] == {"score": None, "resident_bytes": None}
    assert [c["cycle_id"] for c in state["cycles"]] == ["cleanroom-v2"]
    cycle = state["cycles"][0]
    assert cycle["status"] == "done" and cycle["nodes"] == []
    assert cycle["policy_sha256"] == SHA_B
    assert cycle["policy_dev"]["selected_step"] == 1
    assert cycle["policy_dev"]["incumbent_v"] == 0.4
    assert cycle["policy_dev"]["heldout_delta"] == 0.02
    assert [(s["step"], s["status"]) for s in cycle["policy_dev"]["steps"]] == [
        (0, "incumbent"),
        (1, "accepted"),
    ]


def test_calibration_root_answers_only_its_own_cycle(tmp_path: Path) -> None:
    root = tmp_path / "cleanroom-v2"
    _json(root / "selection.json", _selection() | {"cycle_id": "cleanroom-v2"})
    assert dream_payload(root, "cleanroom-v2", None)["status"] == "ready"
    assert dream_payload(root, "iter_999", None)["status"] == "missing"


def test_state_reports_baseline_bytes_and_eval_detail(tmp_path: Path) -> None:
    study = tmp_path / "live"
    ndir = study / "cycles" / "iter_000" / "nodes" / "iter_000-b0-a0"
    _json(
        ndir / "score.json",
        {
            "fail_class": "ok",
            "score": 0.75,
            "sizes": {"resident_bytes": 1_000_000, "baseline_resident_bytes": 4_000_000},
            "per_task": {"0": {"successes": 1, "episodes": 1}},
        },
    )
    _json(
        ndir / "eval" / "result.json",
        {
            "baseline_resident_bytes": 4_000_000,
            "sizes": {"peak_cuda_bytes": 9_000_000},
            "stage_seconds": {"load": 1.5},
        },
    )
    _json(
        study / "loop_state.json",
        {"status": "done", "iterations": [{"iteration": 0, "cycle_id": "iter_000"}]},
    )
    state = build_state(study)
    assert state["loop"]["baseline"] == {"score": None, "resident_bytes": 4_000_000}
    cycle = state["cycles"][0]
    assert cycle["probes_used_live"] == 1
    node = cycle["nodes"][1]
    assert node["baseline_mb"] == 4.0
    assert node["peak_cuda_bytes"] == 9_000_000
    assert node["stage_seconds"] == {"load": 1.5}
    assert node["per_task"] == {"0": {"successes": 1, "episodes": 1}}


def test_export_and_prompt_artifacts(tmp_path: Path) -> None:
    study = tmp_path / "live"
    ndir = study / "cycles" / "iter_000" / "nodes" / "iter_000-b1-a0"
    _write(ndir / "export" / "solution.py", "x = 1\n")
    _write(ndir / "evidence" / "TASK.md", "# Task\n")
    source = parse_source("node:iter_000:iter_000-b1-a0")
    assert read_artifact(study, source, "export") == "x = 1\n"
    assert read_artifact(study, source, "prompt") == "# Task\n"
    assert "export" in watch.ARTIFACTS and "prompt" in watch.ARTIFACTS


def test_calibration_layout_artifacts(tmp_path: Path) -> None:
    """A calibration root keeps its steps at `<root>/step_NN`, not under `policies/<cycle>`."""
    root = tmp_path / "cleanroom-v2"
    _json(root / "selection.json", _selection() | {"cycle_id": "cleanroom-v2"})
    sdir = root / "step_03"
    _write(sdir / "candidate.py", "candidate\n")
    _write(sdir / "workspace" / "method.py", "seed\n")
    _json(sdir / "replay" / "result.json", {"worlds": {}})
    _json(root / "evidence" / "history" / "step_03" / "verdict.json", {"step": 3, "slot": "m3_llm"})
    source = parse_source("policy:cleanroom-v2:3")
    verdict = read_artifact(root, source, "verdict")
    curve = read_artifact(root, source, "curve")
    assert read_artifact(root, source, "policy") == "candidate\n"
    assert read_artifact(root, source, "seed") == "seed\n"
    assert verdict is not None and json.loads(verdict)["slot"] == "m3_llm"
    assert curve is not None and json.loads(curve) == {"worlds": {}}
    assert read_artifact(root, parse_source("policy:cleanroom-v2:1"), "policy") is None


def _get(srv: Any, path: str) -> tuple[int, bytes]:
    port = srv.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_dream_route_serves_ready_and_404s_an_unknown_cycle(study: Path) -> None:
    srv = make_server(study, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _get(srv, "/api/dream?cycle=iter_000")
        assert status == 200 and json.loads(body)["status"] == "ready"
        status, body = _get(srv, "/api/dream?cycle=iter_000&world=w1")
        assert status == 200 and json.loads(body)["tree"]["world_id"] == "w1"
        status, body = _get(srv, "/api/dream?cycle=iter_999")
        assert status == 404 and json.loads(body) == {"error": "unknown cycle"}
        status, body = _get(srv, "/api/dream")
        assert status == 400 and "cycle" in json.loads(body)["error"]
        status, _ = _get(srv, "/api/dream?cycle=..%2Fetc")
        assert status == 400
    finally:
        srv.shutdown()
        srv.server_close()
