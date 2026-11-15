"""Tests for Dream-RSI action space, history digest, proposer transports, and round integration."""

from __future__ import annotations

import dataclasses
import hashlib
import http.server
import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import rsi.live_round as live_round_mod
import rsi.propose as propose_mod
from rsi.action_space import (
    ACTION_SPACE,
    build_recipe_changes,
    canonical_signature_from_candidate,
    validate_proposal,
)
from rsi.digest import collect_history, history_digest
from rsi.live import (
    build_round1_candidates,
    build_round2_candidates,
    build_round3_candidates,
    build_round4_candidates,
    build_round5_candidates,
    build_round6_candidates,
)
from rsi.propose import (
    TransportResponseError,
    candidates_from_journal,
    load_dotenv,
    propose,
)


def _valid_c1(cid: str = "r7-c1-test", *, solver_steps: int = 1) -> dict[str, Any]:
    return {
        "id": cid,
        "base": "C1",
        "knobs": {
            "text_mlp_prune": 0.10,
            "vision_mlp_prune": 0.30,
            "group_size": 32,
            "vision_bits": 2,
            "action_precision": "bf16_attn_hqq2_mlp",
            "solver_steps": solver_steps,
        },
        "hypothesis": "Tests BF16 action attention with 1-step solver on C1.",
        "expected_resident_mb": 1220.0,
    }


def _valid_t2(cid: str = "r7-t2-test") -> dict[str, Any]:
    return {
        "id": cid,
        "base": "T2",
        "knobs": {
            "text_mlp_prune": 0.15,
            "vision_mlp_prune": 0.35,
            "group_size": 32,
            "vision_bits": 2,
            "action_precision": "bf16_full",
            "solver_steps": 4,
        },
        "hypothesis": "Tests full BF16 action head on T2 with moderate MLP cuts.",
        "expected_resident_mb": 940.0,
    }


def _valid_c2(cid: str = "r7-c2-test") -> dict[str, Any]:
    return {
        "id": cid,
        "base": "C2",
        "knobs": {
            "text_mlp_prune": 0.10,
            "vision_mlp_prune": 0.25,
            "vision_bits": 4,
            "action_precision": "nf4",
            "solver_steps": 1,
        },
        "hypothesis": "Tests mild MLP cuts on C2 one-step NF4.",
        "expected_resident_mb": 1240.0,
    }


def test_r1_to_r6_candidates_unchanged_after_refactor() -> None:
    expected = {
        "r1": (9, "73b7352614aaf2c5bf17e1b11c3034b4c13840501acef559c837766c91bcdef3"),
        "r2": (6, "6ccb142471e2564384f96316b4be2ff1a60d0042ada53433ccbfe4abab7bd677"),
        "r3": (8, "3885d6dba3848ce9159aa055a7c3b9ac8805576d67bdaf23d28e1e1258eceaae"),
        "r4": (6, "83454a4efab711a16614c47e706d3becd5ce7bcc57dba278404c39ec9d6c8ac1"),
        "r5": (6, "25f28143da22f9cb703d40aaeb799dc77df9f8bc658e3c6b37a0dcceadbc8ab6"),
        "r6": (6, "b1ee6ba76f47b61e2d216cc0fc9593c946ef806da0377debb16b5f987bd6be11"),
    }
    builders = {
        "r1": build_round1_candidates,
        "r2": build_round2_candidates,
        "r3": build_round3_candidates,
        "r4": build_round4_candidates,
        "r5": build_round5_candidates,
        "r6": build_round6_candidates,
    }
    for rnd, fn in builders.items():
        cands = fn()
        count, digest = expected[rnd]
        assert len(cands) == count
        actual = hashlib.sha256(
            json.dumps([dataclasses.asdict(c) for c in cands], sort_keys=True).encode()
        ).hexdigest()
        assert actual == digest


def test_validator_accepts_valid_proposal_and_rejects_violations() -> None:
    c1 = _valid_c1()
    t2 = _valid_t2()
    c2 = _valid_c2()
    out = validate_proposal([c1, t2, c2], max_n=3)
    assert len(out) == 3
    assert out[2]["knobs"]["group_size"] is None
    assert "bases" in ACTION_SPACE and set(ACTION_SPACE["bases"]) == {"T2", "C1", "C2"}

    # Unknown candidate key
    bad = dict(c1, extra_key=1)
    with pytest.raises(ValueError, match="unknown keys"):
        validate_proposal([bad], max_n=2)

    # Unknown knob key
    bad_knob = _valid_c1()
    bad_knob["knobs"]["bogus"] = 1
    with pytest.raises(ValueError, match="unknown knob keys"):
        validate_proposal([bad_knob], max_n=2)

    # Out-of-range prune fraction
    oor = _valid_c1()
    oor["knobs"]["text_mlp_prune"] = 0.55
    with pytest.raises(ValueError, match="out of range"):
        validate_proposal([oor], max_n=2)

    # Disallowed base/knob combo (C2 with group_size=32)
    bad_c2 = dict(c2, knobs={**c2["knobs"], "group_size": 32})
    with pytest.raises(ValueError, match="group_size"):
        validate_proposal([bad_c2], max_n=2)

    # More than requested count
    with pytest.raises(ValueError, match="exceeding requested count"):
        validate_proposal([c1, t2], max_n=1)

    # Duplicate of already-probed id
    with pytest.raises(ValueError, match="duplicates already-probed cell"):
        validate_proposal([c1], max_n=2, probed_ids={c1["id"]})

    # Duplicate of already-probed config under a new id
    sig = canonical_signature_from_candidate(c1["base"], c1["knobs"])
    with pytest.raises(ValueError, match="duplicates already-probed cell 'r4-existing'"):
        validate_proposal([c1], max_n=2, probed_signatures={sig: "r4-existing"})

    # Within-proposal duplicate configuration rejected
    with pytest.raises(ValueError, match="duplicates earlier proposal candidate"):
        validate_proposal([_valid_t2("t2-a"), _valid_t2("t2-b")], max_n=2)

    # Less than half full-vocab rejected
    t2_b = _valid_t2("t2-b")
    t2_b["knobs"]["text_mlp_prune"] = 0.20
    with pytest.raises(ValueError, match="at least half"):
        validate_proposal([_valid_t2("t2-a"), t2_b], max_n=2)

    # Builder sanity on C2
    changes = build_recipe_changes(
        base="C2",
        text_mlp_prune=0.10,
        vision_mlp_prune=0.25,
        group_size=None,
        vision_bits=4,
        action_precision="nf4",
        solver_steps=1,
    )
    assert any(ch["method"] == "solver_steps" and ch["count"] == 1 for ch in changes)


def test_history_digest_deterministic_and_complete() -> None:
    d1 = history_digest()
    d2 = history_digest()
    assert d1 == d2

    digest, seen_ids, sigs = collect_history()
    assert digest["gates"] == {"smoke": ">= 8/10", "development": ">= 77/100"}
    assert set(digest["reference_points"]) >= {"B16", "C1", "C2"}
    assert len(digest["probed_cells"]) == len(seen_ids) >= 27
    assert len(sigs) >= 20

    by_id = {c["id"]: c for c in digest["probed_cells"]}
    assert by_id["r4-c1-t15-v35-g32"]["base"] == "C1"
    assert by_id["r4-c1-t15-v35-g32"]["dev"] == "86/100"
    assert by_id["r1-t20-v45-g32"]["base"] == "T2"
    assert by_id["r1-t20-v45-g32"]["dev"] == "83/100"
    assert by_id["r5-hqq3-t20-v40"]["dev"] == "85/100"
    assert any("smoke_failing_tasks" in c for c in digest["probed_cells"])


def test_propose_command_repair_retry_force_and_live_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in (
        "RSI_PROPOSER_CMD",
        "RSI_PROPOSER_BASE_URL",
        "RSI_PROPOSER_MODEL",
        "RSI_PROPOSER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    counter_file = tmp_path / "calls.txt"
    fake_script = tmp_path / "fake_proposer.py"
    valid_reply = json.dumps([_valid_c1("r7-c1-a"), _valid_t2("r7-t2-a")])
    fake_script.write_text(
        "import pathlib, sys\n"
        f"cf = pathlib.Path({str(counter_file)!r})\n"
        "n = int(cf.read_text()) + 1 if cf.exists() else 1\n"
        "cf.write_text(str(n))\n"
        "_prompt = sys.stdin.read()\n"
        "if n == 1:\n"
        "    print('not valid json')\n"
        "elif n == 2:\n"
        "    assert 'Validation error' in _prompt\n"
        f"    print({valid_reply!r})\n"
        "else:\n"
        f"    print({valid_reply!r})\n"
    )
    cmd = f"{sys.executable} {fake_script}"

    # Neither / both transport rejection
    with pytest.raises(ValueError, match="No proposer transport"):
        propose("r7", 2, proposals_dir=tmp_path / "props")
    with pytest.raises(ValueError, match="Specify exactly one transport"):
        propose(
            "r7",
            2,
            cmd,
            base_url="http://127.0.0.1:9999/v1",
            model="m",
            proposals_dir=tmp_path / "props",
        )
    with pytest.raises(ValueError, match="requires both"):
        propose("r7", 2, base_url="http://127.0.0.1:9999/v1", proposals_dir=tmp_path / "props")

    props_dir = tmp_path / "props"
    journal = propose("r7", 2, cmd, proposals_dir=props_dir)
    assert journal["transport"] == "command"
    assert [c["id"] for c in journal["candidates"]] == ["r7-c1-a", "r7-t2-a"]
    assert [v["valid"] for v in journal["validation_log"]] == [False, True]
    assert (props_dir / "r7.json").exists()

    # Overwrite requires force=True
    with pytest.raises(FileExistsError, match="--force"):
        propose("r7", 2, cmd, proposals_dir=props_dir)
    propose("r7", 2, cmd, force=True, proposals_dir=props_dir)

    # Journal -> Stage2Candidate + live_round resolver
    monkeypatch.setattr(
        live_round_mod,
        "candidates_from_journal",
        lambda rid: candidates_from_journal(rid, props_dir),
    )
    resolved = live_round_mod.resolve_round_candidates("r7")
    assert [c.cell_id for c in resolved] == ["r7-c1-a", "r7-t2-a"]
    assert resolved[0].recipe["changes"] == build_recipe_changes(
        base="C1",
        text_mlp_prune=0.10,
        vision_mlp_prune=0.30,
        group_size=32,
        vision_bits=2,
        action_precision="bf16_attn_hqq2_mlp",
        solver_steps=1,
    )
    assert len(live_round_mod.resolve_round_candidates("r4")) == 6


class _StubState:
    def __init__(self, responses: list[tuple[int, dict[str, Any]]]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []


def _start_stub_server(state: _StubState) -> tuple[http.server.ThreadingHTTPServer, str]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            hdrs = {k.lower(): v for k, v in self.headers.items()}
            state.requests.append((self.path, hdrs, json.loads(raw)))
            code, payload = state.responses.pop(0)
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/v1"


def test_openai_http_transport_retries_null_and_length_and_scrubs_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "fake-secret-token-xyz"
    monkeypatch.setenv("TEST_PROPOSER_KEY", secret)

    # Case A: Attempt 1 returns content: null -> repair retry succeeds, no max_tokens sent
    valid_reply = f"```json\n{json.dumps([_valid_c1('r7-http-c1'), _valid_c2('r7-http-c2')])}\n```"
    state = _StubState(
        [
            (
                200,
                {
                    "model": "test-stub-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": None},
                            "finish_reason": "stop",
                        }
                    ],
                },
            ),
            (
                200,
                {
                    "model": "test-stub-model",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": valid_reply},
                            "finish_reason": "stop",
                        }
                    ],
                },
            ),
        ]
    )
    server, url = _start_stub_server(state)
    try:
        journal = propose(
            "r7",
            2,
            base_url=url,
            model="req-model",
            api_key_env="TEST_PROPOSER_KEY",
            temperature=0.0,
            proposals_dir=tmp_path / "props_a",
        )
    finally:
        server.shutdown()
        server.server_close()

    assert len(state.requests) == 2
    for path, hdrs, req_body in state.requests:
        assert path == "/v1/chat/completions"
        assert hdrs["authorization"] == f"Bearer {secret}"
        assert "max_tokens" not in req_body
        assert req_body["model"] == "req-model"
        assert req_body["temperature"] == 0.0
    assert journal["transport"] == "openai_compatible"
    assert journal["response_model"] == "test-stub-model"
    assert journal["usage"] == {"prompt_tokens": 100, "completion_tokens": 50}
    assert [v["valid"] for v in journal["validation_log"]] == [False, True]
    assert secret not in (tmp_path / "props_a" / "r7.json").read_text()

    # Case B: Attempt 1 returns finish_reason='length' -> triggers single repair retry -> Attempt 2 succeeds
    state_b = _StubState(
        [
            (
                200,
                {
                    "model": "test-stub-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": '[{"id":'},
                            "finish_reason": "length",
                        }
                    ],
                },
            ),
            (
                200,
                {
                    "model": "test-stub-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": valid_reply},
                            "finish_reason": "stop",
                        }
                    ],
                },
            ),
        ]
    )
    server_b, url_b = _start_stub_server(state_b)
    try:
        journal_b = propose(
            "r7",
            2,
            base_url=url_b,
            model="req-model",
            api_key_env="EMPTY_KEY_LOCAL",
            proposals_dir=tmp_path / "props_b",
        )
    finally:
        server_b.shutdown()
        server_b.server_close()

    assert "authorization" not in state_b.requests[0][1]
    assert journal_b["validation_log"][0]["valid"] is False
    assert "length" in journal_b["validation_log"][0]["errors"][0]

    # Case C: Double content=null fails after 1 repair retry with TransportResponseError
    state_c = _StubState(
        [
            (200, {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}),
            (200, {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]}),
        ]
    )
    server_c, url_c = _start_stub_server(state_c)
    try:
        with pytest.raises(TransportResponseError, match="after 1 repair retry"):
            propose("r7", 2, base_url=url_c, model="req-model", proposals_dir=tmp_path / "props_c")
    finally:
        server_c.shutdown()
        server_c.server_close()

    # Case D: Upstream error echoing secret scrubs secret from error message
    state_d = _StubState([(401, {"error": f"bad token {secret}"})])
    server_d, url_d = _start_stub_server(state_d)
    try:
        with pytest.raises(RuntimeError) as exc_info:
            propose(
                "r7",
                2,
                base_url=url_d,
                model="req-model",
                api_key_env="TEST_PROPOSER_KEY",
                proposals_dir=tmp_path / "props_d",
            )
        assert secret not in str(exc_info.value)
        assert "***" in str(exc_info.value)
    finally:
        server_d.shutdown()
        server_d.server_close()


def test_dotenv_loading_and_cli_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for var in (
        "RSI_PROPOSER_CMD",
        "RSI_PROPOSER_BASE_URL",
        "RSI_PROPOSER_MODEL",
        "RSI_PROPOSER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    valid_reply = json.dumps([_valid_c1("r8-env-c1"), _valid_c2("r8-env-c2")])
    state = _StubState(
        [
            (
                200,
                {
                    "model": "env-stub",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": valid_reply},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        ]
    )
    server, url = _start_stub_server(state)
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "# temporary test dotenv\n"
        f"RSI_PROPOSER_BASE_URL={url}\n"
        'RSI_PROPOSER_MODEL="dotenv-model"\n'
        "export RSI_PROPOSER_API_KEY='dotenv-secret'\n"
    )
    monkeypatch.setattr(propose_mod, "DOTENV_PATH", dotenv_file)
    monkeypatch.setattr(propose_mod, "PROPOSALS_DIR", tmp_path / "props")
    # Real environment variable must win over .env
    monkeypatch.setenv("RSI_PROPOSER_MODEL", "real-env-model")

    try:
        assert propose_mod.main(["--round", "r8", "--n", "2", "--dry-run"]) == 0
        dry_out = capsys.readouterr().out
        assert "Dream-RSI round 'r8'" in dry_out

        rc = propose_mod.main(["--round", "r8", "--n", "2"])
        assert rc == 0
    finally:
        server.shutdown()
        server.server_close()

    assert len(state.requests) == 1
    _, hdrs, req_body = state.requests[0]
    assert req_body["model"] == "real-env-model"
    assert hdrs["authorization"] == "Bearer dotenv-secret"
    assert "max_tokens" not in req_body
    # Direct load_dotenv does not overwrite existing vars
    assert load_dotenv(dotenv_file) == {}


def test_relative_proposer_cmd_inherits_model_and_tolerant_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in (
        "RSI_PROPOSER_CMD",
        "RSI_PROPOSER_BASE_URL",
        "RSI_PROPOSER_MODEL",
        "RSI_PROPOSER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    fake_repo = tmp_path / "fake_repo"
    script_path = fake_repo / "scripts" / "mock_cmd.sh"
    script_path.parent.mkdir(parents=True)
    valid_reply = json.dumps([_valid_c1("r9-cmd-c1"), _valid_c2("r9-cmd-c2")])
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cat >/dev/null\n"
        'echo "[opencode] model=${RSI_PROPOSER_MODEL:?missing model}"\n'
        'echo "Proposing [1] candidates:"\n'
        f"echo {json.dumps(valid_reply)}\n"
        'echo "Done (trailing prose)."\n'
    )
    script_path.chmod(0o755)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "RSI_PROPOSER_CMD=scripts/mock_cmd.sh\nRSI_PROPOSER_MODEL=opencode-go/test-model\n"
    )
    load_dotenv(dotenv_file)
    monkeypatch.setattr(propose_mod, "REPO_ROOT", fake_repo)
    monkeypatch.chdir(elsewhere)

    journal = propose(
        "r9",
        2,
        proposals_dir=tmp_path / "props",
        dream_dir=tmp_path / "empty_dream",
        measurements_dir=tmp_path / "empty_meas",
        numbers_path=tmp_path / "missing_numbers.json",
    )
    assert journal["transport"] == "command"
    assert journal["proposer_cmd"] == "scripts/mock_cmd.sh"
    assert [c["id"] for c in journal["candidates"]] == ["r9-cmd-c1", "r9-cmd-c2"]
    assert "model=opencode-go/test-model" in journal["raw_reply"]
