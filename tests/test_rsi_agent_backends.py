"""Tests for the Claude Code and pi agent backends (offline unless RSI_LIVE_AGENT=1)."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from rsi import agent_backends
from rsi.agent_backends import (
    AgentSpec,
    agent_command,
    claude_command,
    load_agent_spec,
    parse_agent_output,
    pi_command,
)
from rsi.sandbox import run_sandboxed_agent

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
STUDY = REPO / "studies" / "program-v1"
CLAUDE = AgentSpec("claude", "haiku", "low", "read,write,edit")
requires_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")


@pytest.fixture
def fake_claude_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake HOME with a credentials file and a fake claude binary on PATH."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / ".credentials.json").write_text("{}", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "claude"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(bin_dir))
    return home


def _pairs(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def test_claude_argv_has_required_flags(fake_claude_home: Path) -> None:
    argv, binds, env = claude_command(CLAUDE)
    for flag in (
        "-p",
        "--model",
        "--effort",
        "--tools",
        "--permission-mode",
        "--add-dir",
        "--setting-sources",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--output-format",
    ):
        assert flag in argv, flag
    assert _pairs(argv, "--model") == ["haiku"]
    assert _pairs(argv, "--effort") == ["low"]
    assert _pairs(argv, "--tools") == ["Read,Write,Edit"]
    assert _pairs(argv, "--add-dir") == ["/tmp/evidence"]
    assert _pairs(argv, "--output-format") == ["stream-json"]
    assert "--verbose" in argv
    assert "Bash" not in " ".join(argv)
    assert env == {}
    assert str(Path(argv[0]).resolve()) == argv[0]
    assert binds[Path(argv[0])] == argv[0]
    creds = fake_claude_home / ".claude" / ".credentials.json"
    assert binds[creds] == "/tmp/.claude/.credentials.json"


def test_claude_effort_flag_omitted_when_none(fake_claude_home: Path) -> None:
    argv, _, _ = claude_command(AgentSpec("claude", "haiku", None, "read,write,edit"))
    assert "--effort" not in argv


@pytest.mark.parametrize("tools", ["read,write,edit,shell", "rm", "read,shell", ""])
def test_claude_rejects_tools_outside_read_write_edit(fake_claude_home: Path, tools: str) -> None:
    with pytest.raises(ValueError):
        claude_command(AgentSpec("claude", "haiku", "low", tools))


def test_claude_bash_tool_maps_and_is_explicitly_allowed(fake_claude_home: Path) -> None:
    argv, _, _ = claude_command(AgentSpec("claude", "haiku", "low", "read,write,edit,bash"))
    assert _pairs(argv, "--tools") == ["Read,Write,Edit,Bash"]
    assert _pairs(argv, "--allowedTools") == ["Bash(*)"]
    assert _pairs(argv, "--permission-mode") == ["acceptEdits"]


def test_claude_without_bash_has_no_allow_rule(fake_claude_home: Path) -> None:
    argv, _, _ = claude_command(CLAUDE)
    assert "--allowedTools" not in argv


def test_claude_missing_credentials_fails_closed(
    fake_claude_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    with pytest.raises(RuntimeError):
        claude_command(CLAUDE)


def test_claude_missing_binary_fails_closed(
    fake_claude_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "/nonexistent-bin")
    with pytest.raises(RuntimeError):
        claude_command(CLAUDE)


def test_pi_mounts_only_the_selected_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sandbox gets one provider's credential, never the whole host auth.json."""
    pi_dir = tmp_path / "pi-agent"
    pi_dir.mkdir()
    auth = {
        "mistral": {"type": "api_key", "key": "mistral-key-value"},
        "anthropic": {
            "type": "oauth",
            "access": "anthropic-access",
            "refresh": "anthropic-refresh",
        },
        "opencode-go": {"type": "api_key", "key": "opencode-key-value"},
    }
    (pi_dir / "auth.json").write_text(json.dumps(auth), encoding="utf-8")
    models = {
        "$schema": "https://example/schema.json",
        "providers": {"mistral": {"models": [{"id": "m"}]}, "lmstudio": {"models": []}},
    }
    (pi_dir / "models.json").write_text(json.dumps(models), encoding="utf-8")
    (pi_dir / "settings.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    monkeypatch.setattr(agent_backends, "resolve_pi_executable", lambda: "/usr/bin/pi")
    monkeypatch.setattr(agent_backends, "_which_resolved", lambda name: Path("/usr/bin/node"))
    argv, binds, env = pi_command(AgentSpec("pi", "mistral/m", None, "read,write,edit"))
    assert argv[argv.index("--tools") + 1] == "read,write,edit"
    assert "--no-tools" not in argv
    assert _pairs(argv, "--mode") == ["json"]
    staged = {dst: src for src, dst in binds.items() if dst.startswith("/tmp/pi-agent")}
    assert set(staged) == {"/tmp/pi-agent/auth.json", "/tmp/pi-agent/models.json"}
    assert "settings.json" not in " ".join(binds.values())
    assert env["PI_CODING_AGENT_DIR"] == "/tmp/pi-agent"
    for src in staged.values():
        assert src.stat().st_mode & 0o777 == 0o600
    auth_text = staged["/tmp/pi-agent/auth.json"].read_text(encoding="utf-8")
    assert json.loads(auth_text) == {"mistral": auth["mistral"]}
    assert "anthropic" not in auth_text and "opencode" not in auth_text
    models_text = staged["/tmp/pi-agent/models.json"].read_text(encoding="utf-8")
    assert list(json.loads(models_text)["providers"]) == ["mistral"]


def test_pi_missing_provider_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pi_dir = tmp_path / "pi-agent"
    pi_dir.mkdir()
    (pi_dir / "auth.json").write_text(
        json.dumps({"anthropic": {"type": "oauth"}}), encoding="utf-8"
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    monkeypatch.setattr(agent_backends, "resolve_pi_executable", lambda: "/usr/bin/pi")
    monkeypatch.setattr(agent_backends, "_which_resolved", lambda name: Path("/usr/bin/node"))
    with pytest.raises(RuntimeError, match="no credential for provider 'mistral'"):
        pi_command(AgentSpec("pi", "mistral/m", None, "read,write,edit"))


def test_pi_missing_auth_file_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pi_dir = tmp_path / "pi-agent"
    pi_dir.mkdir()
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    monkeypatch.setattr(agent_backends, "resolve_pi_executable", lambda: "/usr/bin/pi")
    monkeypatch.setattr(agent_backends, "_which_resolved", lambda name: Path("/usr/bin/node"))
    with pytest.raises(RuntimeError, match="auth file missing"):
        pi_command(AgentSpec("pi", "mistral/m", None, "read,write,edit"))


def test_pi_auth_preflight_blocks_a_not_ready_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pi_dir = tmp_path / "pi-agent"
    pi_dir.mkdir()
    (pi_dir / "auth.json").write_text(json.dumps({"mistral": {"key": "k"}}), encoding="utf-8")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    monkeypatch.setattr(agent_backends, "resolve_pi_executable", lambda: "/usr/bin/pi")
    monkeypatch.setattr(agent_backends, "_which_resolved", lambda name: Path("/usr/bin/node"))
    monkeypatch.setattr(agent_backends, "pi_auth_status", lambda provider, model: "invalid")
    with pytest.raises(RuntimeError, match="pi auth check failed"):
        agent_command(AgentSpec("pi", "mistral/m", None, "read,write,edit"))


def test_pi_auth_preflight_ready_allows_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pi_dir = tmp_path / "pi-agent"
    pi_dir.mkdir()
    (pi_dir / "auth.json").write_text(json.dumps({"mistral": {"key": "k"}}), encoding="utf-8")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    monkeypatch.setattr(agent_backends, "resolve_pi_executable", lambda: "/usr/bin/pi")
    monkeypatch.setattr(agent_backends, "_which_resolved", lambda name: Path("/usr/bin/node"))
    monkeypatch.setattr(agent_backends, "pi_auth_status", lambda provider, model: "ready")
    argv, _, _ = agent_command(AgentSpec("pi", "mistral/m", None, "read,write,edit"))
    assert _pairs(argv, "--mode") == ["json"]


def test_pi_auth_preflight_unqueryable_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_backends, "resolve_pi_executable", lambda: "/nonexistent/pi")
    assert agent_backends.pi_auth_status("mistral", "mistral/m") is None


def _pi_agent_dir(tmp_path: Path, models: dict | None) -> Path:
    pi_dir = tmp_path / "pi-agent"
    pi_dir.mkdir()
    (pi_dir / "auth.json").write_text(
        json.dumps({"mistral": {"type": "api_key", "key": "k"}}), encoding="utf-8"
    )
    if models is not None:
        (pi_dir / "models.json").write_text(json.dumps(models), encoding="utf-8")
    return pi_dir


def _pi_spec() -> AgentSpec:
    return AgentSpec("pi", "mistral/deepseek-v4-1-flash-1017", "max", "read,write,edit")


def test_pi_rejects_a_mistral_block_on_openai_completions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4: that adapter stringifies Mistral thinking arrays into `[object Object]`."""
    pi_dir = _pi_agent_dir(
        tmp_path,
        {
            "providers": {
                "mistral": {
                    "baseUrl": "https://api.mistral.ai/v1",
                    "api": "openai-completions",
                    "models": [{"id": "deepseek-v4-1-flash-1017", "reasoning": True}],
                }
            }
        },
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    monkeypatch.setattr(agent_backends, "resolve_pi_executable", lambda: "/usr/bin/pi")
    monkeypatch.setattr(agent_backends, "_which_resolved", lambda name: Path("/usr/bin/node"))
    monkeypatch.setattr(agent_backends, "pi_auth_status", lambda provider, model: "ready")
    with pytest.raises(RuntimeError) as exc:
        agent_command(_pi_spec())
    message = str(exc.value)
    assert "openai-completions" in message
    assert "mistral-conversations" in message
    assert "builtin entry" in message


def test_pi_rejects_a_mistral_block_with_a_v1_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The native adapter appends `/v1` itself, so a `/v1` base URL 404s at run time."""
    pi_dir = _pi_agent_dir(
        tmp_path,
        {
            "providers": {
                "mistral": {
                    "baseUrl": "https://api.mistral.ai/v1",
                    "api": "mistral-conversations",
                    "models": [{"id": "deepseek-v4-1-flash-1017"}],
                }
            }
        },
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    monkeypatch.setattr(agent_backends, "resolve_pi_executable", lambda: "/usr/bin/pi")
    monkeypatch.setattr(agent_backends, "_which_resolved", lambda name: Path("/usr/bin/node"))
    monkeypatch.setattr(agent_backends, "pi_auth_status", lambda provider, model: "ready")
    with pytest.raises(RuntimeError) as exc:
        agent_command(_pi_spec())
    message = str(exc.value)
    assert "/v1" in message
    assert "mistral-conversations" in message
    assert "builtin entry" in message


def test_pi_accepts_a_mistral_block_on_the_native_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pi_dir = _pi_agent_dir(
        tmp_path,
        {
            "providers": {
                "mistral": {
                    "baseUrl": "https://api.mistral.ai",
                    "api": "mistral-conversations",
                    "models": [{"id": "deepseek-v4-1-flash-1017"}],
                }
            }
        },
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    monkeypatch.setattr(agent_backends, "resolve_pi_executable", lambda: "/usr/bin/pi")
    monkeypatch.setattr(agent_backends, "_which_resolved", lambda name: Path("/usr/bin/node"))
    monkeypatch.setattr(agent_backends, "pi_auth_status", lambda provider, model: "ready")
    argv, _, _ = agent_command(_pi_spec())
    assert _pairs(argv, "--mode") == ["json"]


def test_pi_accepts_the_builtin_mistral_entry_without_a_local_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No local `mistral` block means pi's builtin entry, which is correct; do not fail."""
    pi_dir = _pi_agent_dir(
        tmp_path, {"providers": {"lmstudio": {"api": "openai-completions", "models": []}}}
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    monkeypatch.setattr(agent_backends, "resolve_pi_executable", lambda: "/usr/bin/pi")
    monkeypatch.setattr(agent_backends, "_which_resolved", lambda name: Path("/usr/bin/node"))
    monkeypatch.setattr(agent_backends, "pi_auth_status", lambda provider, model: "ready")
    argv, binds, _ = agent_command(_pi_spec())
    assert _pairs(argv, "--mode") == ["json"]
    staged = {dst for dst in binds.values() if dst.startswith("/tmp/pi-agent")}
    assert staged == {"/tmp/pi-agent/auth.json"}


def test_pi_accepts_a_missing_models_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pi_dir = _pi_agent_dir(tmp_path, None)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    monkeypatch.setattr(agent_backends, "resolve_pi_executable", lambda: "/usr/bin/pi")
    monkeypatch.setattr(agent_backends, "_which_resolved", lambda name: Path("/usr/bin/node"))
    monkeypatch.setattr(agent_backends, "pi_auth_status", lambda provider, model: "ready")
    argv, _, _ = agent_command(_pi_spec())
    assert _pairs(argv, "--mode") == ["json"]


def test_pi_model_api_prefers_a_per_model_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pi_dir = _pi_agent_dir(
        tmp_path,
        {
            "providers": {
                "mistral": {
                    "api": "mistral-conversations",
                    "models": [{"id": "deepseek-v4-1-flash-1017", "api": "openai-completions"}],
                }
            }
        },
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    assert agent_backends._pi_model_api("mistral", "mistral/deepseek-v4-1-flash-1017") == (
        "openai-completions"
    )
    with pytest.raises(RuntimeError, match="mistral-conversations"):
        agent_backends._require_pi_model(_pi_spec())


def test_parse_pi_stream_fixture() -> None:
    """Finding B: cost and turns come from the JSON stream, not from raw stdout."""
    stream = (FIXTURES / "pi_stream.jsonl").read_text(encoding="utf-8")
    parsed = parse_agent_output("pi", stream)
    assert parsed["text"] == "DONE"
    assert parsed["num_turns"] == 3
    assert parsed["cost_usd"] == pytest.approx(0.000744)
    assert parsed["is_error"] is False


def test_parse_pi_stream_without_terminal_event_is_not_error() -> None:
    stream = json.dumps(
        {
            "type": "turn_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
        }
    )
    parsed = parse_agent_output("pi", stream + "\n")
    assert parsed == {"text": "x", "num_turns": 1, "cost_usd": None, "is_error": None}


def test_parse_pi_plain_text_stays_plain() -> None:
    parsed = parse_agent_output("pi", "no json here\n")
    assert parsed == {
        "text": "no json here\n",
        "num_turns": None,
        "cost_usd": None,
        "is_error": None,
    }


def test_parse_claude_json() -> None:
    stdout = json.dumps(
        {"result": "done", "num_turns": 3, "total_cost_usd": 0.0123, "is_error": False}
    )
    parsed = parse_agent_output("claude", stdout)
    assert parsed == {"text": "done", "num_turns": 3, "cost_usd": 0.0123, "is_error": False}


def test_parse_claude_garbage_and_empty() -> None:
    garbage = parse_agent_output("claude", "not json at all")
    assert garbage == {
        "text": "not json at all",
        "num_turns": None,
        "cost_usd": None,
        "is_error": None,
    }
    empty = parse_agent_output("claude", "")
    assert empty == {"text": "", "num_turns": None, "cost_usd": None, "is_error": None}


def test_parse_claude_stream_fixture() -> None:
    stream = (FIXTURES / "claude_stream.jsonl").read_text(encoding="utf-8")
    parsed = parse_agent_output("claude", stream)
    assert parsed["text"].startswith("DONE")
    assert parsed["num_turns"] == 5
    assert parsed["is_error"] is False
    assert isinstance(parsed["cost_usd"], float) and parsed["cost_usd"] > 0


def test_parse_stream_skips_garbage_and_partial_lines() -> None:
    result = json.dumps(
        {"type": "result", "num_turns": 2, "total_cost_usd": 0.5, "is_error": True, "result": "x"}
    )
    stdout = 'not json\n{"type": "assistant"\n' + result + '\n{"type": "assist'
    parsed = parse_agent_output("claude", stdout)
    assert parsed == {"text": "x", "num_turns": 2, "cost_usd": 0.5, "is_error": True}


def test_parse_stream_without_result_line_is_plain() -> None:
    stdout = json.dumps({"type": "system", "subtype": "init"}) + "\n"
    parsed = parse_agent_output("claude", stdout)
    assert parsed == {"text": stdout, "num_turns": None, "cost_usd": None, "is_error": None}


def test_parse_pi_is_text_only() -> None:
    parsed = parse_agent_output("pi", '{"result": "x", "total_cost_usd": 1.0}')
    assert parsed["text"] == '{"result": "x", "total_cost_usd": 1.0}'
    assert parsed["cost_usd"] is None and parsed["num_turns"] is None


def test_load_agent_spec_reads_protocol() -> None:
    spec = load_agent_spec(STUDY)
    assert spec == AgentSpec("pi", "mistral/deepseek-v4-1-flash-1017", "max", "read,write,edit")


def test_load_agent_spec_backend_override_drops_protocol_model(tmp_path: Path) -> None:
    proto = json.loads((STUDY / "protocol.json").read_text(encoding="utf-8"))
    (tmp_path / "protocol.json").write_text(json.dumps(proto), encoding="utf-8")
    spec = load_agent_spec(tmp_path, backend="claude")
    assert spec.backend == "claude"
    assert spec.model == "haiku"
    assert spec.effort == "low"
    assert load_agent_spec(tmp_path, backend="pi", effort="high").effort == "high"


def test_custom_agent_records_backend_custom(tmp_path: Path) -> None:
    res = run_sandboxed_agent(
        prompt_text="p",
        workspace_dir=tmp_path / "ws",
        evidence_dir=tmp_path / "ev",
        logs_dir=tmp_path / "logs",
        agent_cmd=["/bin/sh", "-c", "true"],
        agent=CLAUDE,
        use_bwrap=False,
        timeout=30,
    )
    assert res.backend == "custom"
    assert res.cost_usd is None and res.num_turns is None
    meta = json.loads((tmp_path / "logs" / "command.json").read_text(encoding="utf-8"))
    assert meta["backend"] == "custom"
    assert meta["cli_version"] is None


@requires_bwrap
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RSI_LIVE_AGENT") != "1", reason="set RSI_LIVE_AGENT=1")
def test_live_claude_agent_writes_file(tmp_path: Path) -> None:
    res = run_sandboxed_agent(
        prompt_text="Write the single word PONG into /tmp/work/out.txt",
        workspace_dir=tmp_path / "ws",
        evidence_dir=tmp_path / "ev",
        logs_dir=tmp_path / "logs",
        agent=CLAUDE,
        timeout=240,
    )
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "ws" / "out.txt").read_text(encoding="utf-8").strip() == "PONG"
    assert isinstance(res.cost_usd, float)


def test_agent_stdout_is_visible_before_exit(tmp_path: Path) -> None:
    seen: list[bool] = []
    holder: dict[str, object] = {}

    def run() -> None:
        holder["res"] = run_sandboxed_agent(
            prompt_text="p",
            workspace_dir=tmp_path / "ws",
            evidence_dir=tmp_path / "ev",
            logs_dir=tmp_path / "logs",
            agent_cmd=["/bin/sh", "-c", "echo first; sleep 3; echo second"],
            use_bwrap=False,
            timeout=30,
        )

    worker = threading.Thread(target=run)
    worker.start()
    log = tmp_path / "logs" / "stdout.log"
    deadline = time.monotonic() + 20
    while worker.is_alive() and time.monotonic() < deadline:
        if log.is_file() and "first" in log.read_text(encoding="utf-8"):
            seen.append(worker.is_alive())
            break
        time.sleep(0.05)
    worker.join(timeout=30)
    assert seen == [True], "first line was not on disk while the agent was still running"
    assert holder["res"].stdout.splitlines() == ["first", "second"]
