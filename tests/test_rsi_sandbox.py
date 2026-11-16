"""Tests for the bubblewrap allowlist sandbox."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from rsi import gpu as rsi_gpu
from rsi.agents import parse_agent_output, run_sandboxed_agent, timed_out_result
from rsi.agents import result as agent_result
from rsi.paths import REPO_ROOT
from rsi.sandbox import (
    SandboxOutputError,
    SandboxTimeoutError,
    build_bwrap_argv,
    bwrap,
    evaluator,
    read_agent_output,
    repo_source_binds,
    run_sandboxed_evaluator,
    run_sandboxed_python,
)

requires_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")


@pytest.fixture
def fake_bwrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make build_bwrap_argv resolve bwrap without executing it."""
    real_which = shutil.which
    monkeypatch.setattr(
        bwrap.shutil,
        "which",
        lambda name, *a, **k: "/usr/bin/bwrap" if name == "bwrap" else real_which(name, *a, **k),
    )


def _pairs(argv: list[str], flag: str) -> list[tuple[str, ...]]:
    return [tuple(argv[i + 1 : i + 3]) for i, a in enumerate(argv) if a == flag]


def test_argv_is_allowlist(tmp_path: Path, fake_bwrap: None) -> None:
    argv = build_bwrap_argv(
        ["sh", "-c", "true"], workspace_dir=tmp_path / "ws", repo_root=tmp_path, evidence_dir=None
    )
    assert ("/", "/") not in _pairs(argv, "--ro-bind")
    assert "--unshare-all" in argv
    assert "--clearenv" in argv
    assert "--share-net" not in argv
    assert "--dev-bind" not in argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"


@pytest.mark.parametrize("network", [False, True])
def test_share_net_iff_network(tmp_path: Path, fake_bwrap: None, network: bool) -> None:
    argv = build_bwrap_argv(
        ["sh"], workspace_dir=tmp_path / "ws", repo_root=tmp_path, network=network
    )
    assert ("--share-net" in argv) is network


def test_env_overrides_are_setenv(tmp_path: Path, fake_bwrap: None) -> None:
    argv = build_bwrap_argv(
        ["sh"],
        workspace_dir=tmp_path / "ws",
        repo_root=tmp_path,
        env_overrides={"RSI_X": "1"},
    )
    assert ("RSI_X", "1") in _pairs(argv, "--setenv")


def test_pi_without_bwrap_raises() -> None:
    with pytest.raises(RuntimeError):
        run_sandboxed_agent(
            prompt_text="x",
            workspace_dir=Path("/nonexistent-ws"),
            evidence_dir=Path("/nonexistent-ev"),
            logs_dir=Path("/nonexistent-logs"),
            use_bwrap=False,
        )


def test_read_agent_output_accepts_plain_file(tmp_path: Path) -> None:
    (tmp_path / "recipe.json").write_text('{"a": 1}', encoding="utf-8")
    assert read_agent_output(tmp_path, "recipe.json") == '{"a": 1}'


def test_read_agent_output_rejects_bad_inputs(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(tmp_path / "target.txt")
    (tmp_path / "dir.txt").mkdir()
    (tmp_path / "big.txt").write_text("x" * 100, encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00")
    for name in ("link.txt", "dir.txt", "missing.txt", "../x", "a/b", "..", "big.txt", "bad.txt"):
        with pytest.raises(SandboxOutputError):
            read_agent_output(tmp_path, name, max_bytes=50)
    assert issubclass(SandboxOutputError, ValueError)


def test_truncates_logs_to_last_chars(tmp_path: Path) -> None:
    res = run_sandboxed_agent(
        prompt_text="p",
        workspace_dir=tmp_path / "ws",
        evidence_dir=tmp_path / "ev",
        logs_dir=tmp_path / "logs",
        agent_cmd=[sys.executable, "-c", "print('x' * 300000)"],
        use_bwrap=False,
        timeout=30,
    )
    assert res.returncode == 0
    assert len(res.stdout) == 200_000
    assert not res.sandboxed


def _pi_stream(turns: int, cost_per_turn: float, padding: int) -> str:
    """A pi `--mode json` transcript whose usage comes before `padding` bytes of filler."""
    events: list[dict[str, object]] = [{"type": "session", "version": 3}, {"type": "agent_start"}]
    for i in range(turns):
        events.append({"type": "turn_start"})
        events.append(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"step {i}"}],
                    "usage": {"cost": {"total": cost_per_turn}},
                },
            }
        )
        events.append({"type": "turn_end", "message": {"role": "assistant", "content": []}})
    events.append({"type": "agent_end", "messages": []})
    events.append({"type": "message_update", "text": "x" * padding})
    return "\n".join(json.dumps(event) for event in events) + "\n"


def test_pi_cost_and_turns_come_from_the_whole_stream() -> None:
    stream = _pi_stream(turns=15, cost_per_turn=0.004188, padding=300_000)
    assert len(stream) > 200_000
    payload = agent_result.agent_result_payload(
        "pi", model="m", effort="max", returncode=0, stdout=stream
    )
    assert payload["num_turns"] == 15
    assert payload["cost_usd"] == pytest.approx(0.06282)
    assert payload["text"] == "step 14"
    assert payload["is_error"] is False
    # Reading only the tail, as the old code did, loses the usage entirely.
    assert parse_agent_output("pi", stream[-200_000:])["cost_usd"] is None


def test_sidecar_round_trips_cost_and_turns(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    payload = agent_result.agent_result_payload(
        "pi", model="m", effort="max", returncode=0, stdout=_pi_stream(3, 0.01, 0)
    )
    assert agent_result.write_agent_result(logs, payload) == payload
    stored = agent_result.read_agent_result(logs)
    assert stored is not None and stored["cost_usd"] == pytest.approx(0.03)
    (logs / "agent_result.json").write_text("{not json", encoding="utf-8")
    assert agent_result.read_agent_result(logs) is None


def test_agent_result_sidecar_is_parsed_from_the_whole_log(tmp_path: Path) -> None:
    res = run_sandboxed_agent(
        prompt_text="p",
        workspace_dir=tmp_path / "ws",
        evidence_dir=tmp_path / "ev",
        logs_dir=tmp_path / "logs",
        agent_cmd=[sys.executable, "-c", "print('EARLY-MARKER'); print('y' * 300000)"],
        use_bwrap=False,
        timeout=30,
    )
    side = json.loads((tmp_path / "logs" / "agent_result.json").read_text(encoding="utf-8"))
    assert set(side) == {
        "backend",
        "model",
        "effort",
        "returncode",
        "cost_usd",
        "num_turns",
        "is_error",
        "text",
    }
    assert side["backend"] == "custom" and side["returncode"] == 0
    assert side["text"].startswith("EARLY-MARKER")
    assert len(res.stdout) == 200_000
    assert (res.cost_usd, res.num_turns) == (None, None)


def test_command_record_keeps_no_argv_or_host_paths(tmp_path: Path) -> None:
    run_sandboxed_agent(
        prompt_text="p",
        workspace_dir=tmp_path / "ws",
        evidence_dir=tmp_path / "ev",
        logs_dir=tmp_path / "logs",
        agent_cmd=[sys.executable, "-c", "pass"],
        use_bwrap=False,
        timeout=30,
    )
    text = (tmp_path / "logs" / "command.json").read_text(encoding="utf-8")
    assert json.loads(text) == {
        "backend": "custom",
        "model": None,
        "effort": None,
        "tools": None,
        "cli_version": None,
        "returncode": 0,
        "network": False,
        "use_bwrap": False,
    }
    assert str(tmp_path) not in text


@requires_bwrap
def test_command_record_reports_the_sandbox_it_used(tmp_path: Path) -> None:
    run_sandboxed_agent(
        prompt_text="p",
        workspace_dir=tmp_path / "ws",
        evidence_dir=tmp_path / "ev",
        logs_dir=tmp_path / "logs",
        agent_cmd=[shutil.which("sh") or "/bin/sh", "-c", "true"],
        timeout=30,
    )
    meta = json.loads((tmp_path / "logs" / "command.json").read_text(encoding="utf-8"))
    assert meta["use_bwrap"] is True
    assert meta["network"] is False


def test_timeout_still_records_the_run_for_the_budget_guard(tmp_path: Path) -> None:
    with pytest.raises(SandboxTimeoutError):
        run_sandboxed_agent(
            prompt_text="p",
            workspace_dir=tmp_path / "ws",
            evidence_dir=tmp_path / "ev",
            logs_dir=tmp_path / "logs",
            agent_cmd=[sys.executable, "-c", "import time; time.sleep(30)"],
            use_bwrap=False,
            timeout=1.0,
        )
    side = json.loads((tmp_path / "logs" / "agent_result.json").read_text(encoding="utf-8"))
    assert side["returncode"] == -1
    recovered = timed_out_result(tmp_path / "logs", sandboxed=False)
    assert recovered is not None
    assert recovered.returncode == -1 and recovered.elapsed_seconds is None
    assert timed_out_result(tmp_path / "missing", sandboxed=False) is None


def test_fake_agent_without_bwrap_runs_with_minimal_env(tmp_path: Path) -> None:
    res = run_sandboxed_agent(
        prompt_text="p",
        workspace_dir=tmp_path / "ws",
        evidence_dir=tmp_path / "ev",
        logs_dir=tmp_path / "logs",
        agent_cmd=[sys.executable, "-c", "import os; print(sorted(os.environ))"],
        use_bwrap=False,
        timeout=30,
    )
    assert "PYTHONDONTWRITEBYTECODE" in res.stdout
    assert "HOME" in res.stdout
    assert "RSI_CANARY" not in res.stdout


def _agent(tmp_path: Path, cmd: list[str], timeout: float = 30.0) -> Path:
    ws = tmp_path / "ws"
    run_sandboxed_agent(
        prompt_text="p",
        workspace_dir=ws,
        evidence_dir=tmp_path / "ev",
        logs_dir=tmp_path / "logs",
        agent_cmd=cmd,
        timeout=timeout,
    )
    return ws


@requires_bwrap
def test_canary_outside_workspace_is_hidden(tmp_path: Path) -> None:
    canary = tmp_path / "canary.txt"
    canary.write_text("CANARY-CONTENT", encoding="utf-8")
    script = (
        f"if [ -e {canary} ]; then cat {canary} > /tmp/work/out.txt; "
        "else echo missing > /tmp/work/out.txt; fi"
    )
    ws = _agent(tmp_path, [shutil.which("sh") or "/bin/sh", "-c", script])
    assert (ws / "out.txt").read_text() == "missing\n"


@requires_bwrap
def test_home_exposes_only_interpreter_scaffold(tmp_path: Path) -> None:
    home = Path.home()
    script = (
        f"ls -A {home} > /tmp/work/home.txt 2>/dev/null; "
        f"find {home} -maxdepth 2 > /tmp/work/depth2.txt 2>/dev/null; true"
    )
    ws = _agent(tmp_path, [shutil.which("sh") or "/bin/sh", "-c", script])
    top = set((ws / "home.txt").read_text().split())
    assert top <= {".local"}
    depth2 = (ws / "depth2.txt").read_text()
    for secret in (".ssh", ".pi", ".config", ".cache", ".gnupg", ".env", ".netrc"):
        assert f"/{secret}" not in depth2


@requires_bwrap
def test_repo_root_is_not_visible(tmp_path: Path) -> None:
    script = f"if [ -e {REPO_ROOT} ]; then ls -A {REPO_ROOT} | wc -l; else echo 0; fi > /tmp/work/repo.txt"
    ws = _agent(tmp_path, [shutil.which("sh") or "/bin/sh", "-c", script])
    assert (ws / "repo.txt").read_text().strip() == "0"


@requires_bwrap
def test_parent_env_is_not_inherited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_CANARY", "secret-value")
    ws = _agent(tmp_path, [shutil.which("sh") or "/bin/sh", "-c", "env > /tmp/work/env.txt"])
    content = (ws / "env.txt").read_text()
    assert "RSI_CANARY" not in content
    assert "secret-value" not in content
    assert "HOME=/tmp" in content


@requires_bwrap
def test_network_off_connect_fails(tmp_path: Path) -> None:
    script = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "    out = 'connected'\n"
        "except OSError as exc:\n"
        "    out = 'blocked'\n"
        "open('/tmp/work/net.txt', 'w').write(out)\n"
    )
    res = run_sandboxed_python(["-c", script], workspace_dir=tmp_path / "ws", timeout=30)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "ws" / "net.txt").read_text() == "blocked"


@requires_bwrap
def test_evidence_read_only_and_workspace_writable(tmp_path: Path) -> None:
    ev = tmp_path / "ev"
    ev.mkdir()
    script = (
        "echo x > /tmp/evidence/w.txt 2>/dev/null && echo wrote > /tmp/work/ev.txt "
        "|| echo denied > /tmp/work/ev.txt; echo ok > /tmp/work/work.txt"
    )
    ws = tmp_path / "ws"
    run_sandboxed_agent(
        prompt_text="p",
        workspace_dir=ws,
        evidence_dir=ev,
        logs_dir=tmp_path / "logs",
        agent_cmd=[shutil.which("sh") or "/bin/sh", "-c", script],
        timeout=30,
    )
    assert (ws / "ev.txt").read_text().strip() == "denied"
    assert (ws / "work.txt").read_text().strip() == "ok"
    assert not (ev / "w.txt").exists()


@requires_bwrap
def test_run_sandboxed_python_runs_venv_python(tmp_path: Path) -> None:
    res = run_sandboxed_python(
        ["-c", "import json, math; print(1)"], workspace_dir=tmp_path / "ws", timeout=60
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "1"


@requires_bwrap
def test_timeout_raises_and_leaves_no_child(tmp_path: Path) -> None:
    token = f"rsi-sandbox-timeout-{os.getpid()}-{time.time_ns()}"
    sleep_arg = "1234.5678"
    with pytest.raises(SandboxTimeoutError):
        run_sandboxed_agent(
            prompt_text="p",
            workspace_dir=tmp_path / "ws",
            evidence_dir=tmp_path / "ev",
            logs_dir=tmp_path / "logs",
            agent_cmd=[shutil.which("sh") or "/bin/sh", "-c", f"sleep {sleep_arg}; :", token],
            timeout=1.0,
        )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _host_processes_matching(token, f"sleep {sleep_arg}"):
            return
        time.sleep(0.1)
    pytest.fail("sandboxed child survived the timeout")


def _host_processes_matching(*needles: str) -> list[str]:
    found: list[str] = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            text = cmdline.read_bytes().decode(errors="replace").replace("\x00", " ")
        except OSError:
            continue
        if any(needle in text for needle in needles):
            found.append(text)
    return found


def _fake_gpu_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake HF hub with the pinned model caches plus one unrelated model that must stay hidden."""
    hub = tmp_path / "hub"
    for name in (
        "models--lerobot--VLA-JEPA-LIBERO",
        "models--Qwen--Qwen3-VL-2B-Instruct",
        "models--facebook--vjepa2-vitl-fpc64-256",
        "datasets--lerobot--libero-assets",
        "models--unrelated--model",
    ):
        (hub / name).mkdir(parents=True)
    repo = tmp_path / "repo"
    (repo / "evaluation").mkdir(parents=True)
    sources = {
        "models": {
            "baseline": {"repo": "lerobot/VLA-JEPA-LIBERO"},
            "qwen": {"repo": "Qwen/Qwen3-VL-2B-Instruct"},
            "world_model": {"repo": "facebook/vjepa2-vitl-fpc64-256"},
        },
        "simulator_assets": {"repo": "lerobot/libero-assets"},
    }
    (repo / "evaluation" / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    return repo


def test_gpu_argv_binds_existing_device_nodes_only(
    tmp_path: Path, fake_bwrap: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rsi_gpu, "gpu_device_paths", lambda: ["/dev/nvidia0", "/dev/nvidiactl"])
    argv = build_bwrap_argv(
        ["sh"],
        workspace_dir=tmp_path / "ws",
        repo_root=tmp_path,
        gpu=True,
        gpu_slots_dir=tmp_path / "slots",
    )
    devs = _pairs(argv, "--dev-bind")
    assert ("/dev/nvidia0", "/dev/nvidia0") in devs
    assert ("/dev/nvidiactl", "/dev/nvidiactl") in devs
    assert ("/dev", "/dev") not in devs
    assert ("/sys/module", "/sys/module") in _pairs(argv, "--ro-bind")
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
    assert ("MUJOCO_GL", "egl") in _pairs(argv, "--setenv")


def test_host_gpu_nodes_are_all_bound_when_present(tmp_path: Path, fake_bwrap: None) -> None:
    argv = build_bwrap_argv(
        ["sh"], workspace_dir=tmp_path / "ws", repo_root=tmp_path, gpu=True, gpu_slots_dir=tmp_path
    )
    devs = {pair[0] for pair in _pairs(argv, "--dev-bind")}
    for path in rsi_gpu.gpu_device_paths():
        assert path in devs


def test_gpu_argv_home_binds_only_named_caches(
    tmp_path: Path, fake_bwrap: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_gpu_hub(tmp_path, monkeypatch)
    argv = build_bwrap_argv(
        ["sh"],
        workspace_dir=tmp_path / "ws",
        repo_root=repo,
        gpu=True,
        gpu_slots_dir=tmp_path / "s",
    )
    sources = [src for src, _ in _pairs(argv, "--ro-bind")]
    cache = [s for s in sources if "/hub/" in s]
    assert sorted(Path(s).name for s in cache) == [
        "datasets--lerobot--libero-assets",
        "models--Qwen--Qwen3-VL-2B-Instruct",
        "models--facebook--vjepa2-vitl-fpc64-256",
        "models--lerobot--VLA-JEPA-LIBERO",
    ]
    home = str(Path.home())
    assert not [s for s in sources if s.startswith(home)]
    dests = [d for _, d in _pairs(argv, "--ro-bind")]
    assert "/tmp/.cache/huggingface/hub/models--lerobot--VLA-JEPA-LIBERO" in dests
    assert ("HF_HUB_OFFLINE", "1") in _pairs(argv, "--setenv")
    made = [pair[0] for pair in _pairs(argv, "--dir")]
    assert "/tmp/repo/outputs/evaluation/libero_config" in made


def test_gpu_off_keeps_cuda_hidden_and_no_extra_binds(tmp_path: Path, fake_bwrap: None) -> None:
    argv = build_bwrap_argv(["sh"], workspace_dir=tmp_path / "ws", repo_root=tmp_path)
    assert ("CUDA_VISIBLE_DEVICES", "") in _pairs(argv, "--setenv")
    assert "/sys/module" not in " ".join(argv)
    assert not _pairs(argv, "--dev-bind")


def test_gpu_network_flag_is_independent(tmp_path: Path, fake_bwrap: None) -> None:
    argv = build_bwrap_argv(
        ["sh"], workspace_dir=tmp_path / "ws", repo_root=tmp_path, gpu=True, gpu_slots_dir=tmp_path
    )
    assert "--share-net" not in argv


def test_source_and_workspace_modes(tmp_path: Path, fake_bwrap: None) -> None:
    (tmp_path / "rsi").mkdir()
    argv = build_bwrap_argv(
        ["sh"],
        workspace_dir=tmp_path / "ws",
        repo_root=tmp_path,
        source_ro={tmp_path / "rsi": "/tmp/repo/rsi"},
        workspace_ro=True,
        out_dir=tmp_path / "out",
    )
    assert "/tmp/repo/rsi" in [d for _, d in _pairs(argv, "--ro-bind")]
    assert ("PYTHONPATH", "/tmp/repo") in _pairs(argv, "--setenv")
    work = [p for p in _pairs(argv, "--ro-bind") if p[1] == "/tmp/work"]
    assert work and not [p for p in _pairs(argv, "--bind") if p[1] == "/tmp/work"]
    assert "/tmp/out" in [p[1] for p in _pairs(argv, "--bind")]


def test_repo_source_binds_only_existing_paths(tmp_path: Path) -> None:
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "studies" / "program-v1").mkdir(parents=True)
    (tmp_path / "studies" / "program-v1" / "protocol.json").write_text("{}", encoding="utf-8")
    binds = repo_source_binds(tmp_path)
    assert binds == {
        (tmp_path / "evaluation").resolve(): "/tmp/repo/evaluation",
        (tmp_path / "studies" / "program-v1" / "protocol.json").resolve(): (
            "/tmp/repo/studies/program-v1/protocol.json"
        ),
    }


class _Stub:
    """Fake bwrap builder that runs a stub evaluator on the host and records its arguments."""

    def __init__(self, tmp_path: Path, mode: str) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.mode = mode
        self.script = tmp_path / "stub_eval.py"
        self.script.write_text(
            "import json, os, sys, time\n"
            "out, mode = sys.argv[1], sys.argv[2]\n"
            "res = os.path.join(out, 'result.json')\n"
            "if mode == 'ok':\n"
            "    json.dump({'status': 'ok', 'phase': 'check'}, open(res, 'w'))\n"
            "elif mode == 'timeout':\n"
            "    time.sleep(30)\n"
            "elif mode == 'symlink':\n"
            "    open(os.path.join(out, 'real.json'), 'w').write('{}')\n"
            "    os.symlink(os.path.join(out, 'real.json'), res)\n"
            "elif mode == 'list':\n"
            "    json.dump([1], open(res, 'w'))\n",
            encoding="utf-8",
        )

    def __call__(self, inner: list[str], **kwargs: object) -> list[str]:
        self.calls.append((inner, kwargs))
        return [sys.executable, str(self.script), str(kwargs["out_dir"]), self.mode]


def _run_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, timeout: float = 20.0
) -> tuple[dict[str, object], _Stub]:
    stub = _Stub(tmp_path, mode)
    monkeypatch.setattr(evaluator, "build_bwrap_argv", stub)
    result = run_sandboxed_evaluator(
        tmp_path / "ws",
        "check",
        tmp_path / "out",
        timeout=timeout,
        gpu_slots_dir=tmp_path / "slots",
    )
    return result, stub


def test_evaluator_returns_parsed_result_and_sandbox_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, stub = _run_stub(tmp_path, monkeypatch, "ok")
    assert result == {"status": "ok", "phase": "check"}
    inner, kwargs = stub.calls[0]
    assert inner[1:4] == ["-P", "-m", "evaluation.compression.study"]
    assert inner[inner.index("--phase") + 1] == "check"
    assert inner[inner.index("--workspace") + 1] == "/tmp/work"
    assert inner[inner.index("--out") + 1] == "/tmp/out"
    assert kwargs["workspace_ro"] is True
    assert kwargs["network"] is False
    assert kwargs["gpu"] is True


def test_evaluator_timeout_is_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _run_stub(tmp_path, monkeypatch, "timeout", timeout=1.0)
    assert result["status"] == "evaluator_failure"


@pytest.mark.parametrize("mode", ["symlink", "list"])
def test_evaluator_rejects_symlink_and_non_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    result, _ = _run_stub(tmp_path, monkeypatch, mode)
    assert result["status"] == "evaluator_failure"


def test_evaluator_missing_result_is_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _run_stub(tmp_path, monkeypatch, "missing")
    assert result["status"] == "evaluator_failure"


def test_evaluator_rejects_unknown_phase(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_sandboxed_evaluator(tmp_path, "final", tmp_path / "out", timeout=1.0)
