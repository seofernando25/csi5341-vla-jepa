"""Bubblewrap allowlist sandbox, process runner and agent-run harness."""

from __future__ import annotations

import functools
import json
import os
import resource
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from rsi.agent_backends import (
    REPO_ROOT,
    SANDBOX_EVIDENCE,
    SANDBOX_OUT,
    SANDBOX_REPO,
    SANDBOX_WORK,
    STUDY_DIR,
    AgentSpec,
    agent_cli_version,
    agent_command,
    load_agent_spec,
    parse_agent_output,
)
from rsi.gpu import GPU_ENV, GPU_SLOTS_DIR, gpu_bind_args, gpu_slot
from rsi.workspace import WORKSPACE_CHECKS

SANDBOX_VENV = "/tmp/venv"
EVALUATOR_MODULE = "evaluation.compression.study"
EVALUATOR_PHASES = ("check", "smoke", "development")
EVALUATOR_RESULT = "result.json"
EVALUATOR_LOG = "log.txt"
MAX_RESULT_BYTES = 8 * 1024 * 1024
DEFAULT_STUDY = "studies/program-v1"
DEFAULT_CALIBRATION = "outputs/compression/calibration.pt"
REPO_SOURCE_PATHS = (
    "evaluation",
    "rsi",
    "pyproject.toml",
    "outputs/compression/calibration.pt",
    "outputs/compression/references.pt",
    "outputs/compression/vendor",
    "outputs/evaluation/observations.pt",
    "studies/evaluation/initial_states.json",
    "studies/evaluation/observation_bank.json",
)
MAX_LOG_CHARS = 200_000
MAX_AGENT_TEXT_CHARS = 65_536
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_MEMORY_BYTES = 4 * 1024 * 1024 * 1024
AGENT_RESULT = "agent_result.json"
COMMAND_FIELDS = (
    "backend",
    "model",
    "effort",
    "tools",
    "cli_version",
    "returncode",
    "network",
    "use_bwrap",
)
HOST_ROOT_DIRS = ("bin", "lib", "lib64", "sbin")
ETC_FILES = (
    "resolv.conf",
    "hosts",
    "nsswitch.conf",
    "ssl",
    "ca-certificates",
    "pki",
    "ld.so.cache",
    "ld.so.conf",
    "ld.so.conf.d",
    "localtime",
    "passwd",
    "group",
)


class SandboxTimeoutError(TimeoutError):
    """Raised when a sandboxed process exceeds its wall-clock timeout."""


class SandboxOutputError(ValueError):
    """Raised when an agent output file is not a small, plain, UTF-8 regular file."""


@dataclass(frozen=True)
class AgentRunResult:
    """Outcome of one sandboxed discovery or policy-development agent invocation."""

    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float | None
    sandboxed: bool
    logs_dir: str
    backend: str
    model: str | None
    effort: str | None
    num_turns: int | None
    cost_usd: float | None
    is_error: bool | None = None
    text: str = ""


def _is_under_usr(path: Path) -> bool:
    usr = Path("/usr")
    return path == usr or usr in path.parents


def _bind_ro(cmd: list[str], path: Path) -> None:
    """Bind a host directory read-only at its own path unless /usr already covers it."""
    if not _is_under_usr(path):
        cmd.extend(["--ro-bind", str(path), str(path)])


def _system_mounts(cmd: list[str]) -> None:
    """Append the read-only system view: /usr, root compatibility links and selected /etc files."""
    cmd.extend(["--ro-bind", "/usr", "/usr"])
    for name in HOST_ROOT_DIRS:
        host = Path("/") / name
        if host.is_symlink():
            cmd.extend(["--symlink", os.readlink(host), str(host)])
        elif host.exists():
            cmd.extend(["--ro-bind", str(host), str(host)])
    for name in ETC_FILES:
        host = Path("/etc") / name
        if host.exists():
            cmd.extend(["--ro-bind-try", str(host), str(host)])


def _rewrite_venv_exe(exe: str, venv_link: Path) -> str:
    """Map an executable inside the repo venv to its path inside the sandbox."""
    try:
        rel = Path(exe).absolute().relative_to(venv_link.absolute())
    except ValueError:
        return exe
    return f"{SANDBOX_VENV}/{rel.as_posix()}"


def _python_homes(venv_python: Path) -> list[Path]:
    """Return the interpreter home dirs along the venv python's symlink chain."""
    homes: dict[Path, None] = {}
    cur = venv_python
    for _ in range(40):
        if not cur.is_symlink():
            break
        target = Path(os.readlink(cur))
        if not target.is_absolute():
            target = cur.parent / target
        homes[target.parent.parent] = None
        cur = target
    homes[cur.resolve().parent.parent] = None
    return list(homes)


def study_rel(study_dir: Path | None, repo_root: Path = REPO_ROOT) -> str | None:
    """Repo-relative study directory: the program-v1 default when none is named, None if outside."""
    if study_dir is None:
        return DEFAULT_STUDY
    try:
        return Path(study_dir).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return None


def study_protocol(study_dir: Path | None, repo_root: Path = REPO_ROOT) -> str | None:
    """Repo-relative `protocol.json` of the named study, or None when it cannot be mounted."""
    rel = study_rel(study_dir, repo_root)
    return f"{rel}/protocol.json" if rel else None


@dataclass(frozen=True)
class DiscoveryVisibility:
    """What a study's protocol hides from discovery agents; empty for a study that hides nothing."""

    repo_exclude: tuple[str, ...] = ()
    workspace_check: str = "evaluator"


def _protocol_block(study_dir: Path | None, repo_root: Path, key: str) -> dict[str, Any]:
    """One top-level object of the study's protocol, or {} when the study or block is absent."""
    rel = study_protocol(study_dir, repo_root)
    path = Path(repo_root) / rel if rel else None
    if path is None or not path.is_file():
        return {}
    block = json.loads(path.read_text(encoding="utf-8")).get(key)
    return block if isinstance(block, dict) else {}


def discovery_visibility(
    study_dir: Path | None = None, repo_root: Path = REPO_ROOT
) -> DiscoveryVisibility:
    """The exclusions a study's protocol sets for discovery agents, validated."""
    block = _protocol_block(study_dir, repo_root, "discovery_visibility")
    check = str(block.get("workspace_check") or "evaluator")
    if check not in WORKSPACE_CHECKS:
        raise ValueError(f"workspace_check {check!r} is not one of {WORKSPACE_CHECKS}")
    exclude = tuple(str(item) for item in block.get("repo_exclude") or ())
    return DiscoveryVisibility(repo_exclude=exclude, workspace_check=check)


def calibration_artifact(study_dir: Path | None = None, repo_root: Path = REPO_ROOT) -> str | None:
    """Repo-relative calibration artifact the study names, or None for the frozen default."""
    artifact = _protocol_block(study_dir, repo_root, "calibration").get("artifact")
    return str(artifact) if artifact else None


def repo_source_binds(
    repo_root: Path = REPO_ROOT, *, study_dir: Path | None = None, exclude: tuple[str, ...] = ()
) -> dict[Path, str]:
    """Read-only repo source and evaluator inputs, each at its path under /tmp/repo, if present.

    A repo-relative path in `exclude` is left out. When it names a file inside a listed
    directory, that directory is bound child by child instead, so the file cannot be reached.
    The study's own `protocol.json` is the only protocol bound, so a run under one study cannot
    read another study's phases, gates or seed. A study that names its own calibration artifact
    gets that file instead of the frozen default, at its own path.
    """
    repo = Path(repo_root).resolve()
    binds: dict[Path, str] = {}
    protocol = study_protocol(study_dir, repo)
    artifact = calibration_artifact(study_dir, repo)
    paths = list(REPO_SOURCE_PATHS)
    if artifact and artifact != DEFAULT_CALIBRATION:
        paths = [rel for rel in paths if rel != DEFAULT_CALIBRATION]
        paths.append(artifact)
    for rel in (*paths, *([protocol] if protocol else [])):
        src = repo / rel
        if not src.exists():
            continue
        if not any(item == rel or item.startswith(f"{rel}/") for item in exclude):
            binds[src] = f"{SANDBOX_REPO}/{rel}"
            continue
        for child in sorted(src.iterdir()):
            child_rel = f"{rel}/{child.name}"
            if child_rel not in exclude:
                binds[child] = f"{SANDBOX_REPO}/{child_rel}"
    return binds


def build_bwrap_argv(
    inner_argv: list[str],
    *,
    workspace_dir: Path,
    evidence_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
    mount_venv: bool = True,
    extra_ro_binds: dict[Path, str] | None = None,
    writable_dirs: dict[Path, str] | None = None,
    network: bool = False,
    env_overrides: dict[str, str] | None = None,
    gpu: bool = False,
    gpu_slots_dir: Path | None = None,
    source_ro: dict[Path, str] | None = None,
    workspace_ro: bool = False,
    out_dir: Path | None = None,
) -> list[str]:
    """Wrap `inner_argv` in an allowlist bubblewrap root with a workspace and optional GPU access."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap (bwrap) is required for isolated agent execution")

    workspace = Path(workspace_dir).resolve()
    if not workspace_ro:
        workspace.mkdir(parents=True, exist_ok=True)
    repo = Path(repo_root).resolve()
    venv_link = repo / ".venv"

    cmd = [bwrap, "--die-with-parent", "--new-session", "--unshare-all", "--cap-drop", "ALL"]
    cmd.append("--clearenv")
    if network:
        cmd.append("--share-net")
    _system_mounts(cmd)
    cmd.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"])
    if gpu:
        slots = Path(gpu_slots_dir or GPU_SLOTS_DIR).resolve()
        slots.mkdir(parents=True, exist_ok=True)
        cmd.extend(gpu_bind_args(repo, slots))
    workspace_mode = "--ro-bind" if workspace_ro else "--bind"
    cmd.extend([workspace_mode, str(workspace), SANDBOX_WORK, "--chdir", SANDBOX_WORK])
    if out_dir is not None:
        cmd.extend(["--bind", str(Path(out_dir).resolve()), SANDBOX_OUT])
    # Writable scratch outside the workspace (e.g. a package dir). Bound before the read-only
    # mounts so a later read-only bind wins if the two ever name the same path.
    for src, dst in (writable_dirs or {}).items():
        target = Path(src).resolve()
        target.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--bind", str(target), dst])
    if evidence_dir is not None:
        cmd.extend(["--ro-bind", str(Path(evidence_dir).resolve()), SANDBOX_EVIDENCE])
    for src, dst in (extra_ro_binds or {}).items():
        cmd.extend(["--ro-bind", str(Path(src).resolve()), dst])
    for src, dst in (source_ro or {}).items():
        cmd.extend(["--ro-bind", str(Path(src).resolve()), dst])

    venv_path: list[str] = []
    resolved_inner = list(inner_argv)
    env: dict[str, str] = {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if gpu:
        env.update(GPU_ENV)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if source_ro:
        env["PYTHONPATH"] = SANDBOX_REPO
    if mount_venv and venv_link.is_dir():
        cmd.extend(["--ro-bind", str(venv_link.resolve()), SANDBOX_VENV])
        for home in _python_homes(venv_link / "bin" / "python"):
            _bind_ro(cmd, home)
        env["VIRTUAL_ENV"] = SANDBOX_VENV
        venv_path.append(f"{SANDBOX_VENV}/bin")
        if resolved_inner:
            resolved_inner[0] = _rewrite_venv_exe(resolved_inner[0], venv_link)

    env = {"PATH": ":".join([*venv_path, "/usr/bin"]), **env}
    env.update(env_overrides or {})
    for key, value in env.items():
        cmd.extend(["--setenv", key, value])

    cmd.append("--")
    cmd.extend(resolved_inner)
    return cmd


def _limit_child(cpu_seconds: int | None, memory_bytes: int | None) -> None:
    """Cap file size, core dumps, CPU time and address space in the child."""
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if cpu_seconds is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    if memory_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def _signal_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def _reap(proc: subprocess.Popen[str]) -> None:
    """Stop a still-running process group, then kill any leftover members."""
    if proc.poll() is None:
        _signal_group(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _signal_group(proc.pid, signal.SIGKILL)
            proc.wait()
    _signal_group(proc.pid, signal.SIGKILL)


def _run_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout: IO[str],
    stderr: IO[str],
    stdin_payload: str | None,
    timeout: float,
    cpu_seconds: int | None,
    memory_bytes: int | None,
) -> int:
    """Run `argv` in its own session with limits; always reap its process group."""
    started = time.monotonic()
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE if stdin_payload is not None else subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        text=True,
        env=env,
        start_new_session=True,
        # The child only calls setrlimit before exec, which is safe with threads.
        preexec_fn=functools.partial(_limit_child, cpu_seconds, memory_bytes),  # noqa: PLW1509
    )
    try:
        if stdin_payload is not None and proc.stdin is not None:
            try:
                proc.stdin.write(stdin_payload)
                proc.stdin.close()
            except BrokenPipeError:
                pass
        while proc.poll() is None:
            if time.monotonic() - started > timeout:
                raise SandboxTimeoutError(f"Sandboxed process timed out after {timeout:.1f}s")
            time.sleep(0.1)
    finally:
        _reap(proc)
    return int(proc.returncode or 0)


def _read_tail(path: Path) -> str:
    """Return the last MAX_LOG_CHARS characters of a log file without reading it whole."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > MAX_LOG_CHARS * 4:
            fh.seek(size - MAX_LOG_CHARS * 4)
        data = fh.read()
    return data.decode("utf-8", errors="replace")[-MAX_LOG_CHARS:]


def _read_all(path: Path) -> str:
    """Return a whole log as text; the child's RLIMIT_FSIZE bounds how large it can grow."""
    return path.read_bytes().decode("utf-8", errors="replace")


def _clip(text: str, limit: int = MAX_AGENT_TEXT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit]


def agent_result_payload(
    backend: str,
    *,
    model: str | None,
    effort: str | None,
    returncode: int,
    stdout: str,
) -> dict[str, Any]:
    """Compact, path-free record of one agent run, parsed from the whole stdout stream.

    `stdout` must be the complete transcript: a pi stream puts its per-turn usage before the
    final (large) message, so parsing only the tail loses cost and turn counts.
    """
    parsed = parse_agent_output(backend, stdout)
    return {
        "backend": backend,
        "model": model,
        "effort": effort,
        "returncode": returncode,
        "cost_usd": parsed["cost_usd"],
        "num_turns": parsed["num_turns"],
        "is_error": parsed["is_error"],
        "text": _clip(parsed["text"]),
    }


def read_agent_result(logs_dir: Path) -> dict[str, Any] | None:
    """The `agent_result.json` sidecar of one run, or None when it is missing or malformed."""
    try:
        data = json.loads((Path(logs_dir) / AGENT_RESULT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("returncode"), int):
        return None
    for key in ("cost_usd", "num_turns"):
        value = data.get(key)
        if value is not None and (not isinstance(value, int | float) or isinstance(value, bool)):
            return None
    return data


def write_agent_result(logs_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Write the sidecar and return the values read back from it."""
    path = Path(logs_dir) / AGENT_RESULT
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    stored = read_agent_result(logs_dir)
    if stored is None:
        raise SandboxOutputError(f"{AGENT_RESULT} is unreadable after writing")
    return stored


def timed_out_result(logs_dir: Path, *, sandboxed: bool) -> AgentRunResult | None:
    """Cost and turns recovered from a timed-out run's sidecar; None when it wrote none."""
    data = read_agent_result(logs_dir)
    if data is None:
        return None
    return AgentRunResult(
        returncode=int(data["returncode"]),
        stdout="",
        stderr="",
        elapsed_seconds=None,
        sandboxed=sandboxed,
        logs_dir=str(logs_dir),
        backend=str(data.get("backend") or "custom"),
        model=data.get("model"),
        effort=data.get("effort"),
        num_turns=data.get("num_turns"),
        cost_usd=data.get("cost_usd"),
        is_error=data.get("is_error"),
        text=str(data.get("text") or ""),
    )


def _write_command(logs: Path, meta: dict[str, Any]) -> None:
    """Write the path-free command record: backend identity, exit code and sandbox mode."""
    record = {name: meta.get(name) for name in COMMAND_FIELDS}
    (Path(logs) / "command.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _finish_run(
    logs: Path,
    meta: dict[str, Any],
    *,
    backend: str,
    spec: AgentSpec | None,
    returncode: int,
    out_path: Path,
) -> dict[str, Any]:
    """Record the exit code, then write and read back the parsed-result sidecar."""
    _write_command(logs, {**meta, "returncode": returncode})
    payload = agent_result_payload(
        backend,
        model=spec.model if spec else None,
        effort=spec.effort if spec else None,
        returncode=returncode,
        stdout=_read_all(out_path),
    )
    return write_agent_result(logs, payload)


def _host_path_env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def read_agent_output(workspace_dir: Path, name: str, max_bytes: int = 65536) -> str:
    """Read a small plain UTF-8 file from the workspace without following symlinks."""
    if not name or name in {".", ".."} or any(c in name for c in ("/", "\\", "\x00")):
        raise SandboxOutputError(f"invalid output name: {name!r}")
    if ".." in name:
        raise SandboxOutputError(f"invalid output name: {name!r}")
    path = Path(workspace_dir) / name
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise SandboxOutputError(f"cannot open {name}: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SandboxOutputError(f"{name} is not a regular file")
        if st.st_size > max_bytes:
            raise SandboxOutputError(f"{name} exceeds {max_bytes} bytes")
        data = os.read(fd, max_bytes + 1)
    finally:
        os.close(fd)
    if len(data) > max_bytes:
        raise SandboxOutputError(f"{name} exceeds {max_bytes} bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SandboxOutputError(f"{name} is not valid UTF-8") from exc


def _run_captured(
    argv: list[str],
    *,
    workspace: Path,
    timeout: float,
    cpu_seconds: int | None,
    memory_bytes: int | None,
) -> tuple[int, str, str]:
    """Run `argv` with a temporary log pair and return (returncode, stdout tail, stderr tail)."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "stdout.txt"
        err_path = Path(tmp) / "stderr.txt"
        with (
            out_path.open("w", encoding="utf-8") as fout,
            err_path.open("w", encoding="utf-8") as ferr,
        ):
            returncode = _run_process(
                argv,
                cwd=workspace,
                env=_host_path_env(),
                stdout=fout,
                stderr=ferr,
                stdin_payload=None,
                timeout=timeout,
                cpu_seconds=cpu_seconds,
                memory_bytes=memory_bytes,
            )
        return returncode, _read_tail(out_path), _read_tail(err_path)


def run_sandboxed_python(
    args: list[str],
    *,
    workspace_dir: Path,
    evidence_dir: Path | None = None,
    extra_ro_binds: dict[Path, str] | None = None,
    env_overrides: dict[str, str] | None = None,
    timeout: float = 60.0,
    cpu_seconds: int | None = 30,
    memory_bytes: int | None = None,
    gpu: bool = False,
    network: bool = False,
    source_ro: dict[Path, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the repo venv python on untrusted code inside the sandbox.

    `memory_bytes=None` means the CPU default cap, or no address-space cap when `gpu` is set,
    because CUDA reserves large virtual ranges.
    """
    if not (REPO_ROOT / ".venv" / "bin" / "python").exists():
        raise RuntimeError("repo .venv is required to run sandboxed python")
    workspace = Path(workspace_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    argv = build_bwrap_argv(
        [f"{SANDBOX_VENV}/bin/python", *args],
        workspace_dir=workspace,
        evidence_dir=evidence_dir,
        extra_ro_binds=extra_ro_binds,
        network=network,
        env_overrides=env_overrides,
        gpu=gpu,
        source_ro=source_ro,
    )
    if memory_bytes is None and not gpu:
        memory_bytes = MAX_MEMORY_BYTES
    returncode, stdout, stderr = _run_captured(
        argv,
        workspace=workspace,
        timeout=timeout,
        cpu_seconds=cpu_seconds,
        memory_bytes=memory_bytes,
    )
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def _evaluator_failure(message: str) -> dict[str, Any]:
    return {"status": "evaluator_failure", "error": message[-2000:]}


def evaluator_argv(phase: str, protocol: str) -> list[str]:
    """Sandboxed `solution` evaluator argv for one phase, reading `protocol` under /tmp/repo."""
    return [
        f"{SANDBOX_VENV}/bin/python",
        "-P",
        "-m",
        EVALUATOR_MODULE,
        "solution",
        "--workspace",
        SANDBOX_WORK,
        "--phase",
        phase,
        "--out",
        SANDBOX_OUT,
        "--protocol",
        protocol,
    ]


def run_sandboxed_evaluator(
    workspace_dir: Path,
    phase: str,
    out_dir: Path,
    *,
    timeout: float,
    gpu_slots_dir: Path | None = None,
    gpu_slot_count: int = 1,
    study_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the fixed `solution` evaluator on a read-only workspace; return its parsed result.json.

    Network is off, the workspace is read-only at /tmp/work and `out_dir` is writable at /tmp/out.
    The run holds one GPU slot. `timeout` bounds the slot wait and the run separately. Any
    timeout, missing or invalid result becomes `{"status": "evaluator_failure"}`. `study_dir`
    selects the `protocol.json` the evaluator reads; without it the program-v1 study is used.
    """
    if phase not in EVALUATOR_PHASES:
        raise ValueError(f"phase must be one of {EVALUATOR_PHASES}; got {phase!r}")
    workspace = Path(workspace_dir).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in (EVALUATOR_RESULT, EVALUATOR_LOG):
        (out / stale).unlink(missing_ok=True)
    protocol = study_protocol(study_dir)
    if protocol is None:
        raise ValueError(
            f"study dir {study_dir} is not under the repo; its protocol cannot be read"
        )
    inner = evaluator_argv(phase, f"{SANDBOX_REPO}/{protocol}")
    argv = build_bwrap_argv(
        inner,
        workspace_dir=workspace,
        workspace_ro=True,
        out_dir=out,
        network=False,
        gpu=True,
        gpu_slots_dir=gpu_slots_dir,
        source_ro=repo_source_binds(REPO_ROOT, study_dir=study_dir),
    )
    try:
        with gpu_slot(Path(gpu_slots_dir or GPU_SLOTS_DIR), gpu_slot_count, timeout=timeout):
            returncode, _, stderr = _run_captured(
                argv,
                workspace=out,
                timeout=timeout,
                cpu_seconds=None,
                memory_bytes=None,
            )
    except (TimeoutError, SandboxTimeoutError) as exc:
        return _evaluator_failure(str(exc))
    try:
        text = read_agent_output(out, EVALUATOR_RESULT, max_bytes=MAX_RESULT_BYTES)
        result = json.loads(text)
    except (SandboxOutputError, ValueError) as exc:
        return _evaluator_failure(f"no valid {EVALUATOR_RESULT} (rc={returncode}): {exc}\n{stderr}")
    if not isinstance(result, dict):
        return _evaluator_failure(f"{EVALUATOR_RESULT} is not a JSON object")
    return result


def run_sandboxed_agent(
    *,
    prompt_text: str,
    workspace_dir: Path,
    evidence_dir: Path,
    logs_dir: Path,
    agent_cmd: str | list[str] | None = None,
    agent: AgentSpec | None = None,
    timeout: float = 240.0,
    use_bwrap: bool = True,
    repo_root: Path = REPO_ROOT,
    extra_ro_binds: dict[Path, str] | None = None,
    writable_dirs: dict[Path, str] | None = None,
    env_overrides: dict[str, str] | None = None,
    network: bool = False,
    gpu: bool = False,
    source_ro: dict[Path, str] | None = None,
    gpu_slots_dir: Path | None = None,
    study_dir: Path | None = None,
) -> AgentRunResult:
    """Execute a discovery or policy-dev agent in an allowlist sandbox (`bwrap`).

    Built-in backends (`agent`, default: the protocol's agent) always get network access and
    require bwrap. A custom `agent_cmd` (tests, wrappers) gets network only with `network=True`
    and may run without bwrap via `use_bwrap=False`; it receives the prompt on stdin.
    `study_dir` selects the study whose protocol names the default agent and the protocol the
    workspace's own `tools/check.py` evaluates against.
    """
    custom = agent_cmd is not None
    if not custom and not use_bwrap:
        raise RuntimeError("the built-in agents require bwrap; use_bwrap=False needs agent_cmd")
    workspace = Path(workspace_dir).resolve()
    evidence = Path(evidence_dir).resolve()
    logs = Path(logs_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    (evidence / "TASK.md").write_text(prompt_text, encoding="utf-8")

    spec = None if custom else (agent or load_agent_spec(study_dir or STUDY_DIR))
    backend_binds: dict[Path, str] = {}
    backend_env: dict[str, str] = {}
    if spec is None:
        if isinstance(agent_cmd, str):
            inner_argv = shlex.split(agent_cmd)
        else:
            inner_argv = list(agent_cmd or [])
    else:
        inner_argv, backend_binds, backend_env = agent_command(spec)
    backend = spec.backend if spec else "custom"
    # The agent's `tools/check.py` runs the same evaluator; point it at this study's protocol.
    protocol = study_protocol(study_dir, repo_root)
    study_env = {"VLA_PROTOCOL": f"{SANDBOX_REPO}/{protocol}"} if protocol else {}

    if use_bwrap:
        argv = build_bwrap_argv(
            inner_argv,
            workspace_dir=workspace,
            evidence_dir=evidence,
            repo_root=repo_root,
            mount_venv=True,
            extra_ro_binds={**backend_binds, **(extra_ro_binds or {})},
            writable_dirs=writable_dirs,
            network=network or not custom,
            env_overrides={**backend_env, **study_env, **(env_overrides or {})},
            gpu=gpu,
            gpu_slots_dir=gpu_slots_dir,
            source_ro=source_ro,
        )
        env = _host_path_env()
    else:
        argv = inner_argv
        env = {
            **_host_path_env(),
            "HOME": str(workspace),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "RSI_WORKSPACE_DIR": str(workspace),
            "RSI_EVIDENCE_DIR": str(evidence),
            **study_env,
            **(env_overrides or {}),
        }

    # `network` and `use_bwrap` are the sandbox mode the watch page reports; no argv or paths.
    meta: dict[str, Any] = {
        "backend": backend,
        "model": spec.model if spec else None,
        "effort": spec.effort if spec else None,
        "tools": spec.tools if spec else None,
        "cli_version": agent_cli_version(spec.backend) if spec else None,
        "returncode": None,
        "network": network or not custom,
        "use_bwrap": use_bwrap,
    }
    _write_command(logs, meta)

    out_path = logs / "stdout.log"
    err_path = logs / "stderr.log"
    started = time.monotonic()
    try:
        with (
            out_path.open("w", encoding="utf-8") as fout,
            err_path.open("w", encoding="utf-8") as ferr,
        ):
            returncode = _run_process(
                argv,
                cwd=workspace,
                env=env,
                stdout=fout,
                stderr=ferr,
                stdin_payload=prompt_text if custom else None,
                timeout=timeout,
                cpu_seconds=None,
                memory_bytes=None,
            )
    except SandboxTimeoutError:
        # The killed run still spent money; keep its partial usage for the loop's budget guard.
        _finish_run(
            logs,
            meta,
            backend=backend,
            spec=spec,
            returncode=-1,
            out_path=out_path,
        )
        raise
    elapsed = round(time.monotonic() - started, 3)
    side = _finish_run(
        logs, meta, backend=backend, spec=spec, returncode=returncode, out_path=out_path
    )
    return AgentRunResult(
        returncode=returncode,
        stdout=_read_tail(out_path),
        stderr=_read_tail(err_path),
        elapsed_seconds=elapsed,
        sandboxed=use_bwrap,
        logs_dir=str(logs),
        backend=backend,
        model=spec.model if spec else None,
        effort=spec.effort if spec else None,
        num_turns=side["num_turns"],
        cost_usd=side["cost_usd"],
        is_error=side["is_error"],
        text=str(side.get("text") or ""),
    )
