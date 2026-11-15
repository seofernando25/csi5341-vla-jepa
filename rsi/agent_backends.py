"""Headless agent backends for Dream-RSI: pi (default) and Claude Code, with output parsing."""

from __future__ import annotations

import atexit
import functools
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rsi.runner import load_protocol
from rsi.watch_events import pi_cost, pi_text

if TYPE_CHECKING:
    from rsi.sandbox import AgentRunResult

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDY_DIR = REPO_ROOT / "studies" / "program-v1"
SANDBOX_WORK = "/tmp/work"
SANDBOX_OUT = "/tmp/out"
SANDBOX_REPO = "/tmp/repo"
SANDBOX_HF_HUB = "/tmp/.cache/huggingface/hub"
SANDBOX_EVIDENCE = "/tmp/evidence"
SANDBOX_PI_AGENT = "/tmp/pi-agent"
SANDBOX_CLAUDE_CREDENTIALS = "/tmp/.claude/.credentials.json"
BACKENDS = ("claude", "pi")
DEFAULT_PI_MODEL = "mistral/deepseek-v4-1-flash-1017"
DEFAULT_BACKEND = "pi"
DEFAULT_CLAUDE_MODEL = "haiku"
DEFAULT_PI_EFFORT = "max"
DEFAULT_CLAUDE_EFFORT = "low"
DEFAULT_TOOLS = "read,write,edit"
CLAUDE_TOOL_NAMES = {"read": "Read", "write": "Write", "edit": "Edit", "bash": "Bash"}
MISTRAL_PLACEHOLDER_API = "openai-completions"
MISTRAL_NATIVE_API = "mistral-conversations"
TASK_PROMPT = (
    f"Read {SANDBOX_EVIDENCE}/TASK.md in full, inspect all files in {SANDBOX_EVIDENCE} "
    f"and {SANDBOX_WORK}, and follow every instruction exactly."
)

Command = tuple[list[str], dict[Path, str], dict[str, str]]


@dataclass(frozen=True)
class AgentSpec:
    """Which headless agent runs, on which model, at which effort, with which tools."""

    backend: str
    model: str
    effort: str | None
    tools: str


def resolve_pi_executable() -> str:
    """Return the `pi` CLI path or raise RuntimeError if unavailable."""
    found = shutil.which("pi")
    if found:
        return str(Path(found).resolve())
    fallback = Path.home() / ".bun/bin/pi"
    if fallback.exists():
        return str(fallback.resolve())
    raise RuntimeError("pi CLI not found in PATH or ~/.bun/bin/pi")


def _which_resolved(name: str) -> Path:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found in PATH; required for the pi agent")
    return Path(found).resolve()


def _outside_usr(path: Path) -> bool:
    usr = Path("/usr")
    return path != usr and usr not in path.parents


def _pi_package_dir(pi_exe: Path) -> Path:
    """Return the global node_modules directory that holds the resolved pi package."""
    for parent in pi_exe.parents:
        if parent.name == "node_modules":
            return parent
    return pi_exe.parent


def pi_argv(spec: AgentSpec, prompt_arg: str) -> list[str]:
    """Build deterministic, extension-free headless `pi` arguments.

    `--mode json` streams one JSON object per line while the agent works, so the watch page can
    follow thinking, text and tool events instead of waiting for a buffered final answer.
    """
    argv = [
        resolve_pi_executable(),
        "--no-extensions",
        "--no-mcp",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--no-session",
        "--print",
        "--mode",
        "json",
        "--model",
        spec.model,
    ]
    if spec.effort:
        argv.extend(["--thinking", spec.effort])
    if spec.tools:
        argv.extend(["--tools", spec.tools])
    else:
        argv.append("--no-tools")
    argv.extend(["-p", prompt_arg])
    return argv


def pi_provider(model: str) -> str:
    """Provider name of a `provider/model` pi model id."""
    return model.split("/", 1)[0] if "/" in model else model


def _pi_agent_dir() -> Path:
    return Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi/agent")).resolve()


@functools.cache
def _pi_staging_dir() -> Path:
    """Process-lifetime 0700 directory holding the filtered pi config bound into the sandbox."""
    path = Path(tempfile.mkdtemp(prefix="pi-agent-"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def _write_private(path: Path, text: str) -> None:
    """Write `text` to `path` readable by the owner only."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    path.chmod(0o600)


def _pi_filtered_config(provider: str) -> dict[str, str]:
    """`auth.json` and `models.json` reduced to the selected provider.

    The host `auth.json` also holds unrelated provider credentials, so only the selected
    provider's entry is copied into the sandbox; the agent is untrusted and has network access.
    """
    pi_dir = _pi_agent_dir()
    auth_path = pi_dir / "auth.json"
    try:
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"pi auth file missing or unreadable at {auth_path}") from exc
    entry = auth.get(provider) if isinstance(auth, dict) else {}
    entry = entry if isinstance(entry, dict) else {}
    if not entry:
        raise RuntimeError(f"pi auth.json has no credential for provider {provider!r}")
    files = {"auth.json": json.dumps({provider: entry}, indent=2) + "\n"}
    try:
        models = json.loads((pi_dir / "models.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return files
    providers = models.get("providers") if isinstance(models, dict) else None
    if isinstance(providers, dict) and provider in providers:
        filtered: dict[str, Any] = {"providers": {provider: providers[provider]}}
        if models.get("$schema") is not None:
            filtered["$schema"] = models["$schema"]
        files["models.json"] = json.dumps(filtered, indent=2) + "\n"
    return files


def pi_command(spec: AgentSpec, prompt_arg: str = TASK_PROMPT) -> Command:
    """Return pi's argv, read-only binds (package, node, filtered config) and env."""
    exe = Path(resolve_pi_executable())
    node = _which_resolved("node")
    binds: dict[Path, str] = {}
    for path in (_pi_package_dir(exe), node.parent.parent):
        if _outside_usr(path):
            binds[path] = str(path)
    staging = _pi_staging_dir()
    for name, text in _pi_filtered_config(pi_provider(spec.model)).items():
        path = staging / name
        _write_private(path, text)
        binds[path] = f"{SANDBOX_PI_AGENT}/{name}"
    env = {"PATH": f"/usr/bin:{node.parent}", "PI_CODING_AGENT_DIR": SANDBOX_PI_AGENT}
    return pi_argv(spec, prompt_arg), binds, env


@functools.cache
def pi_auth_status(provider: str, model: str) -> str | None:
    """`pi auth check` status for one provider/model, or None when the CLI cannot be queried."""
    try:
        out = subprocess.run(
            [
                resolve_pi_executable(),
                "auth",
                "check",
                "--provider",
                provider,
                "--model",
                model,
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        status = json.loads(out.stdout.strip() or "{}").get("status")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        return None
    return status if isinstance(status, str) else None


def _pi_provider_block(provider: str) -> dict[str, Any]:
    """The local `models.json` provider block, or {} when pi's builtin entry is used."""
    try:
        models = json.loads((_pi_agent_dir() / "models.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    providers = models.get("providers") if isinstance(models, dict) else None
    entry = providers.get(provider) if isinstance(providers, dict) else None
    return entry if isinstance(entry, dict) else {}


def _pi_model_setting(provider: str, model: str, key: str) -> str | None:
    """One `models.json` setting for `model`: a per-model override, else the provider block."""
    block = _pi_provider_block(provider)
    model_id = model.split("/", 1)[1] if "/" in model else model
    for item in block.get("models") or []:
        if isinstance(item, dict) and item.get("id") == model_id:
            value = item.get(key)
            if isinstance(value, str):
                return value
    value = block.get(key)
    return value if isinstance(value, str) else None


def _pi_model_api(provider: str, model: str) -> str | None:
    """The `api` adapter pi will use for `model`."""
    return _pi_model_setting(provider, model, "api")


def _mistral_config_problem(provider: str, model: str) -> str | None:
    """Why a local `mistral` block cannot serve `model`, or None when it is usable."""
    api = _pi_model_api(provider, model)
    base_url = _pi_model_setting(provider, model, "baseUrl")
    problems: list[str] = []
    if api == MISTRAL_PLACEHOLDER_API:
        problems.append(
            f"api={MISTRAL_PLACEHOLDER_API} stringifies Mistral thinking arrays as [object Object]"
        )
    if base_url is not None and base_url.rstrip("/").endswith("/v1"):
        problems.append(f"baseUrl {base_url!r} appends /v1, which the native adapter answers 404")
    return "; ".join(problems) or None


def _require_pi_model(spec: AgentSpec) -> None:
    """Reject a local `mistral` block that cannot serve the model through the native adapter.

    `openai-completions` appends `delta.content` to the answer text without a type check, so
    Mistral's thinking arrays become the literal string `[object Object]` in the agent's own
    history and the run degrades into repeating it. `mistral-conversations` is correct but its
    base URL must not end in `/v1`: the native adapter appends `/v1/...` itself and 404s.
    pi's builtin `mistral` entry has both settings right.
    """
    provider = pi_provider(spec.model)
    if provider != "mistral":
        return
    problem = _mistral_config_problem(provider, spec.model)
    if problem is None:
        return
    raise RuntimeError(
        f"models.json cannot serve {spec.model}: {problem}; set api={MISTRAL_NATIVE_API} with "
        'baseUrl "https://api.mistral.ai" (no /v1), or delete the local "mistral" provider block '
        "and use pi's builtin entry"
    )


def _require_pi_auth(spec: AgentSpec) -> None:
    """Fail before a GPU slot is taken when the model config or the provider key is unusable."""
    _require_pi_model(spec)
    status = pi_auth_status(pi_provider(spec.model), spec.model)
    if status is not None and status != "ready":
        raise RuntimeError(f"pi auth check failed for {spec.model}: status={status}")


def _claude_tools(tools: str) -> str:
    names = [name.strip() for name in tools.split(",") if name.strip()]
    if not names or any(name not in CLAUDE_TOOL_NAMES for name in names):
        allowed = ",".join(CLAUDE_TOOL_NAMES)
        raise ValueError(f"claude backend allows only {allowed}; got {tools!r}")
    return ",".join(CLAUDE_TOOL_NAMES[name] for name in names)


def claude_command(spec: AgentSpec, prompt_arg: str = TASK_PROMPT) -> Command:
    """Return Claude Code headless argv; the binary and credentials are the only host binds."""
    found = shutil.which("claude")
    if not found:
        raise RuntimeError("claude CLI not found in PATH; required for the claude agent")
    exe = Path(found).resolve()
    creds = Path.home() / ".claude" / ".credentials.json"
    if not creds.is_file():
        raise RuntimeError(f"claude credentials file missing at {creds}; log in with claude")
    argv = [str(exe), "-p", prompt_arg, "--model", spec.model]
    if spec.effort:
        argv.extend(["--effort", spec.effort])
    claude_tools = _claude_tools(spec.tools)
    # acceptEdits covers file edits only; headless Bash needs an explicit allow rule.
    bash_allow = ["--allowedTools", "Bash(*)"] if "Bash" in claude_tools.split(",") else []
    argv.extend(
        [
            "--tools",
            claude_tools,
            "--permission-mode",
            "acceptEdits",
            *bash_allow,
            "--add-dir",
            SANDBOX_EVIDENCE,
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
    )
    binds = {exe: str(exe), creds: SANDBOX_CLAUDE_CREDENTIALS}
    return argv, binds, {}


def agent_command(spec: AgentSpec, prompt_arg: str = TASK_PROMPT) -> Command:
    """Dispatch to the backend named by `spec`."""
    if spec.backend == "claude":
        return claude_command(spec, prompt_arg)
    if spec.backend == "pi":
        _require_pi_auth(spec)
        return pi_command(spec, prompt_arg)
    raise ValueError(f"unknown agent backend {spec.backend!r}")


def _number(value: Any) -> float | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def _result_object(stdout: str) -> dict[str, Any] | None:
    """Final `result` object of a stream-json transcript, or the whole stdout of the old single-object form."""
    try:
        whole = json.loads(stdout)
    except ValueError:
        whole = None
    if isinstance(whole, dict) and ("result" in whole or whole.get("type") == "result"):
        return whole
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            obj = json.loads(text)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            return obj
    return None


def _pi_stream_result(stdout: str) -> dict[str, Any] | None:
    """Final text, turn count and summed cost of a `pi --mode json` transcript, or None."""
    turns = 0
    cost = 0.0
    text: str | None = None
    errored = False
    finished = False
    seen = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            obj = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        kind = obj.get("type")
        if kind == "turn_end":
            seen = True
            turns += 1
            message = obj.get("message")
            if isinstance(message, dict):
                body = pi_text(message)
                if body:
                    text = body
                if message.get("stopReason") in ("error", "aborted"):
                    errored = True
        elif kind == "message_end":
            message = obj.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                seen = True
                body = pi_text(message)
                if body:
                    text = body
                if message.get("stopReason") in ("error", "aborted"):
                    errored = True
                message_cost = pi_cost(obj)
                if message_cost is not None:
                    cost += message_cost
        elif kind == "agent_end":
            seen = finished = True
            if text is None:
                messages = obj.get("messages")
                for message in reversed(messages if isinstance(messages, list) else []):
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        text = pi_text(message) or None
                        break
    if not seen:
        return None
    return {
        "text": text or "",
        "num_turns": turns or None,
        "cost_usd": round(cost, 6) if cost else None,
        "is_error": errored if finished else None,
    }


def parse_agent_output(backend: str, stdout: str) -> dict[str, Any]:
    """Return text, num_turns, cost_usd and is_error from one agent's stdout.

    Claude emits stream-json (one object per line, final `result` line) or the old single object.
    pi emits `--mode json` events; its cost and turn count are summed from the stream.
    """
    plain: dict[str, Any] = {"text": stdout, "num_turns": None, "cost_usd": None, "is_error": None}
    if backend == "pi":
        return _pi_stream_result(stdout) or plain
    if backend != "claude":
        return plain
    obj = _result_object(stdout)
    if obj is None:
        return plain
    turns = _number(obj.get("num_turns"))
    is_error = obj.get("is_error")
    return {
        "text": str(obj.get("result") or ""),
        "num_turns": int(turns) if turns is not None else None,
        "cost_usd": _number(obj.get("total_cost_usd")),
        "is_error": is_error if isinstance(is_error, bool) else None,
    }


def _default_model(backend: str) -> str:
    if backend == "pi":
        return os.environ.get("RSI_PI_MODEL") or DEFAULT_PI_MODEL
    return DEFAULT_CLAUDE_MODEL


def _default_effort(backend: str) -> str | None:
    if backend == "pi":
        return os.environ.get("RSI_PI_EFFORT") or DEFAULT_PI_EFFORT
    return DEFAULT_CLAUDE_EFFORT


def load_agent_spec(
    study_dir: Path = STUDY_DIR,
    *,
    backend: str | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> AgentSpec:
    """Read `dream_rsi.agent` from the protocol; overrides ignore it when the backend changes."""
    drsi = load_protocol(study_dir)["dream_rsi"]
    stored: dict[str, Any] = drsi.get("agent", {})
    proto_backend = str(stored.get("backend", DEFAULT_BACKEND))
    chosen = backend or proto_backend
    base = stored if chosen == proto_backend else {}
    return AgentSpec(
        backend=chosen,
        model=model or str(base.get("model") or _default_model(chosen)),
        effort=effort or base.get("effort") or _default_effort(chosen),
        tools=str(drsi.get("policy_dev_tools", DEFAULT_TOOLS)),
    )


def default_agent_spec() -> AgentSpec:
    """The agent named by the repository's frozen protocol."""
    return load_agent_spec(STUDY_DIR)


@functools.cache
def agent_cli_version(backend: str) -> str | None:
    """Host-side `--version` string of the backend CLI, captured once per process."""
    try:
        exe = shutil.which("claude") if backend == "claude" else resolve_pi_executable()
        if not exe:
            return None
        out = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def agent_record(spec: AgentSpec | None, results: Sequence[AgentRunResult]) -> dict[str, Any]:
    """Manifest block: backend identity plus summed cost and turns that the backend reported."""
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    turns = [r.num_turns for r in results if r.num_turns is not None]
    return {
        "backend": spec.backend if spec else "custom",
        "model": spec.model if spec else None,
        "effort": spec.effort if spec else None,
        "tools": spec.tools if spec else None,
        "cli_version": agent_cli_version(spec.backend) if spec else None,
        "calls": len(results),
        "num_turns": sum(turns) if turns else None,
        "cost_usd": round(sum(costs), 6) if costs else None,
    }
