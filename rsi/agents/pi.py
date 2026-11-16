"""The pi backend: headless argv, a provider-filtered config bound into the sandbox, and preflight checks."""

from __future__ import annotations

import atexit
import functools
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rsi.agents.spec import TASK_PROMPT, AgentSpec, Command
from rsi.paths import SANDBOX_PI_AGENT

MISTRAL_PLACEHOLDER_API = "openai-completions"
MISTRAL_NATIVE_API = "mistral-conversations"


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


def require_pi_auth(spec: AgentSpec) -> None:
    """Fail before a GPU slot is taken when the model config or the provider key is unusable."""
    _require_pi_model(spec)
    status = pi_auth_status(pi_provider(spec.model), spec.model)
    if status is not None and status != "ready":
        raise RuntimeError(f"pi auth check failed for {spec.model}: status={status}")
