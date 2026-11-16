"""The bubblewrap allowlist root: system mounts, repo source binds and the `bwrap` argv."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from rsi.gpu import GPU_ENV, GPU_SLOTS_DIR, gpu_bind_args
from rsi.paths import (
    REPO_ROOT,
    SANDBOX_EVIDENCE,
    SANDBOX_OUT,
    SANDBOX_REPO,
    SANDBOX_VENV,
    SANDBOX_WORK,
)

DEFAULT_STUDY = "studies/program-v1"
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


def repo_source_binds(
    repo_root: Path = REPO_ROOT, *, study_dir: Path | None = None, exclude: tuple[str, ...] = ()
) -> dict[Path, str]:
    """Read-only repo source and evaluator inputs, each at its path under /tmp/repo, if present.

    A repo-relative path in `exclude` is left out. When it names a file inside a listed
    directory, that directory is bound child by child instead, so the file cannot be reached.
    The study's own `protocol.json` is the only protocol bound, so a run under one study cannot
    read another study's phases, gates or seed.
    """
    repo = Path(repo_root).resolve()
    binds: dict[Path, str] = {}
    protocol = study_protocol(study_dir, repo)
    for rel in (*REPO_SOURCE_PATHS, *([protocol] if protocol else [])):
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
