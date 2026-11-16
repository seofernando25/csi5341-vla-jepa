"""Node snapshots in one git store per cycle, forks and exports, output validation and evidence."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
MAX_PROPOSAL_BYTES = 256 * 1024
MAX_SOLUTION_BYTES = 1024 * 1024
DIFF_CHAR_LIMIT = 200_000
STORE_NAME = "snapshots.git"
ROOT_TAG = "node/root"
SKIP_NAMES = frozenset({"__pycache__", ".check"})
SKIP_SUFFIXES = (".pyc",)
EVIDENCE_FILES = ("proposal.md", "score.json", "error.txt")
OUTPUT_CONTRACT = "a node must write both proposal.md and solution.py; both are required"
TEMPLATE_DIR = Path(__file__).resolve().parent / "workspace_template"
MEASUREMENT_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "compression"
# The template carries both check tools; a study's protocol picks one and the other is dropped.
WORKSPACE_CHECKS = ("evaluator", "self_contained")
# The self-contained check measures with copies of the evaluator's own primitives.
MEASUREMENT_MODULES = ("resident.py", "memory.py")
GIT_DATE = "2000-01-01T00:00:00Z"
GIT_SETTINGS = (
    "core.autocrlf=false",
    "core.fileMode=true",
    "core.hooksPath=/dev/null",
    "core.excludesFile=/dev/null",
    "gc.auto=0",
)
_CELL_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
Entry = tuple[str, str, str]


class WorkspaceError(ValueError):
    """A workspace, snapshot, git store or output violates the rules for Dream-RSI nodes."""


def _skipped(name: str) -> bool:
    return name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES)


def _scan(src: Path) -> tuple[list[str], list[str]]:
    """Relative dirs and files to keep; rejects special entries, `.git` and oversize trees."""
    dirs: list[str] = []
    files: list[str] = []
    total = 0
    pending: list[tuple[str, bool]] = [("", False)]
    while pending:
        rel, skip = pending.pop()
        with os.scandir(src / rel) as entries:
            children = sorted(entries, key=lambda e: e.name)
        for entry in children:
            name = f"{rel}/{entry.name}" if rel else entry.name
            info = entry.stat(follow_symlinks=False)
            if entry.name == ".git":
                raise WorkspaceError(f".git not allowed in workspace: {name}")
            if "\n" in entry.name:
                raise WorkspaceError(f"newline in file name not allowed: {name!r}")
            child_skip = skip or _skipped(entry.name)
            if stat.S_ISLNK(info.st_mode):
                raise WorkspaceError(f"symlink not allowed in workspace: {name}")
            if stat.S_ISDIR(info.st_mode):
                if not child_skip:
                    dirs.append(name)
                pending.append((name, child_skip))
            elif stat.S_ISREG(info.st_mode):
                if not child_skip:
                    files.append(name)
                    total += info.st_size
            else:
                raise WorkspaceError(f"special file not allowed in workspace: {name}")
    if total > MAX_WORKSPACE_BYTES:
        raise WorkspaceError(f"workspace is {total} bytes, above {MAX_WORKSPACE_BYTES}")
    return sorted(dirs), sorted(files)


def _owner_write(path: Path) -> int:
    return stat.S_IWUSR | stat.S_IXUSR if path.is_dir() else stat.S_IWUSR


def _copy_tree(src: Path, dst: Path) -> None:
    src, dst = Path(src), Path(dst)
    if not src.is_dir() or src.is_symlink():
        raise WorkspaceError(f"source is not a directory: {src}")
    if dst.exists() or dst.is_symlink():
        raise WorkspaceError(f"destination exists: {dst}")
    dirs, files = _scan(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.mkdir()
    try:
        for rel in dirs:
            (dst / rel).mkdir()
        for rel in files:
            shutil.copy2(src / rel, dst / rel)
        for rel in [*reversed(dirs), *reversed(files), ""]:
            target = dst / rel
            if rel:
                shutil.copystat(src / rel, target)
            target.chmod(target.stat().st_mode & ~0o222)
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise


def _apply_check_tool(dst: Path, check_tool: str) -> None:
    """Install one of the template's two check tools as `tools/check.py` and drop the other.

    `evaluator` runs the fixed evaluator out of `/tmp/repo`. `self_contained` measures with
    copies of the evaluator's own resident walk and payload sweep, so it still works when a
    study hides `evaluation/compression` from the workspace.
    """
    if check_tool not in WORKSPACE_CHECKS:
        raise WorkspaceError(f"unknown workspace check {check_tool!r}; expected {WORKSPACE_CHECKS}")
    tools = Path(dst) / "tools"
    variant, target = tools / "check_selfcontained.py", tools / "check.py"
    tools.chmod(0o755)
    target.chmod(0o644)
    if check_tool == "self_contained":
        target.write_bytes(variant.read_bytes())
        for name in MEASUREMENT_MODULES:
            copy = tools / name
            copy.write_bytes((MEASUREMENT_DIR / name).read_bytes())
            copy.chmod(0o444)
    variant.chmod(0o644)
    variant.unlink()
    target.chmod(0o444)
    tools.chmod(0o555)


def create_root_workspace(dst: Path, check_tool: str = "evaluator") -> Path:
    """Copy the template into dst as a frozen tree and return dst."""
    _copy_tree(TEMPLATE_DIR, Path(dst))
    _apply_check_tool(Path(dst), check_tool)
    return Path(dst)


def _env(index: Path | None = None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "rsi",
        "GIT_AUTHOR_EMAIL": "rsi@localhost",
        "GIT_COMMITTER_NAME": "rsi",
        "GIT_COMMITTER_EMAIL": "rsi@localhost",
        "GIT_AUTHOR_DATE": GIT_DATE,
        "GIT_COMMITTER_DATE": GIT_DATE,
    }
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    return env


def _run(
    argv: list[str], *, input: bytes | None = None, index: Path | None = None
) -> subprocess.CompletedProcess[bytes]:
    config = [arg for setting in GIT_SETTINGS for arg in ("-c", setting)]
    return subprocess.run(
        ["git", *config, *argv],
        input=input,
        env=_env(index),
        capture_output=True,
        check=False,
    )


def _git(store: Path, *args: str, input: bytes | None = None, index: Path | None = None) -> bytes:
    done = _run(["--git-dir", str(store), *args], input=input, index=index)
    if done.returncode:
        message = done.stderr.decode(errors="replace").strip()
        raise WorkspaceError(f"git {args[0]} failed: {message}")
    return done.stdout


def _ref(store: Path, name: str) -> str | None:
    done = _run(["--git-dir", str(store), "rev-parse", "-q", "--verify", name])
    return done.stdout.decode().strip() if done.returncode == 0 else None


def _safe_cell_id(cell_id: str) -> str:
    if not _CELL_ID.match(cell_id) or cell_id in {".", ".."}:
        raise WorkspaceError(f"unsafe cell_id: {cell_id!r}")
    return cell_id


def init_store(cycle_dir: Path) -> Path:
    """Create the cycle's bare snapshot repository under cycle_dir; idempotent."""
    store = Path(cycle_dir) / STORE_NAME
    if not (store / "HEAD").is_file():
        store.parent.mkdir(parents=True, exist_ok=True)
        done = _run(["init", "--bare", "-q", str(store)])
        if done.returncode:
            raise WorkspaceError(f"git init failed: {done.stderr.decode(errors='replace')}")
    return store


def _mode(path: Path) -> str:
    return "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"


def _stage(store: Path, entries: Iterable[Entry], index: Path) -> str:
    """Write the entries (mode, blob, path) into a fresh index and return its tree id."""
    payload = b"".join(f"{mode} {sha}\t{rel}\0".encode() for mode, sha, rel in entries)
    _git(store, "update-index", "-z", "--add", "--index-info", input=payload, index=index)
    return _git(store, "write-tree", index=index).decode().strip()


def _commit_tree(store: Path, src: Path, parent: str | None) -> str:
    """Commit the kept files of src, with fixed identity and date, on top of parent."""
    _, files = _scan(src)
    paths = "".join(f"{src / rel}\n" for rel in files).encode()
    blobs = _git(store, "hash-object", "-w", "--no-filters", "--stdin-paths", input=paths)
    shas = blobs.decode().split()
    entries = [(_mode(src / rel), sha, rel) for rel, sha in zip(files, shas, strict=True)]
    with tempfile.TemporaryDirectory(prefix="rsi-index-") as tmp:
        tree = _stage(store, entries, Path(tmp) / "index")
    parents = ["-p", parent] if parent else []
    return _git(store, "commit-tree", tree, *parents, "-m", "node").decode().strip()


def commit_root(store: Path, check_tool: str = "evaluator") -> str:
    """Commit the template as the root node (tag node/root); idempotent, returns its commit id."""
    existing = _ref(store, f"refs/tags/{ROOT_TAG}")
    if existing is not None:
        return existing
    with tempfile.TemporaryDirectory(prefix="rsi-template-") as tmp:
        root = create_root_workspace(Path(tmp) / "template", check_tool)
        commit = _commit_tree(store, root, None)
    _git(store, "update-ref", f"refs/tags/{ROOT_TAG}", commit)
    return commit


def snapshot(store: Path, workspace: Path, parent_commit: str, cell_id: str) -> str:
    """Commit a node's workspace on top of parent_commit, tag it node/<cell_id>, return the commit."""
    cell = _safe_cell_id(cell_id)
    commit = _commit_tree(store, Path(workspace), parent_commit)
    _git(store, "update-ref", f"refs/tags/node/{cell}", commit)
    return commit


def _tree_entries(store: Path, commit: str) -> list[Entry]:
    raw = _git(store, "ls-tree", "-r", "-z", "--full-tree", commit)
    entries: list[Entry] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        mode, _kind, sha = meta.decode().split()
        entries.append((mode, sha, path.decode("utf-8")))
    return entries


def _materialize(store: Path, commit: str, dst: Path, *, read_only: bool) -> Path:
    if dst.exists() or dst.is_symlink():
        raise WorkspaceError(f"destination exists: {dst}")
    entries = _tree_entries(store, commit)
    request = "".join(f"{sha}\n" for _mode, sha, _rel in entries).encode()
    blobs = _git(store, "cat-file", "--batch", input=request)
    dst.mkdir(parents=True)
    try:
        folders = {dst}
        pos = 0
        for mode, _sha, rel in entries:
            header_end = blobs.index(b"\n", pos)
            _sha_hex, _kind, size_text = blobs[pos:header_end].decode().split()
            size = int(size_text)
            body = blobs[header_end + 1 : header_end + 1 + size]
            pos = header_end + 2 + size
            target = dst / rel
            folders.update(dst / parent for parent in Path(rel).parents if str(parent) != ".")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            perm = 0o755 if mode == "100755" else 0o644
            target.chmod(perm & ~0o222 if read_only else perm)
        for folder in sorted(folders, key=lambda p: len(p.parts), reverse=True):
            folder.chmod(0o555 if read_only else 0o755)
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    return dst


def fork(store: Path, parent_commit: str, dst: Path) -> Path:
    """Writable checkout of parent_commit at dst, without any git metadata; returns dst."""
    return _materialize(Path(store), parent_commit, Path(dst), read_only=False)


def export(store: Path, commit: str, dst: Path) -> Path:
    """Read-only materialised copy of commit at dst for the evaluator; returns dst."""
    return _materialize(Path(store), commit, Path(dst), read_only=True)


def code_digest(store: Path, commit: str) -> str:
    """Git tree id of a commit's code with the top-level proposal.md left out."""
    entries = [entry for entry in _tree_entries(store, commit) if entry[2] != "proposal.md"]
    with tempfile.TemporaryDirectory(prefix="rsi-index-") as tmp:
        return _stage(store, entries, Path(tmp) / "index")


def diff_vs_parent(store: Path, commit: str) -> str:
    """Text diff from the commit's parent (or the empty tree for the root), capped in length."""
    parents = _git(store, "rev-list", "--parents", "-n", "1", commit).decode().split()[1:]
    if parents:
        base = parents[0]
    else:
        base = _git(store, "hash-object", "-t", "tree", "-w", "--stdin", input=b"").decode()
    text = _git(store, "diff", "--no-color", "--no-ext-diff", base.strip(), commit)
    return text.decode("utf-8", errors="replace")[:DIFF_CHAR_LIMIT]


def _read_regular(workspace: Path, name: str, limit: int) -> bytes:
    path = workspace / name
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        # Name the file and the reason, never the absolute host path.
        raise WorkspaceError(f"{name} missing or not openable ({exc.strerror or exc})") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise WorkspaceError(f"{name} is not a regular file")
        if info.st_size > limit:
            raise WorkspaceError(f"{name} exceeds {limit} bytes")
        data = os.read(fd, limit + 1)
    finally:
        os.close(fd)
    if len(data) > limit:
        raise WorkspaceError(f"{name} exceeds {limit} bytes")
    return data


def validate_outputs(workspace: Path) -> None:
    """Raise WorkspaceError unless proposal.md and solution.py are sound node outputs.

    Both files are required: a solution-only workspace fails, and the message names the missing
    file and the contract so the failure is not read as a model-quality result.
    """
    try:
        _check_outputs(Path(workspace))
    except WorkspaceError as exc:
        raise WorkspaceError(f"{exc}; {OUTPUT_CONTRACT}") from exc


def _check_outputs(root: Path) -> None:
    _read_regular(root, "proposal.md", MAX_PROPOSAL_BYTES)
    source = _read_regular(root, "solution.py", MAX_SOLUTION_BYTES)
    try:
        tree = ast.parse(source.decode("utf-8"), filename="solution.py")
    except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
        raise WorkspaceError(f"solution.py does not parse: {exc}") from exc
    if not any(isinstance(node, ast.FunctionDef) and node.name == "compress" for node in tree.body):
        raise WorkspaceError("solution.py defines no top-level compress")


def write_evidence(evidence_dir: Path, nodes: list[dict], tree_index: dict) -> None:
    """Write tree.json and every prior node's proposal, score and error files into evidence_dir."""
    root = Path(evidence_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "tree.json").write_text(json.dumps(tree_index, indent=2) + "\n", encoding="utf-8")
    for node in nodes:
        cell = _safe_cell_id(str(node["cell_id"]))
        source = Path(node["node_dir"])
        target = root / "nodes" / cell
        target.mkdir(parents=True, exist_ok=True)
        for name in EVIDENCE_FILES:
            item = source / name
            if item.is_symlink():
                raise WorkspaceError(f"symlink not allowed in evidence: {item}")
            if item.is_file():
                shutil.copyfile(item, target / name)
