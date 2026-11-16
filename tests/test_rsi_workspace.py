"""Node snapshots in the cycle git store, forks, exports, output validation, evidence and the check tool."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from rsi import workspace
from rsi.paths import REPO_ROOT
from rsi.workspace import (
    WorkspaceError,
    code_digest,
    commit_root,
    create_root_workspace,
    diff_vs_parent,
    export,
    fork,
    init_store,
    snapshot,
    validate_outputs,
    write_evidence,
)

GOOD_SOLUTION = "def compress(policy, calibration):\n    return policy\n"


def make_source(root: Path) -> Path:
    src = root / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "solution.py").write_text(GOOD_SOLUTION)
    (src / "pkg" / "helper.py").write_text("VALUE = 3\n")
    (src / "run.sh").write_text("#!/bin/sh\necho hi\n")
    os.chmod(src / "run.sh", 0o750)
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "helper.cpython-312.pyc").write_bytes(b"\0")
    (src / ".check").mkdir()
    (src / ".check" / "old.json").write_text("{}")
    (src / "stale.pyc").write_bytes(b"\0")
    return src


def make_store(root: Path) -> tuple[Path, str]:
    store = init_store(root / "cycle")
    return store, commit_root(store)


def git(store: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "--git-dir", str(store), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def test_init_store_and_root_are_idempotent(tmp_path: Path) -> None:
    store = init_store(tmp_path / "cycle")
    assert store == tmp_path / "cycle" / "snapshots.git"
    assert init_store(tmp_path / "cycle") == store
    root = commit_root(store)
    assert commit_root(store) == root
    assert git(store, "rev-parse", "node/root").strip() == root


def test_snapshot_commits_and_skips_outputs(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    commit = snapshot(store, make_source(tmp_path), root, "a")
    assert git(store, "rev-parse", "node/a").strip() == commit
    listed = git(store, "ls-tree", "-r", "--name-only", commit).split()
    assert "solution.py" in listed and "pkg/helper.py" in listed and "run.sh" in listed
    for gone in ("__pycache__", ".check", "stale.pyc"):
        assert not any(name.startswith(gone) or f"/{gone}" in name for name in listed)


def test_identical_content_and_parent_give_identical_commit(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    first = snapshot(store, make_source(tmp_path / "one"), root, "a")
    second = snapshot(store, make_source(tmp_path / "two"), root, "b")
    assert first == second


def test_different_parent_gives_different_commit(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    src = make_source(tmp_path)
    child = snapshot(store, src, root, "a")
    other = snapshot(store, src, child, "b")
    assert other != child
    assert git(store, "rev-parse", f"{other}^").strip() == child


def test_fork_reproduces_files_modes_and_has_no_git(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    src = make_source(tmp_path)
    commit = snapshot(store, src, root, "a")
    child = fork(store, commit, tmp_path / "child")
    assert child == tmp_path / "child"
    assert not (child / ".git").exists()
    for rel in ("solution.py", "pkg/helper.py", "run.sh"):
        assert (child / rel).read_bytes() == (src / rel).read_bytes()
    assert stat.S_IMODE((child / "run.sh").stat().st_mode) == 0o755
    assert stat.S_IMODE((child / "solution.py").stat().st_mode) & stat.S_IWUSR
    (child / "solution.py").write_text(GOOD_SOLUTION + "# edit\n")
    assert git(store, "show", f"{commit}:solution.py") == GOOD_SOLUTION


def test_fork_refuses_existing_destination(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    (tmp_path / "taken").mkdir()
    with pytest.raises(WorkspaceError, match="exists"):
        fork(store, root, tmp_path / "taken")


def test_symlink_is_rejected_by_name(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    src = make_source(tmp_path)
    os.symlink("/etc/passwd", src / "leak")
    with pytest.raises(WorkspaceError, match="leak"):
        snapshot(store, src, root, "a")
    assert not git(store, "tag", "--list", "node/a").strip()


def test_fifo_is_rejected_by_name(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    src = make_source(tmp_path)
    os.mkfifo(src / "pipe")
    with pytest.raises(WorkspaceError, match="pipe"):
        snapshot(store, src, root, "a")


def test_oversize_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, root = make_store(tmp_path)
    monkeypatch.setattr(workspace, "MAX_WORKSPACE_BYTES", 8)
    with pytest.raises(WorkspaceError, match="above"):
        snapshot(store, make_source(tmp_path), root, "a")


def test_git_directory_in_workspace_is_rejected(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    src = make_source(tmp_path)
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(WorkspaceError, match=r"\.git"):
        snapshot(store, src, root, "a")


def test_workspace_gitignore_cannot_hide_solution(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    src = make_source(tmp_path)
    (src / ".gitignore").write_text("solution.py\n*.py\n")
    commit = snapshot(store, src, root, "a")
    assert "solution.py" in git(store, "ls-tree", "--name-only", commit).split()
    assert git(store, "show", f"{commit}:solution.py") == GOOD_SOLUTION


def test_diff_vs_parent_shows_child_change(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    src = make_source(tmp_path)
    parent = snapshot(store, src, root, "a")
    child_ws = fork(store, parent, tmp_path / "child")
    (child_ws / "solution.py").write_text(GOOD_SOLUTION + "# child edit\n")
    child = snapshot(store, child_ws, parent, "b")
    diff = diff_vs_parent(store, child)
    assert "+# child edit" in diff
    assert "helper.py" not in diff


def test_root_diff_is_whole_template(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    assert "solution.py" in diff_vs_parent(store, root)


def test_code_digest_ignores_only_proposal(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    src = make_source(tmp_path)
    first = snapshot(store, src, root, "a")
    (src / "proposal.md").write_text("# plan\n")
    second = snapshot(store, src, root, "b")
    assert code_digest(store, first) == code_digest(store, second)
    (src / "solution.py").write_text(GOOD_SOLUTION + "# changed\n")
    third = snapshot(store, src, root, "c")
    assert code_digest(store, third) != code_digest(store, first)


def test_export_is_read_only_with_content(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    commit = snapshot(store, make_source(tmp_path), root, "a")
    out = export(store, commit, tmp_path / "export")
    assert (out / "pkg" / "helper.py").read_text() == "VALUE = 3\n"
    for path in [out, out / "solution.py", out / "pkg", out / "pkg" / "helper.py"]:
        assert path.stat().st_mode & 0o222 == 0, path
    assert stat.S_IMODE((out / "run.sh").stat().st_mode) == 0o555


def test_three_node_chain_is_git_ancestry(tmp_path: Path) -> None:
    store, root = make_store(tmp_path)
    wa = fork(store, root, tmp_path / "wa")
    (wa / "solution.py").write_text(GOOD_SOLUTION + "# a\n")
    a = snapshot(store, wa, root, "a")
    wb = fork(store, a, tmp_path / "wb")
    (wb / "solution.py").write_text(GOOD_SOLUTION + "# b\n")
    b = snapshot(store, wb, a, "b")
    assert git(store, "rev-list", "--parents", "-n", "1", b).split()[1:] == [a]
    assert git(store, "rev-list", "--parents", "-n", "1", a).split()[1:] == [root]
    assert git(store, "rev-list", "--count", b).strip() == "3"


def test_root_workspace_is_template_copy(tmp_path: Path) -> None:
    root = create_root_workspace(tmp_path / "root")
    for rel in ("solution.py", "README.md", "tools/check.py"):
        assert (root / rel).is_file()
    assert root.stat().st_mode & 0o222 == 0
    store, commit = make_store(tmp_path / "s")
    node = fork(store, commit, tmp_path / "node")
    (node / "proposal.md").write_text("# identity\n")
    validate_outputs(node)


def _outputs(root: Path, proposal: str | None = "# p\n", solution: str | None = GOOD_SOLUTION):
    root.mkdir(parents=True, exist_ok=True)
    if proposal is not None:
        (root / "proposal.md").write_text(proposal)
    if solution is not None:
        (root / "solution.py").write_text(solution)
    return root


def test_validate_outputs_accepts_good_outputs(tmp_path: Path) -> None:
    validate_outputs(_outputs(tmp_path / "ok"))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"proposal": None}, "proposal.md"),
        ({"solution": None}, "solution.py"),
        ({"solution": "def other(policy, calibration):\n    return policy\n"}, "compress"),
        ({"solution": "def compress(policy, calibration)\n    return policy\n"}, "parse"),
        (
            {
                "solution": "def compress(policy, calibration):\n    return policy\n"
                + "#" * (1 << 20)
            },
            "exceeds",
        ),
        ({"proposal": "p" * ((256 << 10) + 1)}, "exceeds"),
    ],
)
def test_validate_outputs_rejects(tmp_path: Path, kwargs: dict, message: str) -> None:
    root = _outputs(tmp_path / "bad", **kwargs)
    with pytest.raises(WorkspaceError, match=message):
        validate_outputs(root)


def test_validate_outputs_rejects_symlink(tmp_path: Path) -> None:
    root = _outputs(tmp_path / "link", solution=None)
    os.symlink("/etc/passwd", root / "solution.py")
    with pytest.raises(WorkspaceError, match="solution.py"):
        validate_outputs(root)


def test_write_evidence_copies_every_prior_node_only(tmp_path: Path) -> None:
    nodes_dir = tmp_path / "node_src"
    full = nodes_dir / "c1"
    partial = nodes_dir / "c2"
    for node in (full, partial):
        node.mkdir(parents=True)
        (node / "proposal.md").write_text(f"proposal {node.name}\n")
        (node / "solution.py").write_text(GOOD_SOLUTION)
    (full / "score.json").write_text('{"score": 0.5}')
    (full / "error.txt").write_text("boom")
    tree = {"root": "r", "nodes": {"c1": None, "c2": "c1"}}
    evidence = tmp_path / "evidence"
    write_evidence(
        evidence,
        [
            {"cell_id": "c1", "parent_id": "r", "node_dir": full},
            {"cell_id": "c2", "parent_id": "c1", "node_dir": partial},
        ],
        tree,
    )
    assert json.loads((evidence / "tree.json").read_text()) == tree
    assert (evidence / "nodes" / "c1" / "score.json").read_text() == '{"score": 0.5}'
    assert (evidence / "nodes" / "c1" / "error.txt").read_text() == "boom"
    assert (evidence / "nodes" / "c2" / "proposal.md").read_text() == "proposal c2\n"
    assert not (evidence / "nodes" / "c2" / "score.json").exists()
    written = sorted(p.relative_to(evidence).as_posix() for p in evidence.rglob("*") if p.is_file())
    assert written == [
        "nodes/c1/error.txt",
        "nodes/c1/proposal.md",
        "nodes/c1/score.json",
        "nodes/c2/proposal.md",
        "tree.json",
    ]


def test_write_evidence_rejects_unsafe_cell_id(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError, match="unsafe"):
        write_evidence(tmp_path / "e", [{"cell_id": "../x", "node_dir": tmp_path}], {})


def _stub_evaluator(path: Path) -> Path:
    path.write_text(
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "out = args[args.index('--out') + 1]\n"
        "status = os.environ['STUB_STATUS']\n"
        "record = {'status': status}\n"
        "if status == 'ok':\n"
        "    record.update(sizes={'size_reduction': 0.0, 'resident_bytes': 10,\n"
        "        'baseline_resident_bytes': 10, 'peak_cuda_bytes': 0},\n"
        "        action_error=0.25, check={'pipeline_median_ms': 12.5, 'pipeline_p95_ms': 14.0})\n"
        "else:\n"
        "    record.update(error='Traceback: boom')\n"
        "os.makedirs(out, exist_ok=True)\n"
        "json.dump(record, open(os.path.join(out, 'result.json'), 'w'))\n"
        "print(json.dumps({'status': status}))\n"
    )
    return path


def _run_check(tmp_path: Path, status: str) -> subprocess.CompletedProcess[str]:
    store, root = make_store(tmp_path)
    ws = fork(store, root, tmp_path / "ws")
    stub = _stub_evaluator(tmp_path / "stub.py")
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        "RSI_CHECK_CMD": f"{sys.executable} {stub}",
        "RSI_GPU_SLOTS_DIR": str(tmp_path / "slots"),
        "RSI_GPU_SLOTS": "1",
        "STUB_STATUS": status,
    }
    return subprocess.run(
        [sys.executable, str(ws / "tools" / "check.py")],
        cwd=ws,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_check_tool_reports_ok(tmp_path: Path) -> None:
    done = _run_check(tmp_path, "ok")
    assert done.returncode == 0, done.stderr
    assert "status: ok" in done.stdout
    assert "memory removed: 0.0%" in done.stdout
    assert "action error" in done.stdout
    assert "median 12.50" in done.stdout


def test_check_tool_reports_solution_error(tmp_path: Path) -> None:
    done = _run_check(tmp_path, "solution_error")
    assert done.returncode == 1
    assert "status: solution_error" in done.stdout
    assert "Traceback: boom" in done.stdout
