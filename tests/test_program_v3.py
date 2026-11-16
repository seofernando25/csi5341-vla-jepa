"""Program v3 hides the compression code and the prompt token ids from discovery agents.

The two exclusions are protocol fields, so these tests read the committed protocol and check
that the code enforces what it says: the agent's sandbox cannot reach `evaluation/compression`
or `evaluation/nf4.py`, the calibration a solution receives carries no `token_ids`, and the
workspace's check tool does not import the compression package it can no longer read.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from evaluation.compression import calibration as calib
from rsi.sandbox import (
    REPO_ROOT,
    discovery_visibility,
    repo_source_binds,
    run_sandboxed_python,
)
from rsi.workspace import commit_root, create_root_workspace, fork, init_store

V2 = REPO_ROOT / "studies" / "program-v2"
V3 = REPO_ROOT / "studies" / "program-v3"
PROMPT = REPO_ROOT / "rsi" / "prompts" / "discovery.md"
TEMPLATE_README = REPO_ROOT / "rsi" / "workspace_template" / "README.md"
COMPRESSION = "evaluation/compression"
NF4 = "evaluation/nf4.py"
FASTPATH = "evaluation/fastpath.py"
VENDOR = "outputs/compression/vendor"
HIDDEN = (COMPRESSION, NF4, FASTPATH, VENDOR)
SHARED = (
    "checkpoint",
    "checkpoint_revision",
    "suite",
    "task_ids",
    "seed",
    "final_seed_offset",
    "development_states_per_task",
    "reserved_states_per_task",
    "max_steps",
    "action_chunk",
    "flow_steps",
    "dtype",
    "batch_size",
    "phases",
    "gates",
    "statistics",
    "reserved_lock",
    "starting_policy",
    "dream_rsi",
)
requires_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")


def _protocol(study: Path) -> dict:
    return json.loads((study / "protocol.json").read_text(encoding="utf-8"))


def _fake_repo(root: Path, protocol: dict) -> tuple[Path, Path]:
    """A minimal repo with the paths `repo_source_binds` lists and one study protocol."""
    for rel in ("evaluation/compression", "evaluation/models.py", "outputs/compression"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / "evaluation/compression/solution.py").write_text("", encoding="utf-8")
    (root / "evaluation/nf4.py").write_text("", encoding="utf-8")
    (root / "outputs/compression/calibration.pt").write_text("", encoding="utf-8")
    (root / "outputs/compression/calibration-v3.pt").write_text("", encoding="utf-8")
    study = root / "studies" / "program-v3"
    study.mkdir(parents=True)
    (study / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    return root, study


def test_program_v3_keeps_program_v2_protocol_except_the_two_exclusions() -> None:
    v2, v3 = _protocol(V2), _protocol(V3)
    for field in SHARED:
        assert v3[field] == v2[field], f"{field} differs between program-v2 and program-v3"
    assert v3["protocol"] == "program-v3-debiased"
    assert "program-v2" in v3["supersedes"]
    assert v3["revisions"] == []
    assert set(v3) - set(v2) == {"calibration", "discovery_visibility"}
    assert {k: v for k, v in v3["run_budget"].items() if k != "note"} == {
        k: v for k, v in v2["run_budget"].items() if k != "note"
    }
    assert {rule["field"] for rule in v3["starting_rules"]} - {
        rule["field"] for rule in v2["starting_rules"]
    } == {
        "discovery_visibility.repo_exclude",
        "discovery_visibility.workspace_check",
        "calibration.exclude_fields",
    }


def test_discovery_visibility_is_read_from_the_protocol() -> None:
    v3 = discovery_visibility(V3)
    assert v3.repo_exclude == HIDDEN
    assert v3.workspace_check == "self_contained"
    v2 = discovery_visibility(V2)
    assert v2.repo_exclude == ()
    assert v2.workspace_check == "evaluator"


def test_agent_source_binds_hide_the_compression_code() -> None:
    binds = repo_source_binds(study_dir=V3, exclude=discovery_visibility(V3).repo_exclude)
    targets = set(binds.values())
    for hidden in HIDDEN:
        assert f"/tmp/repo/{hidden}" not in targets
    assert not [dst for dst in targets if dst.startswith(f"/tmp/repo/{COMPRESSION}/")]
    assert not [dst for dst in targets if dst.startswith(f"/tmp/repo/{VENDOR}/")]
    assert "/tmp/repo/evaluation/models.py" in targets
    assert "/tmp/repo/evaluation/pipeline.py" in targets


def test_repo_source_binds_drop_an_excluded_directory_whole(tmp_path: Path) -> None:
    """A listed directory in `exclude` goes, children included; a listed file's siblings stay."""
    repo, study = _fake_repo(tmp_path, {})
    (repo / VENDOR / "hqq").mkdir(parents=True)
    (repo / VENDOR / "hqq" / "quant.py").write_text("", encoding="utf-8")
    binds = repo_source_binds(repo, study_dir=study, exclude=(VENDOR,))
    assert not [dst for dst in binds.values() if "vendor" in dst]
    trimmed = repo_source_binds(repo, study_dir=study, exclude=(NF4,))
    assert "/tmp/repo/evaluation/nf4.py" not in set(trimmed.values())
    assert "/tmp/repo/evaluation/models.py" in set(trimmed.values())


def test_evaluator_source_binds_keep_the_compression_code() -> None:
    """The fixed evaluator runs in its own sandbox and still reads the whole package."""
    binds = repo_source_binds(study_dir=V3)
    # The whole package is one bind, so nf4.py and the compression code come with it.
    assert binds[(REPO_ROOT / "evaluation").resolve()] == "/tmp/repo/evaluation"


def test_repo_source_binds_swap_in_the_studys_calibration_artifact(tmp_path: Path) -> None:
    protocol = {
        "calibration": {"artifact": "outputs/compression/calibration-v3.pt"},
    }
    repo, study = _fake_repo(tmp_path, protocol)
    binds = repo_source_binds(repo, study_dir=study)
    targets = set(binds.values())
    assert "/tmp/repo/outputs/compression/calibration-v3.pt" in targets
    assert "/tmp/repo/outputs/compression/calibration.pt" not in targets
    plain = repo_source_binds(repo, study_dir=None)
    assert "/tmp/repo/outputs/compression/calibration.pt" in set(plain.values())


def test_for_protocol_strips_the_excluded_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "cal.pt"
    torch.save({"linear_inputs": {}, "bank_indices": [0], "token_ids": torch.arange(3)}, artifact)
    excluded = {"calibration": {"artifact": "cal.pt", "exclude_fields": ["token_ids"]}}
    assert set(calib.for_protocol(excluded, tmp_path)) == {"linear_inputs", "bank_indices"}
    kept = {"calibration": {"artifact": "cal.pt"}}
    assert "token_ids" in calib.for_protocol(kept, tmp_path)


def test_program_v3_calibration_carries_no_token_ids() -> None:
    protocol = _protocol(V3)
    artifact = REPO_ROOT / protocol["calibration"]["artifact"]
    if not artifact.is_file():
        pytest.skip(f"{protocol['calibration']['artifact']} is a runtime artifact")
    data = calib.for_protocol(protocol, REPO_ROOT)
    assert "token_ids" not in data
    assert {"linear_inputs", "block_influence", "bank_indices"} <= set(data)


def test_workspace_check_does_not_import_the_compression_package(tmp_path: Path) -> None:
    workspace = create_root_workspace(tmp_path / "ws", "self_contained")
    check = (workspace / "tools" / "check.py").read_text(encoding="utf-8")
    assert "evaluation.compression" not in check
    assert (workspace / "tools" / "resident.py").is_file()
    assert (workspace / "tools" / "memory.py").is_file()
    assert not (workspace / "tools" / "check_selfcontained.py").exists()
    evaluator = create_root_workspace(tmp_path / "plain", "evaluator")
    assert "evaluation.compression.study" in (evaluator / "tools" / "check.py").read_text()
    assert not (evaluator / "tools" / "resident.py").exists()


def test_root_commit_carries_the_protocols_check_tool(tmp_path: Path) -> None:
    store = init_store(tmp_path / "cycle")
    commit = commit_root(store, "self_contained")
    checkout = fork(store, commit, tmp_path / "checkout")
    check = (checkout / "tools" / "check.py").read_text(encoding="utf-8")
    assert "evaluation.compression" not in check
    assert (checkout / "tools" / "resident.py").is_file()


def test_self_contained_check_summarises_without_the_evaluator(tmp_path: Path) -> None:
    workspace = create_root_workspace(tmp_path / "ws", "self_contained")
    spec = importlib.util.spec_from_file_location("check_tool", workspace / "tools" / "check.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        lines = module._summary(
            50, 100, [{"latency_ms": 1.0, "difference": 0.25, "finite": True}], 2
        )
    finally:
        sys.path.remove(str(workspace / "tools"))
        sys.modules.pop("check_tool", None)
        sys.modules.pop("resident", None)
        sys.modules.pop("memory", None)
    assert lines[0] == "status: ok"
    assert "memory removed: 50.0%" in lines[1]
    assert "0.25" in lines[2]


@requires_bwrap
def test_agent_sandbox_cannot_read_the_compression_code(tmp_path: Path) -> None:
    probe = (
        "import importlib.util, json, os\n"
        "paths = ['/tmp/repo/evaluation/compression', "
        "'/tmp/repo/evaluation/compression/solution.py', '/tmp/repo/evaluation/nf4.py', "
        "'/tmp/repo/evaluation/models.py', '/tmp/repo/outputs/compression/vendor']\n"
        "found = {path: os.path.exists(path) for path in paths}\n"
        "found['spec'] = importlib.util.find_spec('evaluation.compression') is not None\n"
        "print(json.dumps(found))\n"
    )
    binds = repo_source_binds(study_dir=V3, exclude=discovery_visibility(V3).repo_exclude)
    done = run_sandboxed_python(
        ["-c", probe], workspace_dir=tmp_path / "ws", source_ro=binds, timeout=120.0
    )
    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert found["/tmp/repo/evaluation/compression"] is False
    assert found["/tmp/repo/evaluation/compression/solution.py"] is False
    assert found["/tmp/repo/evaluation/nf4.py"] is False
    assert found["spec"] is False
    assert found["/tmp/repo/evaluation/models.py"] is True
    assert found["/tmp/repo/outputs/compression/vendor"] is False


def test_prompt_and_workspace_readme_name_no_hint() -> None:
    for path in (PROMPT, TEMPLATE_README):
        text = path.read_text(encoding="utf-8")
        assert "token_ids" not in text, path
        assert "evaluation/compression" not in text, path
    prompt = PROMPT.read_text(encoding="utf-8")
    assert "fraction of memory removed" in prompt
    assert "{smoke_gate}" in prompt
