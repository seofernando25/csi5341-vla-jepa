"""Fixed evaluator for agent-written solutions: sizes, gaming attempts, failures; CPU only."""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from evaluation.compression import solution, study
from evaluation.compression.inventory import ResidentWalk
from evaluation.compression.memory import live_payloads

BASE = (256 * 256 + 256) * 4  # fp32 Linear(256, 256) weight and bias
HALF = 256 * 256 * 2 + 256 * 4  # fp16 weight, fp32 bias
TINY = 4
PROTOCOL = json.loads(solution.PROTOCOL_PATH.read_text())


class FakePolicy(nn.Module):
    """Stands in for the VLA policy: one Linear and the two inference methods."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(256, 256)

    def reset(self) -> None:
        return None

    def select_action(self, batch: object) -> torch.Tensor:
        return torch.zeros(1, 7)

    def predict_action_chunk(self, batch: object) -> torch.Tensor:
        return torch.zeros(1, 7, 7)


@pytest.fixture
def cpu_env(monkeypatch, tmp_path):
    monkeypatch.setattr(solution, "REQUIRED_DEVICE", "cpu")
    monkeypatch.setattr(solution.calib, "setup", lambda: None)
    monkeypatch.setattr(solution.calib, "load", lambda: {"bank_indices": []})
    monkeypatch.setattr(
        solution, "load_policy", lambda variant: (FakePolicy(), {"variant": variant})
    )
    monkeypatch.setattr(solution, "Pipeline", lambda policy: None)
    monkeypatch.setattr(study, "screen", fake_screen)
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    return tmp_path


def workspace(root: Path, source: str, **extra: str) -> Path:
    path = root / "ws"
    path.mkdir(exist_ok=True)
    (path / "solution.py").write_text(source)
    for name, text in extra.items():
        (path / f"{name}.py").write_text(text)
    return path


def run(root: Path, source: str, phase: str = "check", limit: float = 60.0, **extra: str) -> dict:
    result = solution.run_solution(
        workspace(root, source, **extra), phase, root / "out", time_limit_s=limit
    )
    assert json.loads((root / "out" / "result.json").read_text())["status"] == result["status"]
    assert (root / "out" / "log.txt").exists()
    return result


def fake_screen(policy, calibration, predictions=80, compiled=False):
    keys = {key: 1.0 for key in solution.CHECK_KEYS}
    keys.update(maximum_normalized_action_difference=0.25, diagnostic_observations=80)
    return keys, []


def truth(held: int) -> float:
    return max(0.0, 1 - held / BASE)


def test_identity_is_zero_reduction(cpu_env):
    result = run(cpu_env, "def compress(policy, calibration):\n    return policy\n")
    assert result["status"] == "ok", result.get("error")
    assert result["sizes"]["size_reduction"] == pytest.approx(0.0, abs=1e-6)
    assert result["baseline_resident_bytes"] == BASE


def test_half_precision_is_half_reduction(cpu_env):
    result = run(cpu_env, "def compress(policy, calibration):\n    return policy.half()\n")
    assert result["status"] == "ok"
    assert result["sizes"]["size_reduction"] == pytest.approx(
        truth(256 * 256 * 2 + 256 * 2), abs=1e-6
    )


GAMING = {
    "list_attribute": """
import torch
def compress(policy, calibration):
    policy.stash = [policy.lin.weight.detach().half().clone()]
    policy.lin.weight = torch.nn.Parameter(torch.zeros(1))
    return policy
""",
    "module_global": """
import torch
STASH = []
def compress(policy, calibration):
    STASH.append(policy.lin.weight.detach().half().clone())
    policy.lin.weight = torch.nn.Parameter(torch.zeros(1))
    return policy
""",
    "shared_view_real_buffer": """
import torch
SHARED = torch.zeros(1)
def compress(policy, calibration):
    policy.lin.register_buffer("real", policy.lin.weight.detach().half().clone(), persistent=False)
    policy.lin.weight = torch.nn.Parameter(SHARED.view(1))
    return policy
""",
    "numpy_attribute": """
import numpy as np
import torch
def compress(policy, calibration):
    policy.packed = np.zeros((256, 256), dtype=np.float16)
    policy.lin.weight = torch.nn.Parameter(torch.zeros(1))
    return policy
""",
}


@pytest.mark.parametrize("case", sorted(GAMING))
def test_gaming_is_counted(cpu_env, case):
    result = run(cpu_env, GAMING[case])
    assert result["status"] == "ok"
    assert result["sizes"]["size_reduction"] == pytest.approx(truth(HALF + TINY), abs=1e-6)


def test_helper_module_state_is_counted(cpu_env):
    helper = "import torch\nSTORE = []\ndef stash(t):\n    STORE.append(t)\n"
    source = """
import torch
import vla_gaming_helper as helper
def compress(policy, calibration):
    helper.stash(policy.lin.weight.detach().half().clone())
    policy.lin.weight = torch.nn.Parameter(torch.zeros(1))
    return policy
"""
    result = run(cpu_env, source, vla_gaming_helper=helper)
    assert result["status"] == "ok"
    assert result["sizes"]["size_reduction"] == pytest.approx(truth(HALF + TINY), abs=1e-6)


def test_file_written_to_temp_dir_is_a_violation(cpu_env):
    source = """
import os, tempfile
import torch
def compress(policy, calibration):
    torch.save(policy.lin.weight.detach().half().clone(), os.path.join(tempfile.gettempdir(), "w.pt"))
    policy.lin.weight = torch.nn.Parameter(torch.zeros(1))
    return policy
"""
    result = run(cpu_env, source)
    assert result["status"] == "size_violation"
    assert "created or changed" in result["error"]


def test_workspace_file_over_one_mib_is_charged(cpu_env):
    source = """
import pathlib
import torch
PACKED = pathlib.Path(__file__).with_name("packed.bin")
PACKED.write_bytes(b"x" * (2 << 20))
def compress(policy, calibration):
    policy.lin.weight = torch.nn.Parameter(torch.zeros(1))
    return policy
"""
    result = run(cpu_env, source)
    assert result["status"] == "ok"
    assert result["sizes"]["workspace_file_bytes"] == 2 << 20
    assert result["sizes"]["resident_bytes"] == (2 << 20) + TINY + 256 * 4
    assert result["sizes"]["size_reduction"] == 0.0


def test_small_workspace_files_are_not_charged(cpu_env):
    source = "def compress(policy, calibration):\n    return policy\n"
    result = run(cpu_env, source, helper="X = 1\n")
    assert result["status"] == "ok"
    assert result["sizes"]["workspace_file_bytes"] == 0
    assert result["sizes"]["size_reduction"] == pytest.approx(0.0, abs=1e-6)


def test_walk_counts_cuda_requirement_against_cpu_tensor():
    walk = ResidentWalk("cuda")
    walk.add(FakePolicy())
    assert any("cpu tensor" in v for v in walk.violations)


def test_walk_counts_meta_tensor_as_violation():
    walk = ResidentWalk("cpu")
    walk.add({"w": torch.empty(4, device="meta")})
    assert walk.violations == ["meta tensor"]


@pytest.mark.parametrize(
    "source, status, message",
    [
        (
            "def compress(policy, calibration):\n    raise RuntimeError('boom')\n",
            "solution_error",
            "boom",
        ),
        (
            "def compress(policy, calibration):\n    return None\n",
            "solution_error",
            "compress must return",
        ),
        (
            "def compress(policy, calibration):\n    return 3\n",
            "solution_error",
            "compress must return",
        ),
        ("def other(policy, calibration):\n    return policy\n", "solution_error", "no compress"),
    ],
)
def test_solution_failures_are_recorded(cpu_env, source, status, message):
    result = run(cpu_env, source)
    assert result["status"] == status
    assert message in result["error"]
    assert (cpu_env / "out" / "result.json").exists()


def test_sleeping_past_the_limit_is_recorded(cpu_env):
    source = (
        "import time\ndef compress(policy, calibration):\n    time.sleep(10)\n    return policy\n"
    )
    result = run(cpu_env, source, limit=1.0)
    assert result["status"] == "solution_error"
    assert "TimeoutError" in result["error"]


def test_workspace_hash_tracks_content_only(tmp_path):
    ws = workspace(tmp_path, "def compress(policy, calibration):\n    return policy\n")
    (ws / "helper.py").write_text("X = 1\n")
    first, large = solution.workspace_hash(ws)
    assert large == []
    assert solution.workspace_hash(ws)[0] == first
    (ws / "helper.py").write_text("X = 2\n")
    changed = solution.workspace_hash(ws)[0]
    assert changed != first
    (ws / "helper.py").write_text("X = 1\n")
    assert solution.workspace_hash(ws)[0] == first


def test_pycache_is_excluded_from_hash(tmp_path):
    ws = workspace(tmp_path, "def compress(policy, calibration):\n    return policy\n")
    before = solution.workspace_hash(ws)[0]
    (ws / "__pycache__").mkdir()
    (ws / "__pycache__" / "solution.cpython.pyc").write_bytes(b"x")
    assert solution.workspace_hash(ws)[0] == before


def test_check_phase_reports_action_error(cpu_env):
    result = run(cpu_env, "def compress(policy, calibration):\n    return policy\n", phase="check")
    assert result["status"] == "ok"
    assert result["action_error"] == pytest.approx(0.25)
    assert set(solution.CHECK_KEYS) <= set(result["check"])


def test_rollout_phase_derives_episodes_from_protocol(cpu_env, monkeypatch):
    seen = {}

    def fake_rollout(variant, tasks, phase, episodes, policy, pipeline, folder):
        seen.update(tasks=tasks, phase=phase, episodes=episodes)
        with (folder / "episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["task_id", "success"])
            writer.writeheader()
            for task in tasks:
                writer.writerow({"task_id": task, "success": 1})
        return {"episodes": len(tasks), "successes": len(tasks), "task_ids": tasks}

    monkeypatch.setattr(solution, "rollout", fake_rollout)
    result = run(cpu_env, "def compress(policy, calibration):\n    return policy\n", phase="smoke")
    assert result["status"] == "ok"
    assert seen["episodes"] == PROTOCOL["phases"]["smoke"]["trials"] // len(PROTOCOL["task_ids"])
    assert result["episodes"] == len(PROTOCOL["task_ids"])
    assert result["rollout"]["per_task"]["0"] == {"successes": 1, "episodes": 1}


def test_final_phase_is_rejected_by_cli(monkeypatch, tmp_path):
    ws = workspace(tmp_path, "def compress(policy, calibration):\n    return policy\n")
    argv = [
        "vla-compress",
        "solution",
        "--workspace",
        str(ws),
        "--phase",
        "final",
        "--out",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        study.main()
    assert exit_info.value.code == 2


def test_numpy_module_walk_counts_nbytes():
    walk = ResidentWalk("cpu")
    walk.add({"a": np.zeros(10, dtype=np.float32)})
    assert walk.nbytes == 40
    assert os.path.basename(solution.PROTOCOL_PATH) == "protocol.json"


# Weights parked outside the returned policy, as cycle-3 nodes iter_002-b1-a3,
# iter_002-b2-a1 and iter_002-b2-a2 did on the `torch` module.
FOREIGN_STASH = """
import torch

def compress(policy, calibration):
    setattr(torch, "_vla_jepa_test_stash", [policy.lin.weight.detach().clone()])
    policy.lin.weight = torch.nn.Parameter(torch.zeros(1))
    return policy
"""

DEFERRED_RESTORE = """
import torch

def compress(policy, calibration):
    setattr(torch, "_vla_jepa_test_stash", [policy.lin.weight.detach().clone()])
    policy.lin.weight = torch.nn.Parameter(torch.zeros(1))
    original = policy.select_action

    def select_action(batch):
        pending = getattr(torch, "_vla_jepa_test_stash", None)
        if pending:
            setattr(torch, "_vla_jepa_test_stash", None)
            policy.lin.weight.data = pending[0]
        return original(batch)

    policy.select_action = select_action
    return policy
"""

FOREIGN_BLOB = """
import torch

def compress(policy, calibration):
    blob = policy.lin.weight.detach().numpy().tobytes()
    setattr(torch, "_vla_jepa_test_blob", blob)
    policy.lin.weight = torch.nn.Parameter(torch.zeros(1))
    return policy
"""


@pytest.fixture(autouse=True)
def clear_foreign_stashes():
    yield
    for name in ("_vla_jepa_test_stash", "_vla_jepa_test_blob"):
        if hasattr(torch, name):
            delattr(torch, name)


def screen_with_inference(policy, calibration, predictions=80, compiled=False):
    policy.select_action(None)
    return fake_screen(policy, calibration, predictions, compiled)


def test_weights_parked_on_a_foreign_module_are_charged(cpu_env):
    result = run(cpu_env, FOREIGN_STASH)
    assert result["status"] == "ok"
    assert result["sizes"]["resident_bytes"] == BASE + TINY
    assert result["sizes"]["size_reduction"] == 0.0


def test_restore_at_the_first_forward_is_charged(cpu_env, monkeypatch):
    monkeypatch.setattr(study, "screen", screen_with_inference)
    result = run(cpu_env, DEFERRED_RESTORE)
    assert result["status"] == "ok"
    sizes = result["sizes"]
    assert sizes["resident_before_inference_bytes"] == BASE + TINY
    assert sizes["resident_after_inference_bytes"] == BASE
    assert sizes["resident_bytes"] == BASE + TINY
    assert sizes["size_reduction"] == 0.0


def test_buffer_parked_on_a_foreign_module_is_charged(cpu_env):
    result = run(cpu_env, FOREIGN_BLOB)
    assert result["status"] == "ok"
    assert result["sizes"]["resident_bytes"] == TINY + 256 * 4 + 256 * 256 * 4
    assert result["sizes"]["size_reduction"] == 0.0


def test_payload_present_before_compress_is_not_charged(cpu_env):
    torch._vla_jepa_test_blob = torch.zeros(1024)
    result = run(cpu_env, "def compress(policy, calibration):\n    return policy\n")
    assert result["status"] == "ok"
    assert result["sizes"]["resident_bytes"] == BASE
    assert result["sizes"]["size_reduction"] == pytest.approx(0.0, abs=1e-6)


def test_measure_merges_new_payloads_without_double_counting(monkeypatch):
    monkeypatch.setattr(solution, "REQUIRED_DEVICE", "cpu")
    skip = frozenset(live_payloads())
    torch._vla_jepa_test_blob = torch.zeros(1024)
    sized = solution.measure(FakePolicy(), (), skip)
    assert sized["resident_bytes"] == BASE + 4096
    assert sized["violations"] == []


def test_peak_cuda_figure_dominates_the_size_record(tmp_path):
    record = {"stage_seconds": {}}
    run_ = solution._Run(tmp_path, "check", tmp_path, record, 1.0)
    run_.baseline_bytes = 1000
    sizes = run_.sizes(before=100, after=200, peak=900, file_bytes=0)
    assert sizes["resident_bytes"] == 900
    assert sizes["size_reduction"] == pytest.approx(0.1)


def test_discovery_prompt_states_that_hiding_memory_fails():
    from rsi.online_evidence import build_discovery_prompt

    prompt = build_discovery_prompt(
        cell_id="c-b0-a0",
        cycle_id="c",
        parent_id="root",
        smoke_trials=10,
        smoke_gate=8,
        best_score=0.0,
        gpu_slots=2,
        minutes=30,
    )
    flat = " ".join(prompt.split())
    assert "Hiding or deferring memory makes the node fail" in flat
    assert "peak CUDA storage" in flat
