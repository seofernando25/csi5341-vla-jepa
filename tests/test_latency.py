"""Latency score and anti-gaming guards (program-v4): CPU only, no simulator, no GPU."""

import csv
import json
import time
from pathlib import Path

import pytest
import torch

from evaluation.compression import latency, solution
from rsi import expansion
from rsi.online_evidence import DISCOVERY_PROMPT, build_discovery_prompt, discovery_prompt_path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "studies/program-v4/protocol.json").read_text())
LIMITS = latency.Limits.from_protocol(PROTOCOL, "check")


def _burn(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def frames(count: int = 40) -> list[dict]:
    return [{"x": torch.full((1, 4), float(index))} for index in range(count)]


class FakePolicy:
    """A policy whose action is a function of the frame, with optional defects."""

    def __init__(
        self,
        cost_s: float = 0.0,
        hits: dict | None = None,
        miss_cost_s: float = 0.0,
        constant: bool = False,
        path_mismatch: bool = False,
        lag: bool = False,
    ) -> None:
        self.cost_s, self.miss_cost_s = cost_s, miss_cost_s
        self.hits = dict(hits or {})
        self.constant, self.path_mismatch, self.lag = constant, path_mismatch, lag
        self.last: torch.Tensor | None = None
        self.seen: list[float] = []

    def reset(self) -> None:
        return None

    def _action(self, batch: dict) -> torch.Tensor:
        if self.constant:
            return torch.zeros(1, 7)
        return torch.full((1, 7), float(batch["x"].mean()))

    def select_action(self, batch: dict) -> torch.Tensor:
        self.seen.append(float(batch["x"].mean()))
        if self.lag:
            current = self._action(batch)
            out = self.last if self.last is not None else current
            self.last = current
            return out
        _burn(self.cost_s)
        key = float(batch["x"].mean())
        if key in self.hits:
            return self.hits[key].clone()
        _burn(self.miss_cost_s)
        action = self._action(batch)
        self.hits[key] = action.clone()
        return action

    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        action = self._action(batch)
        if self.path_mismatch:
            action = action + 1.0
        return action.unsqueeze(1).repeat(1, 7, 1)


class SlowReset(FakePolicy):
    def reset(self) -> None:
        _burn(0.02)


# -- clock capture --


def test_clock_rebinds_are_detected(monkeypatch):
    assert latency.rebinds(latency.CPU_CLOCK) == []
    monkeypatch.setattr(time, "perf_counter_ns", lambda: 0)
    assert "time.perf_counter_ns was replaced during compress" in latency.rebinds(latency.CPU_CLOCK)


def test_a_patched_clock_does_not_change_the_measurement(monkeypatch):
    policy = FakePolicy(cost_s=0.002)
    monkeypatch.setattr(time, "perf_counter_ns", lambda: 0)
    timing = latency.time_calls(policy, frames(30), LIMITS, 0, latency.CPU_CLOCK)
    assert timing["median_ms"] > 0.5


# -- timing and caching --


def test_timed_calls_use_frames_never_seen_in_warmup():
    policy = FakePolicy()
    timing = latency.time_calls(
        policy, frames(40), LIMITS, 0, latency.CPU_CLOCK, first=LIMITS.equivalence_frames
    )
    assert timing["warmup"] == LIMITS.warmup_calls
    assert timing["calls"] == LIMITS.timed_calls
    warm = policy.seen[: LIMITS.warmup_calls]
    timed = policy.seen[LIMITS.warmup_calls :]
    assert not set(warm) & set(timed)
    assert not set(policy.seen) & set(range(LIMITS.equivalence_frames))
    assert max(warm) > min(timed)


def test_a_cache_of_warmup_frames_gains_nothing_on_new_frames():
    data = frames(40)
    warm = {float(item["x"].mean()): torch.zeros(1, 7) for item in data[: LIMITS.warmup_calls]}
    policy = FakePolicy(hits=warm, miss_cost_s=0.004)
    timing = latency.time_calls(policy, data, LIMITS, 0, latency.CPU_CLOCK)
    assert timing["median_ms"] >= 3.0


def test_a_cached_solution_is_caught():
    data = frames(40)
    hits = {float(item["x"].mean()): torch.zeros(1, 7) for item in data}
    policy = FakePolicy(hits=hits, miss_cost_s=0.05)
    novel = latency.novel_frame_cost(
        policy, data, median_ms=0.1, limits=LIMITS, seed=0, clock=latency.CPU_CLOCK
    )
    assert novel["violation"]
    assert novel["median_ms"] > novel["limit_ms"]


# -- equivalence --


def test_an_honest_policy_passes_the_equivalence_check():
    result = latency.equivalence(FakePolicy(), frames(20), LIMITS, 0, latency.CPU_CLOCK)
    assert result["violations"] == []
    assert result["action_spread"] > 0
    assert result["path_equivalent"]


def test_a_constant_answer_fails_the_equivalence_check():
    result = latency.equivalence(
        FakePolicy(constant=True), frames(20), LIMITS, 0, latency.CPU_CLOCK
    )
    assert "actions do not depend on the observation" in result["violations"]


def test_a_special_cased_select_action_fails_path_equivalence():
    policy = FakePolicy(path_mismatch=True)
    result = latency.equivalence(policy, frames(20), LIMITS, 0, latency.CPU_CLOCK)
    assert (
        "select_action disagrees with predict_action_chunk on the same frame"
        in result["violations"]
    )


def test_a_stale_answer_fails_order_invariance():
    result = latency.equivalence(FakePolicy(lag=True), frames(20), LIMITS, 0, latency.CPU_CLOCK)
    assert "a call on another frame changed the answer for the first frame" in result["violations"]


def test_work_moved_into_reset_is_charged():
    assert latency.reset_cost(SlowReset(), frames(8), latency.CPU_CLOCK) >= 15.0


# -- numeric guards --


def test_episode_overhead_above_the_reference_fails():
    assert latency.overhead_violation(30.0, 40.0, LIMITS) is None
    assert latency.overhead_violation(30.0, 200.0, LIMITS) is not None


def test_cpu_above_the_cap_fails():
    assert latency.cpu_violation(10.0, 15.0, LIMITS) is None
    assert latency.cpu_violation(10.0, 25.0, LIMITS) is not None


def test_child_cpu_above_the_cap_fails():
    assert latency.child_cpu_violation(0.5, LIMITS) is None
    assert latency.child_cpu_violation(2.0, LIMITS) is not None


def test_peak_memory_above_the_reference_fails():
    assert latency.peak_violation(1000, 1005, LIMITS) is None
    assert latency.peak_violation(1000, 1011, LIMITS) is not None


def test_score_is_the_speedup_when_the_gate_and_memory_pass():
    score, reason = latency.speedup(
        100.0,
        50.0,
        gate=8,
        successes=9,
        episodes=10,
        reference_peak=1000,
        candidate_peak=900,
        limits=LIMITS,
    )
    assert score == pytest.approx(2.0)
    assert reason is None


def test_score_is_zero_below_the_gate_or_above_the_memory():
    below, reason = latency.speedup(
        100.0,
        50.0,
        gate=8,
        successes=7,
        episodes=10,
        reference_peak=1000,
        candidate_peak=900,
        limits=LIMITS,
    )
    assert below == 0.0 and "smoke gate" in reason
    above, reason = latency.speedup(
        100.0,
        50.0,
        gate=8,
        successes=9,
        episodes=10,
        reference_peak=1000,
        candidate_peak=1100,
        limits=LIMITS,
    )
    assert above == 0.0 and "peak CUDA" in reason


# -- protocol and harness wiring --


def test_the_v4_protocol_scores_latency_and_keeps_the_v2_constants():
    assert PROTOCOL["score"]["kind"] == latency.SCORE_KIND
    assert PROTOCOL["score"]["baseline"] == 1.0
    v2 = json.loads((ROOT / "studies/program-v2/protocol.json").read_text())
    for field in ("checkpoint_revision", "suite", "task_ids", "seed", "phases", "gates"):
        assert PROTOCOL[field] == v2[field]
    v1 = json.loads((ROOT / "studies/program-v1/protocol.json").read_text())
    assert "score" not in v1


def test_limits_read_the_check_block_for_the_check_phase():
    check = latency.Limits.from_protocol(PROTOCOL, "check")
    full = latency.Limits.from_protocol(PROTOCOL, "smoke")
    assert check.timed_calls < full.timed_calls
    assert check.reference_tasks == (0,)
    assert full.reference_tasks == (0, 1, 2)


def test_classify_reads_the_latency_score():
    raw = {
        "status": "ok",
        "successes": 9,
        "episodes": 10,
        "sizes": {"size_reduction": None, "resident_bytes": 1, "baseline_resident_bytes": 2},
        "score": 1.4,
    }
    verdict = expansion.classify(raw, 8)
    assert verdict.fail_class == "ok"
    assert verdict.score == pytest.approx(1.4)
    assert verdict.size_reduction is None


def test_classify_keeps_the_memory_score():
    raw = {"status": "ok", "successes": 9, "episodes": 10, "sizes": {"size_reduction": 0.5}}
    assert expansion.classify(raw, 8).score == pytest.approx(0.5)


def test_classify_maps_a_guard_violation_to_implementation_failure():
    verdict = expansion.classify({"status": "guard_violation", "error": "clock"}, 8)
    assert verdict.fail_class == expansion.IMPLEMENTATION_FAILURE
    assert verdict.score == 0.0


def test_the_v4_prompt_states_the_score_and_guards_without_methods():
    path = discovery_prompt_path(ROOT / "studies/program-v4")
    assert path.name == "discovery.md" and path.parent.name == "program-v4"
    prompt = build_discovery_prompt(
        cell_id="c-b0-a0",
        cycle_id="c",
        parent_id="root",
        smoke_trials=10,
        smoke_gate=8,
        best_score=1.0,
        gpu_slots=1,
        minutes=30,
        prompt_path=path,
    )
    flat = " ".join(prompt.split())
    assert "reference median latency / your median latency" in flat
    assert "device synchronize" in flat
    for guard in (
        "Returning a stored answer",
        "Background or asynchronous work",
        "Moving work to the CPU",
        "Peak GPU memory above the reference's",
    ):
        assert guard in flat
    for method in ("CUDA graph", "torch.compile", "Triton", "kernel fusion", "fuse"):
        assert method.lower() not in flat.lower()


def test_the_check_tool_prints_the_latency_score():
    import importlib.util

    path = ROOT / "rsi/workspace_template/tools/check.py"
    spec = importlib.util.spec_from_file_location("check_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = module._summary(
        {
            "status": "ok",
            "latency": {
                "score": 1.75,
                "score_kind": "latency_speedup",
                "reference": {"timing": {"median_ms": 100.0}},
                "candidate": {"median_ms": 57.1, "p95_ms": 60.0},
                "violations": ["a guard"],
                "zero_reason": None,
            },
        }
    )
    assert any("score: 1.7500" in line for line in lines)
    assert any("reference median 100.00" in line for line in lines)
    assert any("guard: a guard" in line for line in lines)


def test_a_study_without_a_prompt_uses_the_shared_one():
    assert discovery_prompt_path(None) == DISCOVERY_PROMPT
    assert discovery_prompt_path(ROOT / "studies/program-v2") == DISCOVERY_PROMPT


def _fake_reference() -> latency.Reference:
    return latency.Reference(
        limits=LIMITS,
        frames=frames(24),
        timing={"median_ms": 100.0, "cpu_ms_per_call": 10.0},
        episode_wall_seconds=1.0,
        overhead_per_step_ms=10.0,
        peak_cuda_bytes=1000,
        steps=10,
        successes=1,
        episodes=1,
    )


def _fake_candidate(out_dir: Path, violations: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "success"])
        writer.writeheader()
        writer.writerow({"task_id": 0, "success": 1})
    return {
        "score": 2.0,
        "score_kind": latency.SCORE_KIND,
        "zero_reason": None,
        "violations": violations,
        "reference": {},
        "candidate": {},
        "equivalence": {},
        "successes": 1,
        "episodes": 1,
        "rollout": {"episodes": 1, "successes": 1, "task_ids": [0], "phase": "development"},
    }


@pytest.fixture
def cpu_evaluator(monkeypatch, tmp_path):
    monkeypatch.setattr(solution, "REQUIRED_DEVICE", "cpu")
    monkeypatch.setattr(solution.calib, "setup", lambda: None)
    monkeypatch.setattr(solution.calib, "load", lambda: {"bank_indices": []})
    monkeypatch.setattr(
        solution, "load_policy", lambda variant: (FakePolicy(), {"variant": variant})
    )
    monkeypatch.setattr(solution, "Pipeline", lambda policy: None)
    monkeypatch.setattr(latency, "run_reference", lambda *args, **kwargs: _fake_reference())
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(scratch))
    return tmp_path


def _workspace(root: Path, source: str | None = None) -> Path:
    path = root / "ws"
    path.mkdir()
    text = source or "def compress(policy, calibration):\n    return policy\n"
    (path / "solution.py").write_text(text)
    return path


def test_the_evaluator_records_the_latency_score(cpu_evaluator, monkeypatch):
    monkeypatch.setattr(
        latency, "run_candidate", lambda *args, **kwargs: _fake_candidate(args[3], [])
    )
    result = solution.run_solution(
        _workspace(cpu_evaluator),
        "check",
        cpu_evaluator / "out",
        time_limit_s=30.0,
        protocol=ROOT / "studies/program-v4/protocol.json",
    )
    assert result["status"] == "ok", result.get("error")
    assert result["score"] == pytest.approx(2.0)
    assert result["latency"]["score_kind"] == latency.SCORE_KIND
    assert result["sizes"]["size_reduction"] is not None


def test_the_evaluator_fails_a_guard_violation(cpu_evaluator, monkeypatch):
    monkeypatch.setattr(
        latency, "run_candidate", lambda *args, **kwargs: _fake_candidate(args[3], ["clock"])
    )
    result = solution.run_solution(
        _workspace(cpu_evaluator),
        "check",
        cpu_evaluator / "out",
        time_limit_s=30.0,
        protocol=ROOT / "studies/program-v4/protocol.json",
    )
    assert result["status"] == "guard_violation"
    assert result["score"] == 0.0
    assert "clock" in result["error"]


def test_the_smoke_phase_passes_the_protocol_gate_and_trials(cpu_evaluator, monkeypatch):
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return _fake_candidate(args[3], [])

    monkeypatch.setattr(latency, "run_candidate", fake)
    result = solution.run_solution(
        _workspace(cpu_evaluator),
        "smoke",
        cpu_evaluator / "out",
        time_limit_s=30.0,
        protocol=ROOT / "studies/program-v4/protocol.json",
    )
    assert result["status"] == "ok", result.get("error")
    assert seen["gate"] == PROTOCOL["gates"]["smoke_min_successes"]
    assert seen["tasks"] == PROTOCOL["task_ids"]
    assert seen["trials"] == PROTOCOL["phases"]["smoke"]["trials"] // len(PROTOCOL["task_ids"])


def test_the_latency_protocol_allows_a_compiler_cache_in_the_temp_dir(cpu_evaluator, monkeypatch):
    monkeypatch.setattr(
        latency, "run_candidate", lambda *args, **kwargs: _fake_candidate(args[3], [])
    )
    source = """
import pathlib
import tempfile


def compress(policy, calibration):
    cache = pathlib.Path(tempfile.gettempdir(), "torchinductor_cache")
    cache.mkdir(exist_ok=True)
    (cache / "kernel.py").write_text("x")
    return policy
"""
    result = solution.run_solution(
        _workspace(cpu_evaluator, source),
        "check",
        cpu_evaluator / "out",
        time_limit_s=30.0,
        protocol=ROOT / "studies/program-v4/protocol.json",
    )
    assert result["status"] == "ok", result.get("error")
