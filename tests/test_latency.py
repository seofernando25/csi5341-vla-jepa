"""Latency score and anti-gaming guards (program-v4): CPU only, no simulator, no GPU."""

import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluation.compression import latency, latency_timing, solution
from evaluation.compression.latency_timing import CPU_CLOCK
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
    assert latency.rebinds(CPU_CLOCK) == []
    monkeypatch.setattr(time, "perf_counter_ns", lambda: 0)
    assert "time.perf_counter_ns was replaced during compress" in latency.rebinds(CPU_CLOCK)


def test_a_patched_clock_does_not_change_the_measurement(monkeypatch):
    policy = FakePolicy(cost_s=0.002)
    monkeypatch.setattr(time, "perf_counter_ns", lambda: 0)
    timing = latency_timing.time_calls(policy, frames(30), LIMITS, 0, CPU_CLOCK)
    assert timing["median_ms"] > 0.5


# -- timing and caching --


def test_timed_calls_use_frames_never_seen_in_warmup():
    policy = FakePolicy()
    timing = latency_timing.time_calls(
        policy, frames(40), LIMITS, 0, CPU_CLOCK, first=LIMITS.equivalence_frames
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
    timing = latency_timing.time_calls(policy, data, LIMITS, 0, CPU_CLOCK)
    assert timing["median_ms"] >= 3.0


def test_a_cached_solution_is_caught():
    data = frames(40)
    hits = {float(item["x"].mean()): torch.zeros(1, 7) for item in data}
    policy = FakePolicy(hits=hits, miss_cost_s=0.05)
    novel = latency.novel_frame_cost(
        policy, data, median_ms=0.1, limits=LIMITS, seed=0, clock=CPU_CLOCK
    )
    assert novel["violation"]
    assert novel["median_ms"] > novel["limit_ms"]


# -- equivalence --


def test_an_honest_policy_passes_the_equivalence_check():
    result = latency.equivalence(FakePolicy(), frames(20), LIMITS, 0, CPU_CLOCK)
    assert result["violations"] == []
    assert result["action_spread"] > 0
    assert result["path_equivalent"]


def test_a_constant_answer_fails_the_equivalence_check():
    result = latency.equivalence(FakePolicy(constant=True), frames(20), LIMITS, 0, CPU_CLOCK)
    assert "actions do not depend on the observation" in result["violations"]


def test_a_special_cased_select_action_fails_path_equivalence():
    policy = FakePolicy(path_mismatch=True)
    result = latency.equivalence(policy, frames(20), LIMITS, 0, CPU_CLOCK)
    assert (
        "select_action disagrees with predict_action_chunk on the same frame"
        in result["violations"]
    )


def test_a_stale_answer_fails_order_invariance():
    result = latency.equivalence(FakePolicy(lag=True), frames(20), LIMITS, 0, CPU_CLOCK)
    assert "a call on another frame changed the answer for the first frame" in result["violations"]


def test_work_moved_into_reset_is_charged():
    assert latency.reset_cost(SlowReset(), frames(8), CPU_CLOCK) >= 15.0


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


def test_score_is_the_lower_bound_when_the_gate_and_memory_pass():
    score, reason = latency.speedup(
        {"point": 2.0, "low": 1.8, "high": 2.2},
        gate=8,
        successes=9,
        episodes=10,
        reference_peak=1000,
        candidate_peak=900,
        limits=LIMITS,
    )
    assert score == pytest.approx(1.8)
    assert reason is None


def test_score_is_zero_below_the_gate_or_above_the_memory():
    below, reason = latency.speedup(
        {"point": 2.0, "low": 1.8, "high": 2.2},
        gate=8,
        successes=7,
        episodes=10,
        reference_peak=1000,
        candidate_peak=900,
        limits=LIMITS,
    )
    assert below == 0.0 and "smoke gate" in reason
    above, reason = latency.speedup(
        {"point": 2.0, "low": 1.8, "high": 2.2},
        gate=8,
        successes=9,
        episodes=10,
        reference_peak=1000,
        candidate_peak=1100,
        limits=LIMITS,
    )
    assert above == 0.0 and "peak CUDA" in reason


# -- block timing and the bootstrap interval --


class FakeWatch:
    """Stands in for the GPU sampler so block timing tests never shell out."""

    def __init__(self) -> None:
        self.samples = 0

    def sample(self) -> None:
        self.samples += 1


def test_the_frame_plan_has_disjoint_blocks_and_warmup():
    warm, blocks = latency.frame_plan(48, LIMITS, LIMITS.equivalence_frames)
    assert len(blocks) == LIMITS.blocks
    assert all(len(block) == latency_timing.calls_per_block(LIMITS) for block in blocks)
    timed = [index for block in blocks for index in block]
    assert len(set(timed)) == len(timed)
    assert not set(warm) & set(timed)
    assert min(timed) >= LIMITS.equivalence_frames
    assert len(warm) == LIMITS.warmup_calls


def test_the_interval_lower_bound_is_below_the_point_estimate():
    reference = [[100.0 + index for index in range(4)] for _ in range(LIMITS.blocks)]
    candidate = [[50.0 + index for index in range(4)] for _ in range(LIMITS.blocks)]
    interval = latency.bootstrap_speedup(reference, candidate, LIMITS)
    assert interval["point"] == pytest.approx(2.0, rel=0.02)
    assert interval["low"] <= interval["point"] <= interval["high"]
    assert interval["blocks"] == LIMITS.blocks


def test_a_burst_in_one_candidate_block_lowers_the_lower_bound():
    reference = [[100.0] * 4 for _ in range(LIMITS.blocks)]
    quiet = [[50.0] * 4 for _ in range(LIMITS.blocks)]
    burst = [[50.0] * 4 for _ in range(LIMITS.blocks - 1)] + [[200.0] * 4]
    quiet_interval = latency.bootstrap_speedup(reference, quiet, LIMITS)
    burst_interval = latency.bootstrap_speedup(reference, burst, LIMITS)
    # The pooled median absorbs one bad block; the interval is what pays for it.
    assert burst_interval["point"] == pytest.approx(quiet_interval["point"])
    assert burst_interval["low"] < quiet_interval["low"]


def test_a_burst_in_one_block_of_both_sides_does_not_create_a_win():
    reference = [[100.0] * 4 for _ in range(LIMITS.blocks)]
    candidate = [[100.0] * 4 for _ in range(LIMITS.blocks)]
    reference[2] = [300.0] * 4
    candidate[2] = [300.0] * 4
    interval = latency.bootstrap_speedup(reference, candidate, LIMITS)
    assert interval["point"] == pytest.approx(1.0)
    assert interval["low"] <= 1.0


def test_interleaved_blocks_alternate_reference_and_candidate():
    data = frames(48)
    _, blocks = latency.frame_plan(len(data), LIMITS, 0)
    watch = FakeWatch()
    reference_blocks, candidate_blocks, results = latency._interleaved(
        FakePolicy(cost_s=0.002),
        FakePolicy(cost_s=0.001),
        data,
        blocks,
        0,
        CPU_CLOCK,
        watch,
    )
    assert len(reference_blocks) == len(candidate_blocks) == LIMITS.blocks
    assert watch.samples == LIMITS.blocks
    assert len(results) == LIMITS.blocks


def test_a_second_reference_is_not_loaded_when_it_does_not_fit(monkeypatch):
    monkeypatch.setattr(latency.torch.cuda, "mem_get_info", lambda: (1000, 10**9))
    loaded = []
    assert latency._load_second(lambda: loaded.append(True), 5000, latency.CLOCK) is None
    assert loaded == []


def test_a_second_reference_is_loaded_when_it_fits(monkeypatch):
    monkeypatch.setattr(latency.torch.cuda, "mem_get_info", lambda: (10**9, 10**9))
    sentinel = object()
    assert latency._load_second(lambda: sentinel, 5000, latency.CLOCK) is sentinel


# -- answer caching: the candidate must be the policy --


class IdentityPost:
    """The post-processor is the identity in these tests; the metric is the raw arm chunk."""

    def post(self, value):
        return value


class NearestCache:
    """A lookup table keyed on the observation: returns the nearest stored action."""

    def __init__(self, data):
        self.keys = [float(item["x"].mean()) for item in data]
        self.values = [torch.full((1, 7), key) for key in self.keys]

    def reset(self):
        return None

    def predict_action_chunk(self, batch):
        key = float(batch["x"].mean())
        nearest = min(range(len(self.keys)), key=lambda index: abs(self.keys[index] - key))
        return self.values[nearest].unsqueeze(1).repeat(1, 7, 1)

    def select_action(self, batch):
        return self.predict_action_chunk(batch)[:, 0]


class Quantized(FakePolicy):
    """A policy whose action is rounded to a grid, like a 4-bit action head."""

    def __init__(self, step: float = 0.02) -> None:
        super().__init__()
        self.step = step

    def _action(self, batch):
        value = float(batch["x"].mean())
        return torch.full((1, 7), round(value / self.step) * self.step)


def _fresh(count: int = 8, offset: float = 0.5) -> list[dict]:
    return [{"x": torch.full((1, 4), offset + index)} for index in range(count)]


def _probe(scale: float = 0.1, spread: float = 0.01):
    originals = [{"x": torch.full((1, 4), float(index))} for index in range(8)]
    perturbed = [{"x": item["x"] + scale + spread * index} for index, item in enumerate(originals)]
    return originals, perturbed


def test_the_identity_passes_the_arm_check():
    fresh = _fresh()
    reference = latency.arm_actions(FakePolicy(), fresh, len(fresh), 0, IdentityPost(), CPU_CLOCK)
    candidate = latency.arm_actions(FakePolicy(), fresh, len(fresh), 0, IdentityPost(), CPU_CLOCK)
    assert latency.arm_error(reference, candidate) == 0.0


def test_a_nearest_neighbour_cache_fails_the_arm_check():
    cache = NearestCache(frames(40))
    fresh = _fresh()
    reference = latency.arm_actions(FakePolicy(), fresh, len(fresh), 0, IdentityPost(), CPU_CLOCK)
    candidate = latency.arm_actions(cache, fresh, len(fresh), 0, IdentityPost(), CPU_CLOCK)
    assert latency.arm_error(reference, candidate) > LIMITS.arm_action_mse_max


def test_a_quantized_policy_passes_the_arm_check():
    fresh = _fresh()
    reference = latency.arm_actions(FakePolicy(), fresh, len(fresh), 0, IdentityPost(), CPU_CLOCK)
    candidate = latency.arm_actions(Quantized(), fresh, len(fresh), 0, IdentityPost(), CPU_CLOCK)
    assert latency.arm_error(reference, candidate) < LIMITS.arm_action_mse_max


def test_the_identity_passes_the_perturbation_check():
    originals, perturbed = _probe()
    reference = latency.perturbation_deltas(
        FakePolicy(), originals, perturbed, 0, IdentityPost(), CPU_CLOCK
    )
    candidate = latency.perturbation_deltas(
        FakePolicy(), originals, perturbed, 0, IdentityPost(), CPU_CLOCK
    )
    assert latency.perturbation_correlation(reference, candidate) == pytest.approx(1.0)


def test_a_quantized_policy_passes_the_perturbation_check():
    originals, perturbed = _probe()
    reference = latency.perturbation_deltas(
        FakePolicy(), originals, perturbed, 0, IdentityPost(), CPU_CLOCK
    )
    candidate = latency.perturbation_deltas(
        Quantized(), originals, perturbed, 0, IdentityPost(), CPU_CLOCK
    )
    assert (
        latency.perturbation_correlation(reference, candidate)
        >= LIMITS.perturbation_min_correlation
    )


def test_a_lookup_table_fails_the_perturbation_check():
    cache = NearestCache(frames(40))
    originals, perturbed = _probe()
    reference = latency.perturbation_deltas(
        FakePolicy(), originals, perturbed, 0, IdentityPost(), CPU_CLOCK
    )
    candidate = latency.perturbation_deltas(
        cache, originals, perturbed, 0, IdentityPost(), CPU_CLOCK
    )
    assert latency.perturbation_correlation(reference, candidate) == 0.0


# -- shared GPU --


def test_the_gpu_watch_counts_only_other_processes():
    text = f"# gpu pid type sm mem\n    0 {os.getpid()} C 99 10\n    0 4242 C 30 10\n"
    assert latency_timing._pmon_other(text, {os.getpid()}) == pytest.approx(30.0)
    apps = f"{os.getpid()}, 5000\n4242, 12000\n"
    assert latency_timing._apps_other(apps, {os.getpid()}) == pytest.approx(12000.0)


def test_another_process_above_the_level_fails_the_node(monkeypatch):
    def fake_run(command, **kwargs):
        if "pmon" in command:
            return SimpleNamespace(stdout="# gpu pid type sm mem\n 0 4242 C 40 10\n")
        return SimpleNamespace(stdout="4242, 12000\n")

    monkeypatch.setattr(latency_timing.subprocess, "run", fake_run)
    watch = latency_timing.GpuWatch(25.0)
    watch.sample()
    assert watch.violation() is not None
    assert watch.summary()["other_utilisation_max_pct"] == pytest.approx(40.0)


def test_a_quiet_gpu_passes_the_watch(monkeypatch):
    def fake_run(command, **kwargs):
        if "pmon" in command:
            return SimpleNamespace(stdout="# gpu pid type sm mem\n 0 4242 C 0 10\n")
        return SimpleNamespace(stdout="4242, 0\n")

    monkeypatch.setattr(latency_timing.subprocess, "run", fake_run)
    watch = latency_timing.GpuWatch(25.0)
    watch.sample()
    assert watch.violation() is None


# -- sandwich --


def _sandwich_report(reference: latency.Reference) -> dict:
    return {
        "score": 1.0,
        "zero_reason": None,
        "violations": [],
        "gate": 0,
        "reference": reference.as_record(),
        "candidate": {
            "peak_cuda_bytes": 1000,
            "mode": "sandwich",
            "sandwich_pending": True,
            "candidate_blocks": [[50.0] * 4 for _ in range(LIMITS.blocks)],
        },
        "successes": 1,
        "episodes": 1,
    }


def _sandwich_reference() -> latency.Reference:
    reference = _fake_reference()
    reference.blocks = [[100.0] * 4 for _ in range(LIMITS.blocks)]
    reference.block_frames = [[index] for index in range(LIMITS.blocks)]
    reference.warm = []
    reference.watch = None
    return reference


def test_the_sandwich_adds_the_after_blocks(monkeypatch):
    reference = _sandwich_reference()
    monkeypatch.setattr(
        latency,
        "time_block",
        lambda *args, **kwargs: {
            "wall": [100.0] * 4,
            "gpu": [],
            "cpu_seconds": 0.0,
            "child_seconds": 0.0,
        },
    )
    monkeypatch.setattr(latency, "warm_up", lambda *args, **kwargs: None)
    finished = latency.finish_sandwich(
        _sandwich_report(reference), reference, PROTOCOL, lambda: object(), CPU_CLOCK
    )
    assert finished["candidate"]["reference_blocks_after"] == LIMITS.blocks
    assert finished["candidate"]["speedup"]["blocks"] == 2 * LIMITS.blocks
    assert finished["score"] == pytest.approx(2.0)


def test_the_sandwich_without_a_second_reference_fails_the_node():
    reference = _sandwich_reference()
    finished = latency.finish_sandwich(
        _sandwich_report(reference), reference, PROTOCOL, None, CPU_CLOCK
    )
    assert finished["candidate"]["reference_blocks_after"] == 0
    assert finished["score"] == 0.0
    assert any("sandwich measurement is incomplete" in item for item in finished["violations"])


def test_the_sandwich_fails_when_the_reference_drifts(monkeypatch):
    reference = _sandwich_reference()

    def drifting(*args, **kwargs):
        return {"wall": [200.0] * 4, "gpu": [], "cpu_seconds": 0.0, "child_seconds": 0.0}

    monkeypatch.setattr(latency, "time_block", drifting)
    monkeypatch.setattr(latency, "warm_up", lambda *args, **kwargs: None)
    finished = latency.finish_sandwich(
        _sandwich_report(reference), reference, PROTOCOL, lambda: object(), CPU_CLOCK
    )
    assert finished["candidate"]["sandwich_drift"] == pytest.approx(1.0)
    assert finished["score"] == 0.0
    assert any("the reference moved" in item for item in finished["violations"])


def test_the_interleaved_path_has_no_after_blocks():
    reference = _sandwich_reference()
    report = _sandwich_report(reference)
    report["candidate"]["mode"] = "interleaved"
    report["candidate"]["sandwich_pending"] = False
    assert latency.finish_sandwich(report, reference, PROTOCOL, None, CPU_CLOCK) is report


# -- protocol and harness wiring --


def test_the_v4_protocol_scores_latency_and_keeps_the_v2_constants():
    assert PROTOCOL["score"]["kind"] == latency.SCORE_KIND
    assert PROTOCOL["score"]["baseline"] == 1.0
    v2 = json.loads((ROOT / "studies/program-v2/protocol.json").read_text())
    for field in ("checkpoint_revision", "suite", "task_ids", "seed", "phases", "gates"):
        assert PROTOCOL[field] == v2[field]
    v1 = json.loads((ROOT / "studies/program-v1/protocol.json").read_text())
    assert "score" not in v1


def test_the_v4_protocol_carries_the_block_timing_and_the_new_guards():
    timing = PROTOCOL["score"]["timing"]
    assert timing["blocks"] >= 6
    assert timing["timed_calls"] % timing["blocks"] == 0
    assert timing["bootstrap_resamples"] >= 1000
    assert timing["check"]["blocks"] >= 6
    assert timing["check"]["timed_calls"] % timing["check"]["blocks"] == 0
    guards = PROTOCOL["score"]["guards"]
    assert guards["arm_action_mse_max"] == pytest.approx(0.01)
    assert guards["perturbation_min_correlation"] > 0
    assert guards["other_gpu_utilisation_max_pct"] > 0
    assert "lower bound" in PROTOCOL["score"]["formula"]
    assert any("guard" in line for line in PROTOCOL["score"]["zero_unless"])


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
    assert "lower bound of the 95 % interval" in flat
    assert "device synchronize" in flat
    for guard in (
        "Returning a stored answer",
        "Not being the policy",
        "Background or asynchronous work",
        "Moving work to the CPU",
        "Peak GPU memory above the reference's",
        "Another process using more than a set share of the GPU",
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
                "candidate": {
                    "median_ms": 57.1,
                    "p95_ms": 60.0,
                    "mode": "interleaved",
                    "speedup": {"point": 1.81, "low": 1.75, "high": 1.9, "blocks": 6},
                    "arm_action_error": 0.0004,
                    "arm_action_error_limit": 0.01,
                    "perturbation_correlation": 0.98,
                    "perturbation_correlation_min": 0.5,
                    "gpu_watch": {"samples": 12, "other_utilisation_max_pct": 4.0},
                },
                "violations": ["a guard"],
                "zero_reason": None,
            },
        }
    )
    assert any("score: 1.7500" in line for line in lines)
    assert any("reference median 100.00" in line for line in lines)
    assert any("95% [1.7500, 1.9000]" in line for line in lines)
    assert any("arm action error: 0.0004" in line for line in lines)
    assert any("perturbation correlation: 0.980" in line for line in lines)
    assert any("other GPU use: max 4.0 %" in line for line in lines)
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
