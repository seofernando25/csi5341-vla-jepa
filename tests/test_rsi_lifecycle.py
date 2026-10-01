"""Study lifecycle: fresh exploration, promotions and failure recovery."""

from pathlib import Path

from rsi.core import observation
from rsi.novelty import compact_node
from rsi.process import ProcessTimeout
from rsi.runner import (
    ROOT,
    Runner,
    classify_external_failure,
    load_config,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO / "rsi/config.json")


def runner_at(tmp_path):
    runner = Runner(REPO, tmp_path / "state", synthetic=True)
    runner.initialize(REPO / "rsi/config.json")
    runner.verify()
    return runner


def test_fresh_study_does_not_import_old_results(tmp_path):
    runner = runner_at(tmp_path)
    assert CONFIG["protocol_version"] == "dream-rsi"
    assert runner.ledger() == []
    assert runner.root_references() == []
    assert runner.anchor_metadata() == []
    assert not runner.events("attempt")


def test_first_batch_starts_from_clean_root(tmp_path):
    runner = runner_at(tmp_path)
    root = dict(ROOT, score=-0.75, status="ok")
    obs = observation([compact_node(root)], CONFIG["K1"], CONFIG)
    plan = runner.plan_online_batch(0, lambda o, s: None, obs, 4)
    assert plan["source_anchors"] == [None] * 4
    assert plan["planned_actions"] == ["root"] * 4


def test_near_root_candidate_gets_matched_long_promotion(tmp_path):
    runner = runner_at(tmp_path)
    runner.ensure_baseline()
    plan = {
        "planned_actions": ["root"],
        "source_anchors": [None],
        "modes": ["novel"],
        "substitutions": [],
        "constrained_actions": 0,
    }
    runner.execute_batch(0, plan)
    assert runner.events("promotion_baseline")[0]["metrics"]["optimizer_steps"] == 1500
    promotion = runner.events("promotion")[0]
    assert promotion["status"] == "ok"
    assert promotion["metrics"]["optimizer_steps"] == 1500
    assert promotion["baseline_score"] == -0.70
    assert promotion["delta_vs_baseline"] > 0
    tree = runner.tree(0)
    assert tree[1]["promotion"]["status"] == "ok"


def test_runtime_failure_is_not_architecture_failure(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "stderr.log").write_text("CUDA error: unspecified launch failure")
    (logs / "stdout.log").write_text("")
    assert classify_external_failure(RuntimeError("subprocess exit 1"), logs) == "runtime_failure"

    (logs / "stderr.log").write_text(
        "RuntimeError: value cannot be converted to type c10::BFloat16 without overflow"
    )
    assert (
        classify_external_failure(RuntimeError("subprocess exit 1"), logs)
        == "implementation_failure"
    )
    assert (
        classify_external_failure(ProcessTimeout("subprocess timeout"), logs) == "runtime_failure"
    )


def test_runtime_and_interruptions_do_not_spend_research_budget(tmp_path):
    runner = runner_at(tmp_path)
    runner.verify()
    for index, status in enumerate(
        ("runtime_failure", "interrupted", "implementation_failure", "ok"), 1
    ):
        event = runner.journal.append(
            "attempt",
            attempt=f"n{index:04d}",
            outer=0,
            parent="root",
            source_anchor=None,
        )
        runner.finish(event, status, status)
    assert runner.research_attempt_count() == 2
    assert runner.runtime_failure_count() == 1


def test_interrupted_promotion_baseline_can_be_re_reserved(tmp_path):
    runner = runner_at(tmp_path)
    runner.journal.append(
        "promotion_baseline_started",
        reservation=1,
        steps=CONFIG["promotion_steps"],
    )
    runner.recover()
    failures = runner.events("promotion_baseline_failure")
    assert failures[-1]["reservation"] == 1
    assert failures[-1]["status"] == "interrupted"
    metrics = runner.ensure_promotion_baseline()
    assert metrics["optimizer_steps"] == CONFIG["promotion_steps"]
    assert runner.events("promotion_baseline")[-1]["reservation"] == 2


def test_promotion_runtime_failure_counts_toward_runtime_safety_cap(tmp_path):
    runner = runner_at(tmp_path)
    runner.journal.append(
        "promotion",
        attempt="n0001",
        outer=0,
        status="runtime_failure",
        metrics={},
        baseline_score=-0.7,
        delta_vs_baseline=None,
        failure_class="runtime_failure",
    )
    runner.journal.append(
        "promotion_baseline_failure",
        reservation=1,
        status="runtime_failure",
        summary="cuda failed",
    )
    assert runner.runtime_failure_count() == 2
    assert runner.promotion_research_count() == 0


def test_extend_and_resume_preserves_history_and_parent_candidates(tmp_path, monkeypatch):
    from rsi.core import atomic

    config = dict(
        CONFIG,
        max_outer_iterations=1,
        max_real_attempts=1,
        min_outer_iterations_before_stop=1,
        min_total_measured_nodes_before_stop=1,
    )
    path = tmp_path / "config.json"
    atomic(path, config)
    runner = Runner(REPO, tmp_path / "state", synthetic=True)
    runner.initialize(path)

    # Exercise the runner lifecycle without invoking 100 policy subprocesses per cycle.
    def improve(outer, current):
        runner.journal.append("cycle_done", outer=outer, selected=current, trajectories=100)
        return current

    monkeypatch.setattr(runner, "improve", improve)
    runner.run()
    before = runner.events()
    assert runner.status()["paused"] == "global_attempt_cap"
    runner.run(resume=True)
    assert runner.events() == before
    runner.extend_budget(1)
    runner.run(resume=True)
    assert runner.events()[: len(before)] == before
    ids = [e["attempt"] for e in runner.events("attempt")]
    assert len(ids) == len(set(ids)) and len(ids) > 1
    assert runner.events("initialized")[0]["prior_history"] == []
    assert runner.tree(1)[1]["id"] == ids[0]
    assert len(runner.events("baseline")) == 1
    assert len(runner.events("cycle_done")) == 2


def test_stop_resume_does_not_repeat_completed_attempt(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    runner.attempt(0, "root")
    first = runner.events("outcome")[0]
    runner.journal.append("started")
    runner.stop.touch()
    original = runner.execute_batch

    def one_batch(outer, plan):
        original(outer, plan)
        runner.stop.touch()

    monkeypatch.setattr(runner, "execute_batch", one_batch)
    runner.run(resume=True)
    assert runner.events("outcome")[0] == first
    assert len([e for e in runner.events("attempt") if e["attempt"] == first["attempt"]]) == 1


def test_interrupted_initial_baseline_gets_new_reservation(tmp_path):
    runner = runner_at(tmp_path)
    runner.journal.append("baseline_started", reservation=1)
    runner.ensure_baseline()
    assert runner.events("baseline_interrupted")[0]["reservation"] == 1
    assert runner.events("baseline")[0]["reservation"] == 2
    runner.ensure_baseline()
    assert len(runner.events("baseline")) == 1


def test_export_keeps_failed_scores_null_and_preserves_identity(tmp_path):
    import json

    import pytest

    from rsi.export import export_results

    runner = runner_at(tmp_path)
    event = runner.journal.append("attempt", attempt="n0001", outer=0, parent="root")
    runner.finish(event, "runtime_failure", "CUDA failed")
    target = tmp_path / "export"
    export_results(runner, target)
    record = json.loads((target / "attempts.jsonl").read_text())
    assert record["score"] is None
    assert record["status"] == "runtime_failure"
    other = Runner(REPO, tmp_path / "other", synthetic=True)
    other.initialize(REPO / "rsi/config.json")
    with pytest.raises(ValueError, match="another study"):
        export_results(other, target)
