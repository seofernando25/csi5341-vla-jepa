"""Report readers tolerate absent clean-room evidence and read program-v1 records."""

import json
from pathlib import Path

import pandas as pd
import pytest

from report.plots import data


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


@pytest.fixture
def study(tmp_path, monkeypatch):
    """Point the readers at an empty temporary studies/program-v1."""
    root = tmp_path / "program-v1"
    monkeypatch.setattr(data, "STUDY", root)
    monkeypatch.setattr(data, "EVAL", tmp_path / "evaluation")
    return root


def test_missing_study_directory_is_not_fatal(study):
    collected = data.collect()
    assert collected["status"] == "missing"
    assert collected["what"] == "compression summary"
    cycles = data.dream_rsi_cycles()
    assert cycles["status"] == "missing"
    assert cycles["cycles"] == []
    assert cycles["count"] == 0
    assert data.divergence().empty
    assert data.launch_profile()["status"] == "missing"


def test_empty_cycle_directory_returns_empty_shape(study):
    (study / "cycles").mkdir(parents=True)
    out = data.dream_rsi_cycles()
    assert out["status"] == "empty"
    assert out["count"] == 0
    assert out["cycles"] == []


def test_one_cycle_reads_tree_manifest_scores_and_proposals(study):
    cycle_dir = study / "cycles" / "iter_0"
    write(
        cycle_dir / "tree.json",
        {
            "cycle_id": "iter_0",
            "nodes": [
                {"cell_id": "a", "parent_id": "root", "fail_class": "ok", "score": 0.5},
            ],
        },
    )
    write(
        cycle_dir / "live_cycle_manifest.json",
        {
            "cycle_id": "iter_0",
            "iteration": 0,
            "best_score": 0.5,
            "probes_used": 2,
            "decision_rounds": 1,
            "world_nodes": 1,
            "policy_path": "rsi/policies/incumbent.py",
            "recorded_utc": "2026-11-15T00:00:00Z",
        },
    )
    write(
        cycle_dir / "nodes" / "a" / "score.json",
        {
            "cell_id": "a",
            "parent_id": "root",
            "fail_class": "ok",
            "score": 0.5,
            "successes": 8,
            "episodes": 10,
        },
    )
    (cycle_dir / "nodes" / "a" / "proposal.md").write_text("# proposal\n")

    out = data.dream_rsi_cycles()
    assert out["status"] == "ok"
    assert out["count"] == 1
    cycle = out["cycles"][0]
    assert cycle["cycle_id"] == "iter_0"
    assert cycle["status"] == "done"
    assert cycle["best_score"] == 0.5
    assert cycle["world_nodes"] == 1
    assert len(cycle["sources"]) == 2
    node = cycle["nodes"][0]
    assert node["cell_id"] == "a"
    assert (node["successes"], node["episodes"]) == (8, 10)
    assert node["proposal_bytes"] == len("# proposal\n")
    assert node["proposal"].endswith("nodes/a/proposal.md")


def test_cycle_without_manifest_is_incomplete(study):
    cycle_dir = study / "cycles" / "iter_1"
    write(cycle_dir / "tree.json", {"cycle_id": "iter_1", "nodes": []})
    cycle = data.dream_rsi_cycles()["cycles"][0]
    assert cycle["status"] == "incomplete"
    assert cycle["world_nodes"] == 0
    assert "best_score" not in cycle


def test_collect_reads_program_v1_summary(study):
    study.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate": "precision-lang-int8",
                "family": "precision",
                "smoke_successes": 9,
                "development_successes": 80,
                "final_successes": 400,
                "final_episodes": 500,
                "benchmark_pipeline_median_ms": 30.0,
                "benchmark_pipeline_p95_ms": 31.0,
                "benchmark_peak_allocated_bytes": 4e9,
                "packed_archive_bytes": 3e9,
            },
            {"candidate": "precision-vision-nf4", "family": "precision"},
        ]
    ).to_csv(study / "summary.csv", index=False)

    out = data.collect()
    assert out["status"] == "ok"
    assert set(out["rows"]) == {"precision-lang-int8"}  # untested recipe is skipped
    row = out["rows"]["precision-lang-int8"]
    assert (row["successes"], row["trials"]) == (400, 500)
    assert row["peak_allocated_gb"] == pytest.approx(4.0)
    assert row["weights_gb"] == pytest.approx(3.0)
    assert out["latency_ladder"]["precision-lang-int8"]["median_ms"] == 30.0
