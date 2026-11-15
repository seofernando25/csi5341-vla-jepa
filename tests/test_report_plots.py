"""Report readers tolerate absent evidence and address cells by plan id under measurements/."""

import json
import shutil
import sys
from pathlib import Path

import pytest

import report.plots.__main__ as plots_main
from report.plots import data, divergence, precision, results
from report.plots.cycles import dream_rsi_cycles

FIXTURE = Path(__file__).parent / "fixtures" / "program-v1-synthetic"


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


@pytest.fixture
def empty(tmp_path, monkeypatch):
    """Point the readers at an absent studies/program-v1."""
    root = tmp_path / "program-v1"
    monkeypatch.setattr(data, "STUDY", root)
    return root


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """A writable copy of the synthetic fixture, wired into every reader and figure writer."""
    study = tmp_path / "program-v1"
    shutil.copytree(FIXTURE, study)
    figures, numbers = tmp_path / "figures", tmp_path / "data"
    figures.mkdir()
    numbers.mkdir()
    shutil.copy(data.DATA / "parameter_census.json", numbers / "parameter_census.json")
    monkeypatch.setattr(data, "STUDY", study)
    monkeypatch.setattr(data, "DATA", numbers)
    monkeypatch.setattr(plots_main, "STUDY", study)
    monkeypatch.setattr(plots_main, "DATA", numbers)
    monkeypatch.setattr(plots_main, "OUT", figures)
    for module in (precision, results, divergence):
        monkeypatch.setattr(module, "OUT", figures)
    return study


def build(synthetic):
    """Run the report build the way `vla-report-plots` does."""
    sys.argv = ["vla-report-plots"]
    plots_main.main()
    return json.loads((synthetic.parent / "data" / "numbers.json").read_text())


def test_missing_study_directory_is_not_fatal(empty):
    collected = data.collect()
    assert collected["status"] == "missing"
    assert collected["what"] == "clean-room measurements"
    assert data.plan()["status"] == "missing"
    assert data.divergence().empty
    cycles = dream_rsi_cycles(empty)
    assert cycles["status"] == "missing"
    assert cycles["count"] == 0


def test_cells_are_addressed_by_plan_id(synthetic):
    collected = data.collect()
    assert {"B16", "precision-lang-nf4", "ladder-nf4-graph-1step-packed"} <= set(collected["rows"])
    row = collected["rows"]["precision-lang-nf4"]
    assert row["source"].endswith("measurements/precision-lang-nf4")
    assert row["development_successes"] == 79
    assert row["development_wilson_pct"][0] < 79 < row["development_wilson_pct"][1]


def test_effective_bits_come_from_per_module_storage(synthetic):
    rows = data.collect()["rows"]
    assert rows["B16"]["effective_bits"]["text"] == pytest.approx(16.0)
    assert rows["precision-lang-nf4"]["effective_bits"]["text"] == pytest.approx(4.16)
    assert rows["precision-lang-nf4"]["effective_bits"]["vision"] == pytest.approx(16.0)


def test_every_figure_builds_from_the_fixture(synthetic):
    numbers = build(synthetic)
    assert numbers["figures"] == {
        "precision": "ok",
        "format": "ok",
        "ladder": "ok",
        "divergence": "ok",
    }
    assert sorted(path.name for path in (synthetic.parent / "figures").glob("*.pdf")) == [
        "divergence.pdf",
        "format.pdf",
        "ladder.pdf",
        "precision.pdf",
    ]


def test_numbers_json_carries_no_pre_cleanroom_row(synthetic):
    numbers = build(synthetic)
    assert set(numbers["rows"]) == {path.stem for path in (synthetic / "recipes").glob("*.json")}
    assert not {"B16c", "C1", "C2", "Q8", "Q4", "S500"} & set(numbers["rows"])
    assert "launch_profile" not in numbers
    assert numbers["latency_ladder"]["ladder-nf4-graph-1step-packed"]["median_ms"] == 31.6


def test_missing_evidence_skips_a_figure_and_records_it(synthetic):
    for folder in (synthetic / "measurements").iterdir():
        if folder.name != "B16":
            shutil.rmtree(folder)
    numbers = build(synthetic)
    assert numbers["figures"]["precision"] == "skipped: evidence absent"
    assert numbers["figures"]["format"] == "skipped: evidence absent"
    assert numbers["figures"]["ladder"] == "skipped: evidence absent"
    assert numbers["figures"]["divergence"] == "skipped: evidence absent"
    assert set(numbers["rows"]) == {"B16"}


def test_no_measurements_writes_no_numbers_file(synthetic):
    shutil.rmtree(synthetic / "measurements")
    sys.argv = ["vla-report-plots"]
    plots_main.main()
    assert not (synthetic.parent / "data" / "numbers.json").exists()


def test_divergence_reads_screen_and_smoke_cells(synthetic):
    frame = data.divergence()
    assert len(frame) == 13
    assert set(frame.component) == {"language", "vision", "action", "joint", "other"}
    assert frame.arm_mse.max() > 1e-3


def test_reserved_pair_needs_both_arms(empty):
    assert data.reserved_pair("B16") is None


def test_empty_cycle_directory_returns_empty_shape(empty):
    (empty / "cycles").mkdir(parents=True)
    out = dream_rsi_cycles(empty)
    assert out["status"] == "empty"
    assert out["cycles"] == []


def test_one_cycle_reads_tree_manifest_scores_and_proposals(empty):
    cycle_dir = empty / "cycles" / "iter_0"
    write(
        cycle_dir / "tree.json",
        {"cycle_id": "iter_0", "nodes": [{"cell_id": "a", "parent_id": "root", "score": 0.5}]},
    )
    write(
        cycle_dir / "live_cycle_manifest.json",
        {
            "cycle_id": "iter_0",
            "iteration": 0,
            "best_score": 0.5,
            "probes_used": 2,
            "world_nodes": 1,
            "policy_path": "rsi/policies/incumbent.py",
            "recorded_utc": "2026-11-15T00:00:00Z",
        },
    )
    write(
        cycle_dir / "nodes" / "a" / "score.json",
        {"cell_id": "a", "fail_class": "ok", "score": 0.5, "successes": 8, "episodes": 10},
    )
    (cycle_dir / "nodes" / "a" / "proposal.md").write_text("# proposal\n")

    out = dream_rsi_cycles(empty)
    assert out["status"] == "ok"
    assert out["count"] == 1
    cycle = out["cycles"][0]
    assert (cycle["status"], cycle["world_nodes"], len(cycle["sources"])) == ("done", 1, 2)
    node = cycle["nodes"][0]
    assert (node["successes"], node["episodes"]) == (8, 10)
    assert node["proposal"].endswith("nodes/a/proposal.md")
