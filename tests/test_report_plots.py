"""Report readers tolerate absent evidence and address cells by plan id under measurements/."""

import json
import shutil
import sys
from pathlib import Path

import pytest

import report.plots.__main__ as plots_main
from report.plots import data, divergence, precision, results, timeline
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
    for module in (precision, results, divergence, timeline):
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
    record = divergence.divergence_record(empty)
    assert record["counts"]["included"] == 0
    assert record["phases"]["smoke"]["spearman"]["rho"] is None
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
        "ladder": "ok",
        "divergence": "ok",
        "failure": "skipped: evidence absent",  # the fixture has no failure analysis
        "reserved": "skipped: evidence absent",  # the fixture has no reserved-500 records
        "size-timeline": "skipped: evidence absent",  # no second search in the fixture
        "latency-timeline": "skipped: evidence absent",  # no program-v4 node table
    }
    assert sorted(path.name for path in (synthetic.parent / "figures").glob("*.pdf")) == [
        "divergence.pdf",
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
    assert numbers["figures"]["ladder"] == "skipped: evidence absent"
    assert numbers["figures"]["divergence"] == "skipped: evidence absent"
    assert set(numbers["rows"]) == {"B16"}


def test_no_measurements_writes_no_numbers_file(synthetic):
    shutil.rmtree(synthetic / "measurements")
    sys.argv = ["vla-report-plots"]
    plots_main.main()
    assert not (synthetic.parent / "data" / "numbers.json").exists()


def test_divergence_pairs_screen_error_with_both_phases(synthetic):
    record = divergence.divergence_record(synthetic)
    assert record["action_error_field"] == "mean_physical_arm_change_mse"
    assert record["action_error_frames"] == 80
    assert record["counts"]["included"] == 26  # 13 recipes, smoke and development each
    assert record["phases"]["smoke"]["gate_min_successes"] == 8
    assert record["phases"]["development"]["gate_min_successes"] == 77
    cell = next(
        c
        for c in record["cells"]
        if c["cell_id"] == "precision-lang-nf4" and c["phase"] == "development"
    )
    assert cell["action_error"] == pytest.approx(2e-6)
    assert (cell["successes"], cell["trials"], cell["gate_passed"]) == (79, 100, True)
    assert cell["sources"] == [
        "measurements/precision-lang-nf4/screen/run.json",
        "measurements/precision-lang-nf4/development/run.json",
    ]


def test_divergence_ranks_but_withholds_saturated_intervals(synthetic):
    record = divergence.divergence_record(synthetic)
    smoke = record["phases"]["smoke"]
    assert smoke["spearman"]["rho"] == pytest.approx(-0.463, abs=1e-3)
    assert smoke["spearman"]["ci95"] is None  # most resamples are constant at 8/10
    assert smoke["gate_passing"]["rho"] is None
    assert smoke["gate_passing"]["n_cells"] == 12
    development = record["phases"]["development"]
    assert development["spearman"]["rho"] == pytest.approx(-0.587, abs=1e-3)
    low, high = development["spearman"]["ci95"]
    assert -1 <= low < development["spearman"]["rho"] < high <= 1
    assert development["gate_passing"]["n_cells"] == 12


def test_divergence_excludes_nodes_that_record_no_action_error(empty):
    write(
        empty / "cycles" / "iter_0" / "nodes" / "a" / "score.json",
        {"cell_id": "a", "successes": 8, "episodes": 10},
    )
    record = divergence.divergence_record(empty)
    node = next(cell for cell in record["cells"] if cell["cell_id"] == "a")
    assert node["action_error"] is None
    assert node["included"] is False
    assert "action error" in node["reason"]
    assert record["counts"]["excluded"] == 1
    assert record["phases"]["smoke"]["n_cells"] == 0


def test_build_writes_divergence_json_from_the_same_record(synthetic):
    numbers = build(synthetic)
    written = json.loads((synthetic / "divergence.json").read_text())
    assert written["study"] == synthetic.name
    assert written["phases"] == numbers["divergence"]["phases"]
    assert written["counts"] == numbers["divergence"]["counts"]
    assert not any(
        cell["sources"] and cell["sources"][0].startswith("/") for cell in written["cells"]
    )


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


def _cycles(tmp_path, scores):
    """A study's cycle ledger with `scores` per cycle: (id, score, valid)."""
    return {
        "source": str(tmp_path / "cycles"),
        "cycles": [
            {
                "cycle_id": f"iter_{i}",
                "nodes": [{"cell_id": c, "score": v, "valid": ok} for c, v, ok in cycle],
            }
            for i, cycle in enumerate(scores)
        ],
    }


def test_running_best_skips_invalid_nodes():
    points = [
        {"score": 0.5, "cycle": 0},
        {"score": 1.0, "cycle": 1, "valid": False},
        {"score": 0.4, "cycle": 1},
        {"score": 0.7, "cycle": 1},
    ]
    assert timeline.running_best(points) == [0.5, 0.5, 0.5, 0.7]
    assert timeline.jumps(points) == [0, 3]


def test_size_timeline_marks_exploits_and_keeps_them_out_of_the_best(synthetic, tmp_path):
    n = {
        "dream_rsi": _cycles(
            tmp_path / "a", [[("a", 0.6, True), ("b", 0.8, True)], [("c", 0.9, False)]]
        ),
        "dream_rsi_v2": _cycles(tmp_path / "b", [[("a", 0.5, True), ("b", 0.7, True)]]),
    }
    ledger = timeline.fig_size_timeline(n)
    assert ledger["program-v1"]["best_pct"] == 80.0
    assert ledger["program-v1"]["exploit_nodes"] == ["c"]
    assert [j["cell_id"] for j in ledger["program-v2"]["jumps"]] == ["a", "b"]
    assert (synthetic.parent / "figures" / "size-timeline.pdf").is_file()


def test_size_timeline_needs_both_searches(synthetic, tmp_path):
    only_one = {"dream_rsi": _cycles(tmp_path, [[("a", 0.6, True)]]), "dream_rsi_v2": {}}
    assert timeline.fig_size_timeline(only_one) is None


def test_latency_timeline_reads_the_node_table(synthetic, tmp_path):
    table = {
        "snapshot_utc": "2026-11-16T17:08Z",
        "cycles_complete": 0,
        "cycles_target": 3,
        "nodes": [
            {"cell_id": "n0", "cycle": 0, "score": 2.0},
            {"cell_id": "n1", "cycle": 0, "score": 0.0},
            {"cell_id": "n2", "cycle": 0, "score": 2.5},
        ],
    }
    write(tmp_path / "nodes.json", table)
    rows = {
        "B16": {"benchmark_median_ms": 70.0},
        "ladder-b16-graph-4step": {"benchmark_median_ms": 20.0},
    }
    ledger = timeline.fig_latency_timeline({"rows": rows}, tmp_path / "nodes.json")
    assert ledger["nodes"] == 3
    assert ledger["best_lower_bound"] == 2.5
    assert ledger["hand_built_speedup"] == 3.5
    assert timeline.fig_latency_timeline({"rows": rows}, tmp_path / "absent.json") is None


def test_export_nodes_keeps_scored_nodes_in_evaluation_order(tmp_path):
    from rsi.export_nodes import export

    for name, at, score in (("iter_000-b0-a0", "2", 2.7), ("iter_000-b1-a0", "1", 2.6)):
        node = tmp_path / "cycles" / "iter_000" / "nodes" / name
        write(node / "score.json", {"score": score, "successes": 8, "episodes": 10})
        write(node / "eval" / "result.json", {"environment": {"recorded_at": at}})
        (node / "proposal.md").write_text("# Proposal: CUDA graph\n")
    (tmp_path / "cycles" / "iter_000" / "nodes" / "iter_000-b2-a0").mkdir()  # still evaluating
    table = export(tmp_path)
    assert [row["cell_id"] for row in table["nodes"]] == ["iter_000-b1-a0", "iter_000-b0-a0"]
    assert table["nodes"][0]["title"] == "CUDA graph"
    assert table["nodes"][0]["smoke"] == "8/10"
