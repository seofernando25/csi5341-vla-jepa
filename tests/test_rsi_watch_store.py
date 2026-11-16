"""Store, reducer and selector behaviour, executed under node.

The page's logic lives in ES modules. `tests/js/store_harness.mjs` imports them by absolute
file URL, runs every transition listed in docs/watch_state.md and prints one JSON report; this
file turns each report entry into a pytest assertion. No view module is imported and no DOM is
needed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "rsi" / "watch" / "static"
HARNESS = Path(__file__).resolve().parent / "js" / "store_harness.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _run_harness(tmp_path: Path) -> dict:
    env = {key: value for key, value in os.environ.items() if key != "NODE_OPTIONS"}
    try:
        result = subprocess.run(
            ["node", str(HARNESS), str(STATIC)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"node did not finish the store harness within 30 s: {HARNESS}")
    except OSError as exc:
        pytest.fail(f"node could not run the store harness {HARNESS}: {exc}")
    detail = (
        f"node exited {result.returncode} for {HARNESS}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0, detail
    assert result.stdout.strip(), f"node printed no report\n{detail}"
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"the store harness printed invalid JSON: {exc}\n{detail}")
    assert isinstance(report.get("checks"), dict), f"the report has no checks\n{detail}"
    return report


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return _run_harness(tmp_path_factory.mktemp("watch-store"))


def _check(report: dict, name: str) -> None:
    entry = report["checks"].get(name)
    assert entry is not None, f"the harness did not report {name}"
    assert entry["ok"], f"{name}: {entry['detail']}"


def test_harness_reports_every_check(report: dict) -> None:
    expected = {
        "loopLive",
        "nodePhase",
        "stateLoadedFreezes",
        "doneFreezes",
        "helloAfterDone",
        "eventsAppend",
        "reset",
        "transcriptLoaded",
        "tabs",
        "view",
        "treeRows",
        "transcriptBlocks",
        "filters",
        "scores",
        "nextSource",
        "defaultSource",
        "artifactGuard",
        "purity",
        "store",
    }
    assert set(report["checks"]) == expected


def test_loop_live_requires_running_and_status(report: dict) -> None:
    _check(report, "loopLive")


def test_node_phase_demotes_when_the_loop_is_not_live(report: dict) -> None:
    _check(report, "nodePhase")


def test_state_loaded_freezes_an_open_transport(report: dict) -> None:
    _check(report, "stateLoadedFreezes")


def test_done_and_closed_freeze_the_transport(report: dict) -> None:
    _check(report, "doneFreezes")


def test_hello_after_done_does_not_thaw_or_clear(report: dict) -> None:
    _check(report, "helloAfterDone")


def test_events_append_only_for_the_attached_source(report: dict) -> None:
    _check(report, "eventsAppend")


def test_reset_clears_events_only_when_not_frozen(report: dict) -> None:
    _check(report, "reset")


def test_transcript_loaded_keeps_the_longer_list(report: dict) -> None:
    _check(report, "transcriptLoaded")


def test_tabs_open_focus_close_and_cap(report: dict) -> None:
    _check(report, "tabs")


def test_view_is_validated_per_kind(report: dict) -> None:
    _check(report, "view")


def test_tree_rows_root_first_with_orphans(report: dict) -> None:
    _check(report, "treeRows")


def test_transcript_blocks_group_deltas_and_tools(report: dict) -> None:
    _check(report, "transcriptBlocks")


def test_filters_and_search(report: dict) -> None:
    _check(report, "filters")


def test_score_series_and_frontier(report: dict) -> None:
    _check(report, "scores")


def test_next_source_wraps(report: dict) -> None:
    _check(report, "nextSource")


def test_default_source_prefers_live_then_best_then_stopped_then_root(report: dict) -> None:
    _check(report, "defaultSource")


def test_artifact_at_guard(report: dict) -> None:
    _check(report, "artifactGuard")


def test_reducer_is_pure(report: dict) -> None:
    _check(report, "purity")


def test_store_notifies_subscribers(report: dict) -> None:
    _check(report, "store")
