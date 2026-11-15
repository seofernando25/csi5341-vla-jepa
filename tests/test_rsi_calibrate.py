"""Tests for the policy-update calibration command (fake agents only, no real pi)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rsi.__main__ import main
from rsi.calibrate import classify_rule_break, run_calibration

FAKE_AGENT = """
import pathlib, sys
mode = sys.argv[1]
method = pathlib.Path("method.py")
if mode == "broken":
    method.write_text("def broken(:\\n")
"""

CALIBRATION_KEYS = {
    "tag",
    "prompt_path",
    "prompt_sha256",
    "model",
    "m",
    "max_parallelism",
    "lambda",
    "train_worlds",
    "steps",
    "train",
    "heldout",
    "by_family",
    "rule_breaks",
    "verdict",
    "selected_slot",
    "selected_sha256",
    "wall_seconds",
}


def _agent(tmp_path: Path, mode: str) -> list[str]:
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT, encoding="utf-8")
    return [sys.executable, str(script), mode]


def _calibrate(tmp_path: Path, mode: str) -> dict:
    return run_calibration(
        tag="t",
        m=4,
        max_parallelism=2,
        families=["deep_winner", "cliff"],
        seeds_per_family=1,
        agent_cmd=_agent(tmp_path, mode),
        timeout=60.0,
        out_root=tmp_path / "out",
        use_bwrap=False,
    )


def test_unchanged_agent_gives_no_change(tmp_path: Path) -> None:
    payload = _calibrate(tmp_path, "noop")
    assert payload["verdict"] == "no_change"
    assert payload["selected_slot"] == "m0_incumbent"
    assert [s["status"] for s in payload["steps"]] == ["not_better"] * 3
    assert set(payload["by_family"][0]) == {"policy", "family", "mean_v", "mean_bound", "regret"}
    on_disk = json.loads((tmp_path / "out" / "t" / "calibration.json").read_text())
    assert CALIBRATION_KEYS <= set(on_disk)


def test_broken_agent_rejects_every_step(tmp_path: Path) -> None:
    payload = _calibrate(tmp_path, "broken")
    assert payload["verdict"] == "no_change"
    assert all(s["status"] == "rejected" for s in payload["steps"])
    assert all(s["repair_used"] for s in payload["steps"])
    assert sum(payload["rule_breaks"].values()) >= 3
    assert payload["rule_breaks"]["audit"] >= 3


def test_out_root_under_studies_is_refused() -> None:
    with pytest.raises(SystemExit):
        main(["calibrate", "--tag", "x", "--out-root", "studies/x"])


def test_calibrate_help_works(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["calibrate", "--help"])
    assert exc.value.code == 0
    assert "--tag" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("error", "key"),
    [
        ("Syntax error in policy code: invalid syntax", "audit"),
        ("Forbidden import 'os'", "audit"),
        ("String literal 'x' copies a recorded cell or world id", "world_literal"),
        ("Sandboxed process timed out after 60.0s", "timeout"),
        ("method.py exceeds 200000 bytes", "output"),
        ("some worker traceback line", "replay"),
    ],
)
def test_rule_break_classification(error: str, key: str) -> None:
    assert classify_rule_break(error) == key
