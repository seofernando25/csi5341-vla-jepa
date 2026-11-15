"""Story model checks: the pure JS model runs under node, both modules parse, no DOM."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STORY = ROOT / "rsi" / "watch_static" / "js" / "story"
MODEL_FILES = (STORY / "model.js", STORY / "model_dream.js")
SCRIPT = ROOT / "tests" / "js" / "story_model.mjs"
NODE = shutil.which("node")
FORBIDDEN = ("document", "window", "localStorage", "fetch", "innerHTML", "Date.now", "Math.random")


def _env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key != "NODE_OPTIONS"}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_story_model_script_passes() -> None:
    result = subprocess.run(
        [NODE, str(SCRIPT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        cwd=ROOT,
        env=_env(),
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"^# tests (\d+)$", result.stdout, re.MULTILINE)
    assert match, result.stdout
    assert int(match.group(1)) >= 25
    assert "# fail 0" in result.stdout


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_story_model_parses_with_node() -> None:
    for path in MODEL_FILES:
        assert path.is_file(), path
        result = subprocess.run(
            [NODE, "--input-type=module", "--check"],
            input=path.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            cwd=ROOT,
            env=_env(),
            check=False,
        )
        assert result.returncode == 0, f"{path.name} does not parse:\n{result.stderr}"


def test_story_model_is_pure() -> None:
    for path in MODEL_FILES:
        assert path.is_file(), path
        code = path.read_text(encoding="utf-8")
        hits = [token for token in FORBIDDEN if token in code]
        assert not hits, f"{path.name} touches {hits}"
