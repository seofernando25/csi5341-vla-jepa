"""Live documents may only name commands and paths that exist."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE_DOCS = sorted(
    [
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "studies" / "README.md",
        ROOT / "web" / "README.md",
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / ".agents" / "skills").glob("*/SKILL.md"),
    ]
)
TOP_LEVEL = {
    ".agents",
    ".github",
    "docs",
    "evaluation",
    "report",
    "rsi",
    "scripts",
    "studies",
    "tests",
    "web",
}
# Built or runtime files: named in documents, never committed.
GENERATED = re.compile(
    r"^report/(figures|data|main\.pdf)\b"
    r"|^studies/program-v1/(summary\.csv|cka\.json)$"
    r"|/(dist|node_modules|outputs)/"
)
PLACEHOLDER = re.compile(r"[<>*{}$…]|\.\.\.")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
COMMAND = re.compile(r"(?<![\w/.-])vla-[a-z]+(?:-[a-z]+)*")
STALE = re.compile(r"pending merge|no measurements yet|\bt3/[a-z0-9-]+", re.IGNORECASE)


def _ids(paths: list[Path]) -> list[str]:
    return [str(path.relative_to(ROOT)) for path in paths]


def _scripts() -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return set(project["scripts"])


def _named_paths(text: str) -> set[str]:
    paths = set()
    for span in CODE_SPAN.findall(text):
        for token in span.split():
            token = token.strip("\"'(),;:").split("::")[0].rstrip(".")
            if "/" not in token or PLACEHOLDER.search(token) or GENERATED.search(token):
                continue
            if token.split("/")[0] in TOP_LEVEL:
                paths.add(token)
    return paths


@pytest.mark.parametrize("doc", LIVE_DOCS, ids=_ids(LIVE_DOCS))
def test_named_commands_are_entry_points(doc: Path) -> None:
    unknown = set(COMMAND.findall(doc.read_text())) - _scripts()
    assert not unknown, f"not in [project.scripts]: {sorted(unknown)}"


@pytest.mark.parametrize("doc", LIVE_DOCS, ids=_ids(LIVE_DOCS))
def test_named_paths_exist(doc: Path) -> None:
    missing = sorted(p for p in _named_paths(doc.read_text()) if not (ROOT / p).exists())
    assert not missing, f"named but absent: {missing}"


@pytest.mark.parametrize("doc", LIVE_DOCS, ids=_ids(LIVE_DOCS))
def test_no_transient_state(doc: Path) -> None:
    """Branch names and merge status belong in HANDOFF.md, which this check does not read."""
    found = sorted(set(STALE.findall(doc.read_text())))
    assert not found, f"transient state in a live document: {found}"


def test_every_entry_point_is_documented() -> None:
    commands = (ROOT / "docs" / "COMMANDS.md").read_text()
    undocumented = sorted(name for name in _scripts() if name not in commands)
    assert not undocumented, f"missing from docs/COMMANDS.md: {undocumented}"
