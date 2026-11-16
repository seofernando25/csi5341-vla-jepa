"""Source ids (`node:<cycle>:<cell>`, `policy:<cycle>:<step>`) and what each source exposes: artifacts and transcript events."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi.watch.dream import policy_root
from rsi.watch.files import ARTIFACT_CAP, HEX, PART, as_str, inside, log_path, read_json, read_text
from rsi.watch.transcript import read_events

ARTIFACTS = (
    "proposal",
    "diff",
    "score",
    "error",
    "solution",
    "policy",
    "verdict",
    "curve",
    "seed",
    "export",
    "prompt",
)
NODE_FILES = {
    "proposal": "proposal.md",
    "diff": "diff.patch",
    "score": "score.json",
    "error": "error.txt",
}


class SourceError(ValueError):
    """A source id that is not `node:<cycle>:<cell>` or `policy:<cycle>:<step>`."""


@dataclass(frozen=True)
class Source:
    """A parsed, validated source id; `name` is a cell id (node) or a step number (policy)."""

    kind: str
    cycle_id: str
    name: str


def parse_source(text: str | None) -> Source:
    """Parse a source id strictly; path components are validated before any filesystem use."""
    parts = (text or "").split(":")
    if len(parts) != 3 or parts[0] not in ("node", "policy"):
        raise SourceError(f"bad source id: {text!r}")
    cycle, name = parts[1], parts[2]
    for part in (cycle, name):
        if not PART.match(part) or part in (".", ".."):
            raise SourceError(f"bad source id component: {part!r}")
    if parts[0] == "policy" and (not name.isdigit() or int(name) < 0):
        raise SourceError(f"policy step must be an integer >= 0: {name!r}")
    return Source(parts[0], cycle, name)


def _solution(study: Path, src: Source, nodes_commit: str | None) -> str | None:
    if not nodes_commit or not HEX.match(nodes_commit):
        return None
    store = study / "cycles" / src.cycle_id / "snapshots.git"
    try:
        out = subprocess.run(
            ["git", "--git-dir", str(store), "show", f"{nodes_commit}:solution.py"],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout[: ARTIFACT_CAP * 4].decode("utf-8", errors="replace")[:ARTIFACT_CAP]


def read_artifact(study_dir: Path, source: Source, name: str) -> str | None:
    """Text of one artifact of a validated source, or None when it does not exist."""
    study = Path(study_dir)
    if source.kind == "node":
        ndir = study / "cycles" / source.cycle_id / "nodes" / source.name
        if name in NODE_FILES:
            return read_text(study, ndir / NODE_FILES[name])
        if name == "export":
            return read_text(study, ndir / "export" / "solution.py")
        if name == "prompt":
            return read_text(study, ndir / "evidence" / "TASK.md")
        if name == "solution":
            score = read_json(study, ndir / "score.json") or {}
            return _solution(study, source, as_str(score.get("commit")))
        return None
    root = policy_root(study, source.cycle_id) or study / "policies" / source.cycle_id
    sdir = root / f"step_{int(source.name):02d}"
    if name == "policy":
        return read_text(study, sdir / "candidate.py")
    if name == "seed":
        return read_text(study, sdir / "workspace" / "method.py")
    history = root / "evidence" / "history" / sdir.name
    if name == "verdict":
        return read_text(study, history / "verdict.json")
    if name == "curve":
        return read_text(study, sdir / "replay" / "result.json") or read_text(
            study, history / "scaling_curve.json"
        )
    return None


def source_finished(study: Path, source: Source) -> bool:
    if source.kind == "node":
        return (study / "cycles" / source.cycle_id / "nodes" / source.name / "score.json").is_file()
    root = policy_root(study, source.cycle_id) or study / "policies" / source.cycle_id
    return (
        root / "evidence" / "history" / f"step_{int(source.name):02d}" / "verdict.json"
    ).is_file()


def events_payload(study_dir: Path, source: Source, after: int) -> dict[str, Any]:
    """Events from `after` for one source, with the next index and done flag."""
    study = Path(study_dir)
    log = log_path(study, source)
    if log.exists() and inside(study, log) is None:
        return {"events": [], "next": after, "done": True, "error": "log path outside study"}
    finished = source_finished(study, source)
    events, nxt, done = read_events(log, after, finished=finished)
    return {"events": events, "next": nxt, "done": done}
