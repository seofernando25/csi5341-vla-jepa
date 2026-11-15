"""What one agent received: the context payload behind the watch page's Context view.

Everything here is read from files the run already writes. Missing material is reported as
`recorded: false` with a note; it is never invented. The payload is redacted by the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rsi.watch_files import (
    CONTEXT_CAP,
    PART,
    SourceLike,
    log_path,
    read_json,
    read_text,
    rel,
    step_dir,
)


def context_paths(study: Path, source: SourceLike) -> list[tuple[str, Path]]:
    """The files the agent reads for one source, in reading order (legacy `files` view)."""
    if source.kind == "node":
        ndir = study / "cycles" / source.cycle_id / "nodes" / source.name
        return [
            ("TASK.md", ndir / "evidence" / "TASK.md"),
            ("tree.json", ndir / "evidence" / "tree.json"),
            ("proposal.md", ndir / "proposal.md"),
        ]
    pdir = study / "policies" / source.cycle_id
    return [
        ("TASK.md", pdir / "evidence" / "TASK.md"),
        ("tree.json", pdir / "evidence" / "tree.json"),
        ("candidate.py", step_dir(study, source) / "candidate.py"),
    ]


def _tokens(text: str) -> int:
    """Rough token estimate: four characters per token, rounded up."""
    return (len(text) + 3) // 4


def _text_section(
    section_id: str,
    title: str,
    text: str | None,
    *,
    path: str | None = None,
    note: str | None = None,
    recorded: bool = True,
) -> dict[str, Any]:
    body = text or ""
    return {
        "id": section_id,
        "title": title,
        "recorded": recorded,
        "text": body if recorded else None,
        "chars": len(body) if recorded else 0,
        "tokens": _tokens(body) if recorded else 0,
        "path": path,
        "note": note,
        "items": [],
        "fields": {},
    }


def _not_recorded(section_id: str, title: str, note: str) -> dict[str, Any]:
    return _text_section(section_id, title, None, note=note, recorded=False)


def _command_section(command: dict[str, Any]) -> dict[str, Any]:
    fields = {key: value for key, value in command.items() if value is not None}
    return {
        "id": "command",
        "title": "Agent configuration",
        "recorded": bool(fields),
        "text": None,
        "chars": 0,
        "tokens": 0,
        "path": None,
        "note": "From logs/command.json. The sandbox argv is not shipped: it carries host paths.",
        "items": [],
        "fields": fields,
    }


def _evidence_sections(study: Path, evidence: Path) -> list[dict[str, Any]]:
    """tree.json and every earlier node's proposal/score/error, as the agent received them."""
    sections: list[dict[str, Any]] = []
    tree_path = evidence / "tree.json"
    tree_text = read_text(study, tree_path, cap=CONTEXT_CAP)
    if tree_text is not None:
        sections.append(
            _text_section(
                "tree",
                "Tree index",
                tree_text,
                path=rel(study, tree_path),
                note="tree.json: the nodes this agent could see.",
            )
        )
    tree = read_json(study, tree_path) or {}
    items: list[dict[str, Any]] = []
    entries = tree.get("nodes") if isinstance(tree.get("nodes"), list) else []
    for entry in entries:
        cell = entry.get("cell_id") if isinstance(entry, dict) else None
        if not isinstance(cell, str) or not PART.match(cell):
            continue
        files = []
        for name in ("proposal.md", "score.json", "error.txt"):
            text = read_text(study, evidence / "nodes" / cell / name, cap=CONTEXT_CAP)
            if text is None:
                continue
            files.append({"name": name, "text": text, "chars": len(text), "tokens": _tokens(text)})
        if files:
            items.append({"cell_id": cell, "files": files})
    if items:
        sections.append(
            {
                "id": "earlier",
                "title": "Earlier node material",
                "recorded": True,
                "text": None,
                "chars": sum(item["chars"] for entry in items for item in entry["files"]),
                "tokens": sum(item["tokens"] for entry in items for item in entry["files"]),
                "path": None,
                "note": "evidence/nodes/<cell>/: what every earlier node left behind.",
                "items": items,
                "fields": {},
            }
        )
    return sections


def _node_sections(
    study: Path, source: SourceLike, command: dict[str, Any]
) -> list[dict[str, Any]]:
    ndir = study / "cycles" / source.cycle_id / "nodes" / source.name
    task_path = ndir / "evidence" / "TASK.md"
    task = read_text(study, task_path, cap=CONTEXT_CAP)
    message = (
        _text_section(
            "message",
            "Task prompt",
            task,
            path=rel(study, task_path),
            note="evidence/TASK.md",
        )
        if task is not None
        else _not_recorded(
            "message", "Task prompt", "evidence/TASK.md is not recorded for this node."
        )
    )
    return [
        _not_recorded(
            "system",
            "System prompt",
            "Not written to the study.",
        ),
        message,
        *_evidence_sections(study, ndir / "evidence"),
        _command_section(command),
    ]


def _policy_sections(
    study: Path, source: SourceLike, command: dict[str, Any]
) -> list[dict[str, Any]]:
    pdir = study / "policies" / source.cycle_id
    sdir = step_dir(study, source)
    history = pdir / "evidence" / "history" / sdir.name
    task_path = pdir / "evidence" / "TASK.md"
    task = read_text(study, task_path, cap=CONTEXT_CAP)
    message = (
        _text_section(
            "message",
            "Task prompt",
            task,
            path=rel(study, task_path),
            note="policies/<cycle>/evidence/TASK.md",
        )
        if task is not None
        else _not_recorded(
            "message", "Task prompt", "The policy-development prompt is not recorded."
        )
    )
    seed_path = sdir / "workspace" / "method.py"
    seed = read_text(study, seed_path, cap=CONTEXT_CAP)
    candidate_path = sdir / "candidate.py"
    candidate = read_text(study, candidate_path, cap=CONTEXT_CAP)
    sections = [
        _not_recorded(
            "system",
            "System prompt",
            "Not written to the study.",
        ),
        message,
        _text_section(
            "seed",
            "Incumbent code (seed)",
            seed,
            path=rel(study, seed_path),
            note="workspace/method.py as handed to the agent.",
            recorded=seed is not None,
        )
        if seed is not None
        else _not_recorded("seed", "Incumbent code (seed)", "workspace/method.py is not recorded."),
        _text_section(
            "candidate",
            "Candidate policy",
            candidate,
            path=rel(study, candidate_path),
            note="candidate.py as audited and replayed.",
            recorded=candidate is not None,
        )
        if candidate is not None
        else _not_recorded("candidate", "Candidate policy", "candidate.py is not recorded."),
    ]
    items = []
    for label, path in (
        ("verdict.json", history / "verdict.json"),
        ("scaling_curve.json", history / "scaling_curve.json"),
        ("beta_sweep.json", pdir / "evidence" / "beta_sweep.json"),
        ("live_cycle_manifests.json", pdir / "evidence" / "live_cycle_manifests.json"),
    ):
        text = read_text(study, path, cap=CONTEXT_CAP)
        if text is None:
            continue
        items.append({"name": label, "text": text, "chars": len(text), "tokens": _tokens(text)})
    if items:
        sections.append(
            {
                "id": "evidence",
                "title": "Replay evidence",
                "recorded": True,
                "text": None,
                "chars": sum(item["chars"] for item in items),
                "tokens": sum(item["tokens"] for item in items),
                "path": None,
                "note": "The archived traces and verdicts the agent inspects.",
                "items": items,
                "fields": {},
            }
        )
    sections.append(_command_section(command))
    return sections


def context_payload(study_dir: Path, source: SourceLike) -> dict[str, Any]:
    """Read-only view of what the agent sees: evidence files, model and sandbox flags.

    The command line is deliberately not shipped: it carries host paths (`/usr/bin/bwrap`,
    the credential mount, the workspace) that say nothing about the agent's context.
    `sections` is the same material split the way the Context view presents it; `files` and
    `command` stay for compatibility.
    """
    study = Path(study_dir)
    files = []
    for name, path in context_paths(study, source):
        text = read_text(study, path, cap=CONTEXT_CAP + 1)
        if text is None:
            continue
        files.append(
            {
                "name": name,
                "path": rel(study, path),
                "text": text[:CONTEXT_CAP],
                "truncated": len(text) > CONTEXT_CAP,
            }
        )
    meta = read_json(study, log_path(study, source).parent / "command.json") or {}
    command = {
        "backend": _str(meta.get("backend")),
        "model": _str(meta.get("model")),
        "effort": _str(meta.get("effort")),
        "tools": _str(meta.get("tools")),
        "cli_version": _str(meta.get("cli_version")),
        "network": meta.get("network") is True,
        "sandboxed": meta.get("use_bwrap") is True,
    }
    sections = (
        _node_sections(study, source, command)
        if source.kind == "node"
        else _policy_sections(study, source, command)
    )
    return {
        "source": f"{source.kind}:{source.cycle_id}:{source.name}",
        "kind": source.kind,
        "log": rel(study, log_path(study, source)),
        "files": files,
        "command": command,
        "sections": sections,
    }


def _str(value: Any) -> str | None:
    return None if value is None else str(value)
