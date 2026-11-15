"""Read-only Dream-RSI policy-development payload for the watch Story dreaming chapter.

The writer (`rsi/policy_dev.py:develop_policies_llm`) leaves one tree per cycle, and a
calibration output root holds the same tree without the `policies/<cycle>` level, so the root is
resolved in that order and reported relative to the study. Every field comes from a written
file; an absent file is null.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rsi.watch_files import STEP_RE, inside, read_json, read_text

__all__ = ["SIZE_CAP", "dream_payload", "policy_root"]

SIZE_CAP, TRACES_CAP = 300_000, 2_000_000
STATUSES = ("incumbent", "accepted", "not_better", "rejected")
MACHINE = re.compile(r"/(?:home|Users|tmp|private|var|mnt|root|opt)/[^\s\"']*")
KEYS = (  # noqa: SIM905 - an eleven-item tuple costs thirteen lines
    "source_dir recorded_utc protocol worlds tree candidates curves rounds heldout verdict agent"
).split()


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _str(value: Any) -> str | None:
    return None if value is None else str(value)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _fields(src: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Project `src` onto `spec` (field -> converter); a missing field becomes null."""
    src = _dict(src)
    return {key: convert(src.get(key)) for key, convert in spec.items()}


POINT = {"probes": _int, "attainment": _num, "rounds": _int}
OUTCOME = {"cell_id": _str, "score": _num, "fail_class": _str}
VERDICT = {"selected_slot": _str, "incumbent_in_sample_V": _num, "selected_in_sample_V": _num}
CANDIDATE = {"mean_v": _num, "delta_vs_best": _num, "attempts": _int, "error": _str}
AGENT = {"backend": _str, "model": _str, "effort": _str, "calls": _int, "num_turns": _int}
AGENT["cost_usd"] = _num
NODE = {"cell_id": _str, "parent_id": _str, "branch": _int, "attempt": _int, "seq": _int}
NODE |= {"score": _num, "fail_class": _str, "evaluated": lambda v: v is True, "summary": _str}


def policy_root(study: Path, cycle: str) -> Path | None:
    """The root: live loop, named study, then a calibration root that names this cycle."""
    for candidate in (study / "policies" / cycle, study / "policies" / study.name):
        safe = inside(study, candidate)
        if safe is not None and safe.is_dir():
            return safe
    root = inside(study, study)
    if root is None:
        return None
    named = _str((read_json(study, root / "selection.json") or {}).get("cycle_id"))
    return None if named is not None and named != cycle else root


def _steps(study: Path, root: Path, selection: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per `step_NN` directory on disk, with its verdict, slot and recorded hash."""
    variants = [v for v in _list(selection.get("variants")) if isinstance(v, dict)]
    slots = {_int(v.get("step")): _str(v.get("slot")) for v in variants}
    shas = {_int(v.get("step")): _str(v.get("policy_sha256")) for v in variants}
    out: list[dict[str, Any]] = []
    for sdir in sorted(root.iterdir()) if root.is_dir() else []:
        match = STEP_RE.match(sdir.name)
        if match is None or not sdir.is_dir():
            continue
        step = int(match.group(1))
        verdict = read_json(study, root / "evidence" / "history" / sdir.name / "verdict.json")
        slot = _str((verdict or {}).get("slot")) or slots.get(step)
        row = {"step": step, "slot": slot, "sha": shas.get(step), "dir": sdir}
        out.append(row | {"verdict": verdict or {}})
    return out


def _step_curve(
    study: Path, root: Path, sdir: Path
) -> tuple[dict[str, Any], dict[str, bool], str | None]:
    """One step's points per world, its per-world flags and its policy hash; replay wins."""
    res = read_json(study, sdir / "replay" / "result.json")
    if res is None:
        res = read_json(study, root / "evidence" / "history" / sdir.name / "scaling_curve.json")
    if res is None:
        return {}, {}, None
    per: dict[str, Any] = {}
    flags: dict[str, bool] = {}
    for wid, world in _dict(res.get("worlds")).items():
        if isinstance(world, dict):
            per[str(wid)] = [
                {"beta": _num(p.get("beta"))} | _fields(p, POINT)
                for p in _list(world.get("points"))
                if isinstance(p, dict)
            ]
            flags[str(wid)] = world.get("degenerate") is True
    return per, flags, _str(res.get("policy_sha256"))


def _traces(study: Path, path: Path) -> list[dict[str, Any]]:
    """JSON lines of one replay trace file; a truncated or half-written line is skipped."""
    rows: list[dict[str, Any]] = []
    for line in (read_text(study, path, cap=TRACES_CAP) or "").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _rounds(
    study: Path, steps: list[dict[str, Any]], world: str | None
) -> tuple[dict[str, Any], dict[str, float]]:
    """Round-by-round batches for one world (all slots, all betas) and each world's max gain."""
    rounds: dict[str, Any] = {}
    gains: dict[str, float] = {}
    for item in steps:
        rows = _traces(study, item["dir"] / "replay" / "traces.jsonl")
        for row in rows:
            wid, gain = _str(row.get("world_id")), _num(row.get("max_gain"))
            if wid is not None and gain is not None:
                gains[wid] = max(gains.get(wid, gain), gain)
            beta = _num(row.get("beta"))
            if world is None or wid != world or beta is None or item["slot"] is None:
                continue
            betas = rounds.setdefault(item["slot"], {}).setdefault(world, {})
            entries = betas.setdefault(f"{beta:g}", [])
            for step in _list(row.get("steps")):
                if isinstance(step, dict):
                    outcomes = [_fields(o, OUTCOME) for o in _list(step.get("outcomes"))]
                    entries.append(
                        {
                            "round": _int(step.get("round")),
                            "batch": [str(cell) for cell in _list(step.get("batch"))],
                            "prefix_size": _int(step.get("prefix_size")),
                            "outcomes": outcomes,
                        }
                    )
        if world is None and rows:
            break
    return rounds, gains


def _world_block(
    study: Path, root: Path, world: str | None, meta: dict[str, bool], gains: dict[str, float]
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """The world list and one frozen replay tree: the requested world, else the first useful one."""
    wdir = root / "evidence" / "worlds"
    docs = [
        (_str(doc.get("world_id")) or path.stem.split("_", 1)[-1], doc)
        for path in sorted(wdir.glob("*.json"))
        if (doc := read_json(study, path)) is not None
    ]
    worlds = [
        {
            "id": wid,
            "nodes": len(_list(doc.get("nodes"))),
            "baseline_score": _num(doc.get("baseline_score")),
            "max_gain": gains.get(wid),
            "degenerate": meta.get(wid),
        }
        for wid, doc in docs
    ]
    if not docs:
        return None, None
    if world is not None:
        chosen = next((doc for doc in docs if doc[0] == world), None)
    else:
        chosen = next((doc for doc in docs if meta.get(doc[0]) is not True), docs[0])
    if chosen is None:
        return worlds, None
    nodes = [_fields(n, NODE) for n in _list(chosen[1].get("nodes")) if isinstance(n, dict)]
    return worlds, {"world_id": chosen[0], "nodes": nodes}


def _candidates(
    steps: list[dict[str, Any]], sha: dict[int, str | None]
) -> list[dict[str, Any]] | None:
    out = []
    for item in steps:
        status = _str(item["verdict"].get("status"))
        row = {"slot": item["slot"], "step": item["step"]}
        row["status"] = status if status in STATUSES else "running"
        row |= _fields(item["verdict"], CANDIDATE)
        row["sha256"] = sha.get(item["step"]) or item["sha"]
        out.append(row)
    return out or None


def _protocol(
    selection: dict[str, Any], candidates: list[dict[str, Any]] | None, curves: dict[str, Any]
) -> dict[str, Any] | None:
    betas = sorted({p["beta"] for per in curves.values() for pts in per.values() for p in pts})
    if not selection and not betas:
        return None
    return {
        "m": len(candidates) if candidates else None,
        "lambda_parallel": _num(selection.get("lambda_parallel")),
        "beta_grid": betas or None,
        "max_parallelism": _int(selection.get("max_parallelism")),
    }


def _selection_block(
    selection: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """held-out, in-sample verdict and agent cost as `selection.json` recorded them."""
    if not selection:
        return None, None, None
    held, agent = _dict(selection.get("heldout")), _dict(selection.get("agent"))
    heldout = None
    if held:
        heldout = {
            "num_worlds": _int(held.get("num_worlds")),
            "mean_V_by_slot": {str(k): _num(v) for k, v in held["mean_V_by_slot"].items()},
            "delta_vs_incumbent": _num(selection.get("heldout_delta_vs_incumbent")),
        }
    return heldout, _fields(selection, VERDICT), _fields(agent, AGENT) if agent else None


def _fit(payload: dict[str, Any]) -> dict[str, Any]:
    """Rounds are the only unbounded block; keep one round per beta if the payload overflows."""
    if len(json.dumps(payload)) > SIZE_CAP:
        for worlds in (payload.get("rounds") or {}).values():
            for betas in worlds.values():
                betas.update({key: entries[:1] for key, entries in betas.items()})
    return payload


def _scrub(value: Any, root: Path) -> Any:
    """No machine path leaves the server: the root prefix is dropped, other paths masked."""
    if isinstance(value, str):
        return MACHINE.sub("<path>", value.replace(str(root) + "/", "").replace(str(root), ""))
    if isinstance(value, dict):
        return {key: _scrub(item, root) for key, item in value.items()}
    return [_scrub(item, root) for item in value] if isinstance(value, list) else value


def dream_payload(study: Path, cycle: str, world: str | None = None) -> dict[str, Any]:
    """One cycle's policy-development record; every absent file is null, never a guess."""
    study = Path(study)
    root = policy_root(study, cycle)
    payload: dict[str, Any] = {"cycle": cycle, "status": "missing"} | dict.fromkeys(KEYS)
    if root is None:
        return payload
    ready = inside(study, root / "selection.json") is not None
    known = root != study or (study / "cycles" / cycle).is_dir()
    if not ready and not (known or (root / "evidence" / "history").is_dir()):
        return payload
    payload["status"] = "ready" if ready else "pending"
    relative = root.relative_to(study).as_posix()
    payload["source_dir"] = None if relative == "." else relative
    selection = read_json(study, root / "selection.json") or {}
    steps = _steps(study, root, selection)
    curves: dict[str, Any] = {}
    meta: dict[str, bool] = {}
    sha: dict[int, str | None] = {}
    for item in steps:
        per, flags, digest = _step_curve(study, root, item["dir"])
        sha[item["step"]] = digest
        meta.update(flags)
        if per and item["slot"]:
            curves[item["slot"]] = per
    rounds, gains = _rounds(study, steps, world)
    worlds, tree = _world_block(study, root, world, meta, gains)
    candidates = _candidates(steps, sha)
    heldout, verdict, agent = _selection_block(selection)
    payload.update(
        recorded_utc=_str(selection.get("recorded_utc")),
        protocol=_protocol(selection, candidates, curves),
        worlds=worlds,
        tree=tree,
        candidates=candidates,
        curves=curves or None,
        rounds=rounds or None,
        heldout=heldout,
        verdict=verdict,
        agent=agent,
    )
    return _scrub(_fit(payload), root)
