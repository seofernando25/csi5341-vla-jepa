"""Compact history digest of probed Dream-RSI cells, gates, and reference points."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from rsi.action_space import canonical_signature_from_changes, extract_knobs_from_changes

DREAM_DIR = Path("studies/dream-rsi")
MEASUREMENTS_DIR = Path("studies/compression/measurements")
NUMBERS_PATH = Path("report/data/numbers.json")


def _natural_round_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (1, int(match.group(1)), path.name) if match else (1, 0, path.name)


def _read_smoke_failing_tasks(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        return sorted(int(r["task_id"]) for r in csv.DictReader(f) if int(r["success"]) == 0)


def _read_dev_failing_tasks(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    totals: dict[int, int] = {}
    fails: dict[int, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            tid = int(row["task_id"])
            totals[tid] = totals.get(tid, 0) + 1
            if int(row["success"]) == 0:
                fails[tid] = fails.get(tid, 0) + 1
    out = {str(tid): f"{fails[tid]}/{totals[tid]}" for tid in sorted(fails)}
    return out or None


def _reference_points(numbers_path: Path) -> dict[str, Any]:
    if not numbers_path.exists():
        return {}
    rows = json.loads(numbers_path.read_text()).get("rows", {})
    refs: dict[str, Any] = {}
    for key in ("B16", "C1", "C2"):
        row = rows.get(key)
        if not row:
            continue
        w_gb = row.get("weights_gb")
        refs[key] = {
            "candidate": row.get("candidate", key),
            "weights_mb": round(float(w_gb) * 1000.0, 1) if w_gb is not None else None,
            "peak_allocated_gb": round(float(row["peak_allocated_gb"]), 3),
            "median_ms": round(float(row["median_ms"]), 2),
            "p95_ms": round(float(row["p95_ms"]), 2),
            "success": f"{int(row['successes'])}/500 ({float(row['success_pct']):.1f}%)",
        }
    return refs


def _fallback_changes_map(dream_dir: Path) -> dict[str, list[dict[str, Any]]]:
    from rsi.action_space import build_recipe_changes
    from rsi.stage2 import build_stage2_candidates

    mapping: dict[str, list[dict[str, Any]]] = {}
    for cand in build_stage2_candidates():
        mapping[cand.cell_id] = cand.recipe["changes"]
    prop_dir = dream_dir / "proposals"
    if prop_dir.exists():
        for path in sorted(prop_dir.glob("*.json")):
            data = json.loads(path.read_text())
            for cand in data.get("candidates", []):
                mapping[cand["id"]] = build_recipe_changes(**cand["knobs"], base=cand["base"])
    return mapping


def collect_history(
    dream_dir: Path | str = DREAM_DIR,
    measurements_dir: Path | str = MEASUREMENTS_DIR,
    numbers_path: Path | str = NUMBERS_PATH,
) -> tuple[dict[str, Any], set[str], dict[tuple[Any, ...], str]]:
    """Collect structured history digest, probed cell IDs, and canonical config signatures."""
    dream = Path(dream_dir)
    meas = Path(measurements_dir)
    fallback_changes = _fallback_changes_map(dream)

    raw_cells: list[tuple[str, dict[str, Any], dict[str, Any] | None, str | None]] = []
    seen_ids: set[str] = set()

    s2_path = dream / "qwen_stage2.json"
    if s2_path.exists():
        s2 = json.loads(s2_path.read_text())
        s2_dev = {r["candidate"]: r for r in s2.get("development_promotions", [])}
        for rec in s2.get("smoke_results", []):
            cid = rec["candidate"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                raw_cells.append(("s2", rec, s2_dev.get(cid), None))

    for path in sorted(dream.glob("live_round*.json"), key=_natural_round_key):
        data = json.loads(path.read_text())
        rnd = str(data.get("round_id", path.stem))
        machine = data.get("machine")
        dev_map: dict[str, dict[str, Any]] = {}
        for arm in ("control", "selected"):
            for rec in data.get("development_promotions", {}).get(arm, []):
                dev_map.setdefault(rec["candidate"], rec)
        for arm in ("control", "selected"):
            for rec in data.get(arm, {}).get("results", []):
                cid = rec["candidate"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    raw_cells.append((rnd, rec, dev_map.get(cid), rec.get("machine") or machine))

    probed_cells: list[dict[str, Any]] = []
    probed_signatures: dict[tuple[Any, ...], str] = {}

    for rnd, smoke_rec, dev_rec, gpu in raw_cells:
        cid = smoke_rec["candidate"]
        run_path = meas / cid / "smoke" / "run.json"
        changes: list[dict[str, Any]] = fallback_changes.get(cid, [])
        if run_path.exists():
            run_data = json.loads(run_path.read_text())
            changes = run_data.get("candidate", {}).get("changes", changes)
            gpu = gpu or run_data.get("environment", {}).get("gpu")

        base, knobs = extract_knobs_from_changes(changes)
        if changes:
            probed_signatures.setdefault(canonical_signature_from_changes(changes), cid)

        res_bytes = round(float(smoke_rec.get("resident_bytes", 0)))
        entry: dict[str, Any] = {
            "id": cid,
            "round": rnd,
            "base": base,
            "knobs": knobs,
            "resident_bytes": res_bytes,
            "resident_mb": round(res_bytes / 1e6, 1),
            "smoke": (
                f"{int(smoke_rec['successes'])}/{int(smoke_rec['episodes'])}"
                if smoke_rec.get("status") == "completed"
                else str(smoke_rec.get("status", "unknown"))
            ),
        }
        if dev_rec is not None:
            entry["dev"] = (
                f"{int(dev_rec['successes'])}/{int(dev_rec['episodes'])}"
                if dev_rec.get("status") == "completed"
                else str(dev_rec.get("status", "unknown"))
            )
        smoke_fails = _read_smoke_failing_tasks(meas / cid / "smoke" / "episodes.csv")
        if smoke_fails:
            entry["smoke_failing_tasks"] = smoke_fails
        dev_fails = _read_dev_failing_tasks(meas / cid / "development" / "episodes.csv")
        if dev_fails:
            entry["dev_failing_tasks"] = dev_fails
        if gpu:
            entry["gpu"] = gpu
        probed_cells.append(entry)

    digest = {
        "gates": {"smoke": ">= 8/10", "development": ">= 77/100"},
        "reference_points": _reference_points(Path(numbers_path)),
        "probed_cells": probed_cells,
    }
    return digest, seen_ids, probed_signatures


def history_digest(
    dream_dir: Path | str = DREAM_DIR,
    measurements_dir: Path | str = MEASUREMENTS_DIR,
    numbers_path: Path | str = NUMBERS_PATH,
) -> str:
    """Return a compact, deterministic JSON digest of all probed cells, gates, and baselines."""
    digest, _, _ = collect_history(dream_dir, measurements_dir, numbers_path)
    return json.dumps(digest, indent=2)
