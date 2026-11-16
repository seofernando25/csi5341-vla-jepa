"""Report inputs read from committed evidence under studies/, plus derived statistics.

Every reader is non-fatal: an absent study directory or record yields a `status` stub
(`{"status": "missing", ...}`) or an empty frame, never an exception, so the plots build can
run before the clean-room evidence lands.

Cells are addressed by plan id: `measurements/<recipe>/<phase>/run.json`, where `<recipe>` is a
file stem under `recipes/`. Nothing here reads pre-cleanroom evidence.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pandas as pd

from evaluation.analysis import paired_difference, wilson
from evaluation.common import ROOT

DATA = ROOT / "report" / "data"
STUDY = ROOT / "studies" / "program-v1"  # clean-room recipes, measurements and Dream-RSI cycles
GB = 1e9
ROLLOUT_PHASES = ("smoke", "development", "final")
PHASES = ("screen", *ROLLOUT_PHASES, "size", "benchmark")


def load_json(path: Path):
    return json.loads(path.read_text())


def optional_json(path: Path) -> dict | None:
    """Parsed JSON, or None when the record is absent or half-written."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def rel(path: Path) -> str:
    """Repository-relative path for provenance, falling back outside the repository."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def missing(path: Path, what: str) -> dict:
    """Status stub a reader returns instead of raising when its evidence is absent."""
    return {"status": "missing", "what": what, "source": rel(path)}


def parameter_census() -> dict:
    """Parameter counts of lerobot/VLA-JEPA-LIBERO by module, from safetensors headers.

    Reads the cached checkpoint when present and stores the result, so later builds work
    without the 5.5 GB checkpoint.
    """
    stored = DATA / "parameter_census.json"
    cache = Path.home() / ".cache/huggingface/hub/models--lerobot--VLA-JEPA-LIBERO/snapshots"
    files = sorted(cache.glob("*/model.safetensors"))
    if not files:
        return load_json(stored)
    with files[0].open("rb") as handle:
        header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
    groups: dict[str, int] = {}
    for name, info in header.items():
        if name == "__metadata__" or name.endswith("lm_head.weight"):  # lm_head is a tied alias
            continue
        if "language_model.layers" in name and name.endswith("proj.weight"):
            group = "language decoder linear"
        elif "language_model" in name:
            group = "language embeddings and norms"
        elif "visual" in name:
            group = "vision encoder and mergers"
        elif "video_encoder" in name:
            group = "V-JEPA 2 encoder (world model)"
        elif "video_predictor" in name:
            group = "latent predictor (world model)"
        elif "action_model" in name:
            group = "action head"
        else:
            group = "other"
        groups[group] = groups.get(group, 0) + math.prod(info["shape"])
    census = {
        "checkpoint": "lerobot/VLA-JEPA-LIBERO@735d9f692981e286ade093b5046627eda876e5d0",
        "dtype": "bfloat16",
        "parameters": dict(sorted(groups.items(), key=lambda item: -item[1])),
    }
    stored.write_text(json.dumps(census, indent=2) + "\n")
    return census


def plan() -> dict:
    """Pre-registered precision plan: arms, hypotheses and gates."""
    path = STUDY / "precision_surface_plan.json"
    record = optional_json(path)
    if record is None:
        return missing(path, "precision surface plan")
    return {
        "status": "ok",
        "source": rel(path),
        "arms": record["arms"],
        "hypotheses": record["hypotheses"],
        "decision_rules": record["decision_rules"],
    }


def recipes() -> dict:
    """Every authored recipe, keyed by plan id."""
    return {path.stem: load_json(path) for path in sorted((STUDY / "recipes").glob("*.json"))}


def cells() -> dict:
    """`{recipe: {phase: record}}` for every completed cell under measurements/."""
    root = STUDY / "measurements"
    found: dict = {}
    for path in sorted(root.glob("*/*/run.json")):
        record = optional_json(path)
        if record is None or record.get("status") != "completed":
            continue
        found.setdefault(path.parent.parent.name, {})[path.parent.name] = record
    return found


def effective_bits(summary: dict) -> dict:
    """`8 S(x)/N(x)` per module root, from the per-module storage the run recorded."""
    out = {}
    for name, module in (summary.get("modules") or {}).items():
        elements = module.get("logical_parameter_elements")
        storage = module.get("resident_tensor_storage_bytes")
        if elements and storage:
            out[name] = 8 * storage / elements
    return out


def wilson_pct(successes, episodes):
    return [100 * value for value in wilson(int(successes), int(episodes))]


def audit_record(study: Path) -> dict[str, dict]:
    """Post-fix score and audit class per node, from the exploit re-measure record."""
    record = optional_json(study / "exploit" / "remeasure.json") or {}
    return {
        node["node"]: node
        for node in record.get("nodes", [])
        if isinstance(node, dict) and node.get("node")
    }


def _final_episodes(candidate: str) -> Path:
    """Reserved episodes for a recipe cell (`measurements/`) or a solution cell (`reserved/`)."""
    for path in (
        STUDY / "measurements" / candidate / "final" / "episodes.csv",
        STUDY / "reserved" / candidate / "final" / "episodes.csv",
    ):
        if path.exists():
            return path
    return STUDY / "measurements" / candidate / "final" / "episodes.csv"


def reserved_pair(candidate: str) -> dict | None:
    """Paired reserved difference against B16, or None when either arm is incomplete."""
    base_path = STUDY / "measurements" / "B16" / "final" / "episodes.csv"
    other_path = _final_episodes(candidate)
    if not (base_path.exists() and other_path.exists()):
        return None
    base, other = pd.read_csv(base_path), pd.read_csv(other_path)
    try:
        difference = paired_difference(base.to_dict("records"), other.to_dict("records"))
        merged = base.merge(other, on=["task_id", "initial_state_hash"], suffixes=("_b", "_o"))
    except (KeyError, ValueError):
        return None
    return {
        "difference_pp": difference["difference_pp"],
        "ci95_pp": difference["ci95_pp"],
        "per_task": {str(t): int(g.success.sum()) for t, g in other.groupby("task_id")},
        "gained": int(((merged.success_b == 0) & (merged.success_o == 1)).sum()),
        "lost": int(((merged.success_b == 1) & (merged.success_o == 0)).sum()),
    }


def compact(candidate: str, phases: dict, family: str | None) -> dict:
    """One report row per recipe, holding only the cells that exist."""
    row: dict = {"recipe": candidate, "family": family}
    screen = phases.get("screen", {}).get("summary", {})
    if screen:
        row["screen_mean_arm_change_mse"] = screen.get("mean_physical_arm_change_mse")
        row["screen_gripper_disagreement"] = screen.get("mean_gripper_disagreement")
        if "peak_allocated_bytes" in screen:
            row["screen_peak_allocated_gb"] = screen["peak_allocated_bytes"] / GB
        row["effective_bits"] = effective_bits(screen)
    for phase in ROLLOUT_PHASES:
        summary = phases.get(phase, {}).get("summary")
        if not summary:
            continue
        successes, episodes = summary.get("successes"), summary.get("episodes")
        row[f"{phase}_successes"] = int(successes)
        row[f"{phase}_episodes"] = int(episodes)
        row[f"{phase}_success_pct"] = 100 * int(successes) / int(episodes)
        row[f"{phase}_wilson_pct"] = wilson_pct(successes, episodes)
    benchmark = phases.get("benchmark", {}).get("summary")
    if benchmark:
        row["benchmark_peak_allocated_gb"] = (
            max(peak["allocated_bytes"] for peak in benchmark["memory_peaks"]) / GB
        )
        row["benchmark_median_ms"] = benchmark["pipeline"]["median_ms"]
        row["benchmark_p95_ms"] = benchmark["pipeline"]["p95_ms"]
        row.setdefault("effective_bits", effective_bits(benchmark))
    size = phases.get("size", {}).get("summary")
    if size:
        row["weights_gb"] = size["serialized_weights"]["bytes"] / GB
    elif benchmark and "serialized_weights" in benchmark:
        row["weights_gb"] = benchmark["serialized_weights"]["bytes"] / GB
    pair = reserved_pair(candidate)
    if pair is not None:
        row["final_paired_pp"] = [pair["difference_pp"], *pair["ci95_pp"]]
        row["final_per_task"] = pair["per_task"]
        row["final_transitions"] = {"gained": pair["gained"], "lost": pair["lost"]}
    row["source"] = rel(STUDY / "measurements" / candidate)
    return {key: value for key, value in row.items() if value is not None}


def peak_gb(row: dict) -> float:
    """Peak allocated memory, preferring the benchmark phase over the screen."""
    return row.get("benchmark_peak_allocated_gb", row.get("screen_peak_allocated_gb"))


def latency_ladder(rows: dict) -> dict:
    """Median latency, p95 and peak memory per recipe that has a benchmark phase."""
    return {
        candidate: {
            "median_ms": row["benchmark_median_ms"],
            "p95_ms": row["benchmark_p95_ms"],
            "peak_allocated_gb": row["benchmark_peak_allocated_gb"],
        }
        for candidate, row in rows.items()
        if "benchmark_median_ms" in row
    }


def _node_file(root: Path, cell: str, name: str) -> dict:
    """First record named `name` for a node, wherever its cycle lives."""
    for path in sorted((root / "cycles").glob(f"*/nodes/{cell}/{name}")):
        return optional_json(path) or {}
    return {}


def _reserved_trials(root: Path) -> int:
    """Reserved episodes a complete cell must record, from the study protocol."""
    protocol = optional_json(root / "protocol.json") or {}
    phases = protocol.get("phases") or {}
    return int((phases.get("reserved") or {}).get("trials", 500))


def solution_reserved(study: Path | None = None) -> dict:
    """Reserved-500 for Dream-RSI solution cells: success, paired difference, memory, development."""
    root = Path(study) if study is not None else STUDY
    base_path = root / "measurements" / "B16" / "final" / "episodes.csv"
    audit = audit_record(root)
    expected = _reserved_trials(root)
    cells = []
    for path in sorted((root / "reserved").glob("*/final/episodes.csv")):
        cell = path.parent.parent.name
        result = optional_json(path.parent / "result.json") or {}
        if result.get("status") != "ok":
            continue
        rows = pd.read_csv(path)
        if expected and len(rows) < expected:
            continue
        successes, episodes = int(rows.success.sum()), len(rows)
        entry = {
            "cell": cell,
            "successes": successes,
            "episodes": episodes,
            "success_pct": 100 * successes / episodes if episodes else None,
            "wilson_pct": wilson_pct(successes, episodes) if episodes else None,
            "source": rel(path),
        }
        if base_path.exists():
            base = pd.read_csv(base_path)
            try:
                difference = paired_difference(base.to_dict("records"), rows.to_dict("records"))
                entry["difference_pp"] = difference["difference_pp"]
                entry["ci95_pp"] = difference["ci95_pp"]
            except (KeyError, ValueError):
                pass
        fixed = (audit.get(cell) or {}).get("new") or {}
        sizes = _node_file(root, cell, "score.json").get("sizes") or {}
        resident = fixed.get("resident_bytes") or sizes.get("resident_bytes")
        baseline = fixed.get("baseline_resident_bytes") or sizes.get("baseline_resident_bytes")
        if resident and baseline:
            entry["resident_gb"] = resident / GB
            entry["size_reduction"] = 1 - resident / baseline
        development = _node_file(root, cell, "development.json")
        if development.get("successes") is not None:
            entry["development_successes"] = int(development["successes"])
            entry["development_episodes"] = int(development.get("episodes") or 0)
        cells.append(entry)
    return {
        "status": "ok" if cells else "empty",
        "source": rel(root / "reserved"),
        "cells": cells,
    }


def collect() -> dict:
    """Compact rows for every measured recipe, plus the B16 failure diagnostics."""
    root = STUDY / "measurements"
    if not root.is_dir():
        return missing(root, "clean-room measurements")
    families = {name: recipe.get("family") for name, recipe in recipes().items()}
    rows = {
        candidate: compact(candidate, phases, families.get(candidate))
        for candidate, phases in cells().items()
    }
    out: dict = {"status": "ok", "rows": rows, "latency_ladder": latency_ladder(rows)}
    base_path = root / "B16" / "final" / "episodes.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        failures = base[base.success == 0]
        out.update(
            b16_failures_at_step_cap=int((failures.steps == failures.steps.max()).sum()),
            b16_step_cap=int(base.steps.max()),
            b16_task5_failures=int((failures.task_id == 5).sum()),
        )
    return out


def memory_ceiling(census: dict, b16_peak_gb: float) -> dict:
    """Ideal peak-memory saving from quantizing only the language-decoder linear weights."""
    n = census["parameters"]["language decoder linear"]
    bytes_per_param = {"INT8": 1.0, "NF4": (4 + 8 / 64 + 32 / (64 * 256)) / 8}
    out = {"decoder_linear_params": n, "total_params": sum(census["parameters"].values())}
    for name, b in bytes_per_param.items():
        saved = n * (2 - b) / GB
        out[name] = {"saved_gb": saved, "ceiling_pct": 100 * saved / b16_peak_gb}
    return out


def component_of(candidate: str, family: str | None) -> str:
    """Module group a recipe targets, from its declared family."""
    text = f"{family or ''} {candidate}"
    for key in ("joint", "language", "vision", "action"):
        if key in text:
            return "language" if key == "language" else key
    return "other"


def recovery_dev() -> dict:
    """SmolVLM four-layer RGB recovery vs B16 on 100 development trials."""
    path = ROOT / "studies" / "recovery" / "diagnostics" / "expanded_development_pair.json"
    d = optional_json(path)
    if d is None:
        return missing(path, "recovery development pair")
    episodes_n = int(d["episodes_per_model"])
    rec = int(d["parent_successes"])
    base = int(d["baseline_successes"])
    return {
        "recovered_successes": rec,
        "baseline_successes": base,
        "episodes": episodes_n,
        "recovered_wilson_pct": [100 * v for v in wilson(rec, episodes_n)],
        "baseline_wilson_pct": [100 * v for v in wilson(base, episodes_n)],
    }
