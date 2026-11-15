"""Report inputs read from committed evidence under studies/, plus derived statistics.

Every reader is non-fatal: an absent study directory or record yields a `status` stub
(`{"status": "missing", ...}`) or an empty frame, never an exception, so the plots build can
run before the clean-room evidence lands.
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
EVAL = ROOT / "studies" / "evaluation"
RUNS = {
    "B16": "20260924T171416675676Z-B16-rollout",
    "Q8": "20260924T174749793177Z-Q8-rollout",
    "Q4": "20260924T183120067911Z-Q4-rollout",
}
GB = 1e9


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


def episodes(variant: str) -> pd.DataFrame:
    return pd.read_csv(EVAL / "runs" / RUNS[variant] / "episodes.csv")


def paired(base: pd.DataFrame, other: pd.DataFrame) -> dict:
    merged = base.merge(other, on=["task_id", "initial_state_hash"], suffixes=("_b", "_o"))
    assert len(merged) == 500
    b, o = merged.success_b, merged.success_o
    return {
        "gained": int(((b == 0) & (o == 1)).sum()),
        "lost": int(((b == 1) & (o == 0)).sum()),
        "both_failed": int(((b == 0) & (o == 0)).sum()),
    }


def measured_reserved(candidate: str) -> dict | None:
    """Reserved-trial summary of a candidate from its episodes, paired with B16; None if absent."""
    base_path = STUDY / "measurements" / "B16" / "final" / "episodes.csv"
    other_path = STUDY / "measurements" / candidate / "final" / "episodes.csv"
    if not (base_path.exists() and other_path.exists()):
        return None
    base = pd.read_csv(base_path)
    other = pd.read_csv(other_path)
    moves = paired(base, other)
    diff = paired_difference(base.to_dict("records"), other.to_dict("records"))
    return {
        "successes": int(other.success.sum()),
        "task_successes": {str(t): int(g.success.sum()) for t, g in other.groupby("task_id")},
        "paired_vs_B16": {
            **diff,
            "transitions": {
                "gained_success": moves["gained"],
                "lost_success": moves["lost"],
            },
        },
    }


def latency_ladder(comp: pd.DataFrame) -> dict:
    """Median latency and peak memory per recipe that has a benchmark phase."""
    out: dict = {}
    for cand in comp.index:
        row = comp.loc[cand]
        if pd.isna(row.get("benchmark_pipeline_median_ms")):
            continue
        out[cand] = {
            "median_ms": float(row["benchmark_pipeline_median_ms"]),
            "p95_ms": float(row["benchmark_pipeline_p95_ms"]),
            "peak_allocated_gb": float(row["benchmark_peak_allocated_bytes"]) / GB,
        }
    return out


def collect() -> dict:
    """Quantization and compression rows, plus the B16 control checks.

    Includes only the variants and recipes the committed evidence contains, and returns a
    `status` stub when the clean-room compression summary is absent.
    """
    comp_path = STUDY / "summary.csv"
    if not comp_path.exists():
        return missing(comp_path, "compression summary")
    comp = pd.read_csv(comp_path).set_index("candidate")
    rows: dict = {}
    summary_path = EVAL / "analysis" / "summary.json"
    t1_path = EVAL / "analysis" / "T1_summary.csv"
    if summary_path.exists() and t1_path.exists():
        summary = load_json(summary_path)["results"]
        t1 = pd.read_csv(t1_path).set_index("variant")
        for variant in ["B16", "Q8", "Q4", "S500"]:
            if variant not in summary:
                continue
            r = summary[variant]
            rows[variant] = {
                "study": "quantization",
                "successes": int(r["success"]["successes"]),
                "per_task": [
                    int(r["success"]["per_task"][str(t)]["successes"])
                    for t in range(10)
                    if str(t) in r["success"]["per_task"]
                ],
                "peak_allocated_gb": r["timing"]["peak_allocated_bytes"] / GB,
                "peak_reserved_gb": r["timing"]["peak_reserved_bytes"] / GB,
                "weights_gb": r["timing"]["serialized_weights_bytes"] / GB,
                "median_ms": r["timing"]["pipeline"]["median_ms"],
                "p95_ms": r["timing"]["pipeline"]["p95_ms"],
                "paired_pp": [
                    float(t1.loc[variant, "difference_pp"]),
                    float(t1.loc[variant, "difference_ci95_low_pp"]),
                    float(t1.loc[variant, "difference_ci95_high_pp"]),
                ],
            }
        base_eps = (
            episodes("B16") if (EVAL / "runs" / RUNS["B16"] / "episodes.csv").exists() else None
        )
        for variant in ["Q8", "Q4"]:
            run_path = EVAL / "runs" / RUNS[variant] / "episodes.csv"
            if base_eps is not None and variant in rows and run_path.exists():
                rows[variant]["transitions"] = paired(base_eps, episodes(variant))
    for cand in comp.index:
        r = comp.loc[cand]
        successes, trials = r.get("final_successes"), r.get("final_episodes")
        if pd.isna(successes) or pd.isna(trials):
            continue
        row = {
            "study": "compression",
            "candidate": cand,
            "successes": int(successes),
            "trials": int(trials),
            "success_pct": 100 * int(successes) / int(trials),
            "wilson_pct": [100 * v for v in wilson(int(successes), int(trials))],
        }
        for key, column, scale in [
            ("peak_allocated_gb", "benchmark_peak_allocated_bytes", GB),
            ("weights_gb", "packed_archive_bytes", GB),
            ("median_ms", "benchmark_pipeline_median_ms", 1.0),
            ("p95_ms", "benchmark_pipeline_p95_ms", 1.0),
        ]:
            value = r.get(column)
            if not pd.isna(value):
                row[key] = float(value) / scale
        res = measured_reserved(cand)
        if res is not None:
            row["per_task"] = dict(
                sorted(res["task_successes"].items(), key=lambda item: int(item[0]))
            )
            p = res["paired_vs_B16"]
            row["paired_pp"] = [p["difference_pp"], *p["ci95_pp"]]
            row["transitions"] = p["transitions"]
        rows[cand] = row
    out: dict = {"status": "ok", "rows": rows, "latency_ladder": latency_ladder(comp)}
    base_path = STUDY / "measurements" / "B16" / "final" / "episodes.csv"
    if base_path.exists():
        base_eps = pd.read_csv(base_path)
        failures = base_eps[base_eps.success == 0]
        out.update(
            b16_failures_at_step_cap=int((failures.steps == failures.steps.max()).sum()),
            b16_step_cap=int(base_eps.steps.max()),
            b16_task5_failures=int((failures.task_id == 5).sum()),
        )
        quant_b16 = EVAL / "runs" / RUNS["B16"] / "episodes.csv"
        if quant_b16.exists():
            m = pd.read_csv(quant_b16).merge(base_eps, on=["task_id", "initial_state_hash"])
            out["b16_repeat_agreement"] = float((m.success_x == m.success_y).mean())
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


def component_of(candidate: str) -> str:
    if candidate.startswith("deploy"):
        return "joint"
    for prefix, comp in [
        ("text", "language"),
        ("vision", "vision"),
        ("merger", "vision"),
        ("action", "action"),
        ("quantvla", "action"),
    ]:
        if candidate.startswith(prefix):
            return comp
    return "other"


def smoke(comp: pd.DataFrame, name: str) -> float:
    """10-trial smoke successes of one recipe; NaN when it was not tested."""
    if name not in comp.index or "smoke_successes" not in comp.columns:
        return math.nan
    return float(comp.loc[name, "smoke_successes"])


def divergence() -> pd.DataFrame:
    """Per-candidate offline arm change vs smoke success, from the compact divergence record."""
    columns = ["candidate", "arm_mse", "gripper", "smoke", "component"]
    path = STUDY / "divergence.json"
    comp_path = STUDY / "summary.csv"
    if not (path.exists() and comp_path.exists()):
        return pd.DataFrame(columns=columns)
    comp = pd.read_csv(comp_path)
    if "smoke_successes" not in comp.columns:
        return pd.DataFrame(columns=columns)
    rec = {r["candidate"]: r for r in load_json(path)["rows"]}
    rows = []
    for cand, smoke_successes in zip(comp.candidate, comp.smoke_successes, strict=True):
        if pd.isna(smoke_successes) or cand not in rec:
            continue
        rows.append(
            {
                "candidate": cand,
                "arm_mse": rec[cand]["arm_mse"],
                "gripper": rec[cand]["gripper"],
                "smoke": int(smoke_successes),
                "component": component_of(cand),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def launch_profile() -> dict:
    """Per-variant kernel-launch profile of one prediction (bank index 0, 10 profiled calls)."""
    path = EVAL / "launch_profile.json"
    prof = optional_json(path)
    if prof is None:
        return missing(path, "launch profile")
    return {
        "observation": prof["observation"],
        "profiled_calls": prof["profiled_calls"],
        "variants": prof["variants"],
    }


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


def _node_record(node_dir: Path) -> dict:
    """One Dream-RSI node: score fields that exist plus proposal metadata."""
    score = optional_json(node_dir / "score.json") or {}
    proposal = node_dir / "proposal.md"
    proposal_bytes = proposal.stat().st_size if proposal.is_file() else None
    record = {
        "cell_id": score.get("cell_id") or node_dir.name,
        "parent_id": score.get("parent_id"),
        "fail_class": score.get("fail_class"),
        "score": score.get("score"),
        "successes": score.get("successes"),
        "episodes": score.get("episodes"),
        "agent_seconds": score.get("agent_seconds"),
        "agent_cost_usd": score.get("agent_cost_usd"),
        "proposal": rel(proposal) if proposal_bytes is not None else None,
        "proposal_bytes": proposal_bytes,
    }
    return {key: value for key, value in record.items() if value is not None}


def dream_rsi_cycles(study: Path | None = None) -> dict:
    """Clean-room Dream-RSI cycles under `cycles/<id>/`.

    Reads only the files the loop writes (`tree.json`, `live_cycle_manifest.json`,
    `nodes/<cell>/score.json`, `proposal.md`) and omits fields that are absent. Returns a
    status stub when no cycle exists; never raises on a missing study directory.
    """
    root = Path(study) if study is not None else STUDY
    cycles_dir = root / "cycles"
    if not cycles_dir.is_dir():
        return {**missing(cycles_dir, "Dream-RSI cycles"), "cycles": [], "count": 0}
    cycles = []
    for cycle_dir in sorted(p for p in cycles_dir.iterdir() if p.is_dir()):
        manifest = optional_json(cycle_dir / "live_cycle_manifest.json") or {}
        tree = optional_json(cycle_dir / "tree.json") or {}
        nodes_dir = cycle_dir / "nodes"
        nodes = (
            [_node_record(p) for p in sorted(nodes_dir.iterdir()) if p.is_dir()]
            if nodes_dir.is_dir()
            else []
        )
        sources = [
            rel(path)
            for path in (cycle_dir / "tree.json", cycle_dir / "live_cycle_manifest.json")
            if path.is_file()
        ]
        world_nodes = manifest.get("world_nodes")
        if world_nodes is None and isinstance(tree.get("nodes"), list):
            world_nodes = len(tree["nodes"])
        cycle = {
            "cycle_id": manifest.get("cycle_id") or tree.get("cycle_id") or cycle_dir.name,
            "iteration": manifest.get("iteration"),
            "status": "done" if manifest else "incomplete",
            "best_score": manifest.get("best_score"),
            "probes_used": manifest.get("probes_used"),
            "decision_rounds": manifest.get("decision_rounds"),
            "effective_sequential_rounds": manifest.get("effective_sequential_rounds"),
            "world_nodes": world_nodes,
            "policy_path": manifest.get("policy_path"),
            "baked_in_beta": manifest.get("baked_in_beta"),
            "recorded_utc": manifest.get("recorded_utc"),
            "nodes": nodes,
            "sources": sources,
        }
        cycles.append({key: value for key, value in cycle.items() if value is not None})
    return {
        "status": "ok" if cycles else "empty",
        "source": rel(cycles_dir),
        "count": len(cycles),
        "cycles": cycles,
    }
