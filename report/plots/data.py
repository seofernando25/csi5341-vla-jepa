"""Report inputs read from committed evidence under studies/, plus derived statistics."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pandas as pd

from evaluation.analysis import paired_difference, wilson
from evaluation.common import ROOT

DATA = ROOT / "report" / "data"
EVAL = ROOT / "studies" / "evaluation"
COMP = ROOT / "studies" / "compression"
RUNS = {
    "B16": "20260924T171416675676Z-B16-rollout",
    "Q8": "20260924T174749793177Z-Q8-rollout",
    "Q4": "20260924T183120067911Z-Q4-rollout",
}
FINAL = {  # compression-study reserved configurations, in report order
    "C1": "deploy-hqq2-vision-action",
    "C2": "fast-nf4-all-solver1-packed",  # NF4 everywhere, full vocabulary, graph engine
}
EXTRA = {  # attribution arms: reported in the text, not plotted
    "N4": "fast-nf4-all-packed",  # C2 with four flow steps
    "B16g": "fast-b16-graph",  # B16 through the graph engine
}
MEASURED = {"C2", *EXTRA}  # reserved results computed from episodes.csv, not the milestone file
LADDER = {  # latency attribution, benchmark phase only, in report order
    "B16 stock": "B16",
    "B16 graph, 4 steps": "fast-b16-graph",
    "B16 graph, 1 step": "fast-b16-graph-solver1",
    "NF4 graph, 4 steps": "fast-nf4-all-packed",
    "NF4 graph, 1 step, dense": "fast-nf4-all-solver1",
    "NF4 graph, 1 step, packed (C2)": "fast-nf4-all-solver1-packed",
}
GB = 1e9


def load_json(path: Path):
    return json.loads(path.read_text())


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


def measured_reserved(candidate: str) -> dict:
    """Reserved-trial summary of a candidate from its committed episodes, paired with B16."""
    base = pd.read_csv(COMP / "measurements" / "B16" / "final" / "episodes.csv")
    other = pd.read_csv(COMP / "measurements" / candidate / "final" / "episodes.csv")
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
    """Median latency and peak memory per execution variant, from benchmark-phase summaries."""
    return {
        label: {
            "median_ms": float(comp.loc[cand, "benchmark_pipeline_median_ms"]),
            "p95_ms": float(comp.loc[cand, "benchmark_pipeline_p95_ms"]),
            "peak_allocated_gb": float(comp.loc[cand, "benchmark_peak_allocated_bytes"]) / GB,
        }
        for label, cand in LADDER.items()
    }


def collect() -> dict:
    summary = load_json(EVAL / "analysis" / "summary.json")["results"]
    t1 = pd.read_csv(EVAL / "analysis" / "T1_summary.csv").set_index("variant")
    comp = pd.read_csv(COMP / "summary.csv").set_index("candidate")
    reserved = load_json(COMP / "reserved_confirmation_milestone.json")["candidates"]
    base_eps = episodes("B16")
    rows = {}
    for variant in ["B16", "Q8", "Q4", "S500"]:
        r = summary[variant]
        rows[variant] = {
            "study": "quantization",
            "successes": int(r["success"]["successes"]),
            "per_task": [int(r["success"]["per_task"][str(t)]["successes"]) for t in range(10)],
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
    for variant in ["Q8", "Q4"]:
        rows[variant]["transitions"] = paired(base_eps, episodes(variant))
    for label, cand in [("B16c", "B16"), *FINAL.items(), *EXTRA.items()]:
        r = comp.loc[cand]
        res = measured_reserved(cand) if label in MEASURED else reserved[cand]
        rows[label] = {
            "study": "compression",
            "candidate": cand,
            "successes": int(res["successes"]),
            "per_task": [int(res["task_successes"][str(t)]) for t in range(10)],
            "peak_allocated_gb": r.benchmark_peak_allocated_bytes / GB,
            "weights_gb": None if pd.isna(r.packed_archive_bytes) else r.packed_archive_bytes / GB,
            "median_ms": r.benchmark_pipeline_median_ms,
            "p95_ms": r.benchmark_pipeline_p95_ms,
        }
        if "paired_vs_B16" in res:
            p = res["paired_vs_B16"]
            rows[label]["paired_pp"] = [p["difference_pp"], *p["ci95_pp"]]
            rows[label]["transitions"] = p["transitions"]
    for row in rows.values():
        row["success_pct"] = row["successes"] / 5
        row["wilson_pct"] = [100 * v for v in wilson(row["successes"], 500)]
    # Episode-level reproducibility of the B16 control across the two studies.
    comp_b16 = pd.read_csv(COMP / "measurements" / "B16" / "final" / "episodes.csv")
    m = base_eps.merge(comp_b16, on=["task_id", "initial_state_hash"])
    failures = base_eps[base_eps.success == 0]
    return {
        "rows": rows,
        "latency_ladder": latency_ladder(comp),
        "b16_repeat_agreement": float((m.success_x == m.success_y).mean()),
        "b16_failures_at_step_cap": int((failures.steps == failures.steps.max()).sum()),
        "b16_step_cap": int(base_eps.steps.max()),
        "b16_task5_failures": int((failures.task_id == 5).sum()),
    }


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
    return float(comp.loc[name, "smoke_successes"]) if name in comp.index else math.nan


def divergence() -> pd.DataFrame:
    """Per-candidate offline arm change vs smoke success, from the compact divergence record."""
    comp = pd.read_csv(COMP / "summary.csv")
    rec = {r["candidate"]: r for r in load_json(COMP / "divergence.json")["rows"]}
    rows = []
    for cand, smoke in zip(comp.candidate, comp.smoke_successes, strict=True):
        if pd.isna(smoke) or cand not in rec:
            continue
        rows.append(
            {
                "candidate": cand,
                "arm_mse": rec[cand]["arm_mse"],
                "gripper": rec[cand]["gripper"],
                "smoke": int(smoke),
                "component": component_of(cand),
            }
        )
    return pd.DataFrame(rows)


def dream_rsi_stage2() -> dict:
    """Dream-RSI stage-2 development evidence for the smallest passing candidate (search only)."""
    from rsi.gate import DEV_GATE  # local: keep the plots build free of the RSI import graph

    cand = "s2-c2-t10-v25"
    dev_path = COMP / "measurements" / cand / "development" / "run.json"
    size_path = COMP / "measurements" / cand / "size" / "run.json"
    dev, size = load_json(dev_path), load_json(size_path)
    return {
        "candidate": cand,
        "phase": "development",
        "development_successes": int(dev["summary"]["successes"]),
        "development_episodes": int(dev["summary"]["episodes"]),
        "gate": DEV_GATE,
        "resident_bytes": int(size["summary"]["resident_tensor_storage_bytes"]),
        "gpu": dev["environment"]["gpu"],
        "sources": [str(dev_path.relative_to(ROOT)), str(size_path.relative_to(ROOT))],
    }


def launch_profile() -> dict:
    """Per-variant kernel-launch profile of one prediction (bank index 0, 10 profiled calls)."""
    prof = load_json(EVAL / "launch_profile.json")
    return {
        "observation": prof["observation"],
        "profiled_calls": prof["profiled_calls"],
        "variants": prof["variants"],
    }


def recipe_funnel() -> dict:
    """Compression recipe attrition across screening, pilot, dev, and held-out phases."""
    counts = load_json(COMP / "final_benchmark_milestone.json")["phase_counts"]
    screened = int(counts["screen"]["completed"])
    failed = int(counts["screen"]["failed"])
    return {
        "planned": screened + failed,
        "screened": screened,
        "screen_failed": failed,
        "pilot": int(counts["smoke"]["completed"]),
        "development": int(counts["development"]["completed"]),
        "final": int(counts["final"]["completed"]),
    }


def sensitivity_dev_failures() -> dict:
    """100-trial development failures (<77/100) from the compression summary table."""
    comp = pd.read_csv(COMP / "summary.csv").set_index("candidate")
    mapping = {
        "action_heads_75": "action_attention-heads-750",
        "merger_75": "merger-channels-750",
        "combined_channels": "deploy-nf4-channel-boundaries-restricted",
        "combined_depth": "deploy-nf4-depth-boundaries-restricted",
        "b16_dev": "B16",
    }
    return {key: int(comp.loc[cand, "development_successes"]) for key, cand in mapping.items()}


def recovery_dev() -> dict:
    """SmolVLM four-layer RGB recovery vs B16 on 100 development trials."""
    d = load_json(ROOT / "studies" / "recovery" / "diagnostics" / "expanded_development_pair.json")
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


def _resident_bytes(candidate: str, fallback: float) -> int:
    size_path = COMP / "measurements" / candidate / "size" / "run.json"
    if size_path.exists():
        return int(load_json(size_path)["summary"]["resident_tensor_storage_bytes"])
    return int(fallback)


def dream_rsi_rounds() -> dict:
    """All committed Dream-RSI live rounds and promoted development candidates."""
    round_files = sorted(
        (ROOT / "studies" / "dream-rsi").glob("live_round*.json"),
        key=lambda p: (
            0,
            int(p.stem.replace("live_round", ""))
            if p.stem.replace("live_round", "").isdigit()
            else 999,
            p.stem,
        ),
    )
    by_round: dict = {}
    promoted: dict = {}
    for path in round_files:
        d = load_json(path)
        rid = str(d["round_id"])
        promos_raw = [
            *d.get("development_promotions", {}).get("selected", []),
            *d.get("development_promotions", {}).get("control", []),
        ]
        seen_promo: dict[str, dict] = {}
        for item in promos_raw:
            cand = str(item["candidate"])
            if cand in seen_promo:
                continue
            succ = int(item["successes"])
            ep = int(item["episodes"])
            res_b = _resident_bytes(cand, item["resident_bytes"])
            entry = {
                "round_id": rid,
                "candidate": cand,
                "successes": succ,
                "episodes": ep,
                "wilson_pct": [100 * v for v in wilson(succ, ep)],
                "resident_bytes": res_b,
                "resident_mb": res_b / 1e6,
                "machine": str(item.get("machine", d.get("machine", ""))),
            }
            seen_promo[cand] = entry
            promoted[cand] = entry
        smoke_records = d.get("world_records") or [
            *d.get("selected", {}).get("results", []),
            *d.get("control", {}).get("results", []),
        ]
        seen_smoke: dict[str, int] = {}
        smoke_pass = 0
        smoke_gate = int(d.get("smoke_gate", 8))
        for rec in smoke_records:
            cand = str(rec["candidate"])
            if cand in seen_smoke:
                continue
            s_succ = int(rec.get("smoke_successes", rec.get("successes", 0)))
            res_b = _resident_bytes(cand, rec["resident_bytes"])
            seen_smoke[cand] = res_b
            if s_succ >= smoke_gate:
                smoke_pass += 1
        round_entry: dict = {
            "round_id": rid,
            "machine": str(d.get("machine", "")),
            "smoke_probes": len(seen_smoke),
            "smoke_passed": smoke_pass,
            "promotions": list(seen_promo.values()),
            "source": str(path.relative_to(ROOT)),
        }
        if not seen_promo and seen_smoke:
            mbs = [b / 1e6 for b in seen_smoke.values()]
            round_entry["failed_resident_mb_range"] = [min(mbs), max(mbs)]
        by_round[rid] = round_entry
    out: dict = {"by_round": by_round, "promoted": promoted}
    if "r2" in by_round and "failed_resident_mb_range" in by_round["r2"]:
        out["r2_cliff_mb"] = by_round["r2"]["failed_resident_mb_range"]
    return out


RESTRICTED_FINAL = (
    "deploy-nf4-vision-text-channels-restricted",
    "deploy-hqq2-vision-action-restricted",
)


def restricted_vocab_final() -> dict:
    """500-trial held-out evaluations of the two restricted-vocabulary recipes."""
    reserved = load_json(COMP / "reserved_confirmation_milestone.json")["candidates"]
    cal = load_json(COMP / "calibration.json")
    vocab_size = int(cal["token_count"])
    out: dict = {}
    for cand in RESTRICTED_FINAL:
        res = reserved[cand]
        succ = int(res["successes"])
        trials = int(res["episodes"])
        final_path = COMP / "measurements" / cand / "final" / "run.json"
        bench_path = COMP / "measurements" / cand / "benchmark" / "run.json"
        size_path = COMP / "measurements" / cand / "size" / "run.json"
        recipe_path = COMP / "recipes" / f"{cand}.json"

        final_run = load_json(final_path)
        bench_run = load_json(bench_path)
        size_run = load_json(size_path)
        recipe = load_json(recipe_path)

        bench_sum = bench_run["summary"]
        resident_bytes = int(size_run["summary"]["resident_tensor_storage_bytes"])
        peak_alloc_bytes = max(int(p["allocated_bytes"]) for p in bench_sum["memory_peaks"])
        uses_compact = any(c.get("method") == "compact_embedding" for c in recipe["changes"])

        out[cand] = {
            "candidate": cand,
            "successes": succ,
            "trials": trials,
            "success_pct": 100.0 * succ / trials,
            "wilson_pct": [100.0 * v for v in wilson(succ, trials)],
            "resident_bytes": resident_bytes,
            "resident_mb": resident_bytes / 1e6,
            "peak_allocated_gb": peak_alloc_bytes / GB,
            "median_ms": float(bench_sum["pipeline"]["median_ms"]),
            "p95_ms": float(bench_sum["pipeline"]["p95_ms"]),
            "vocabulary_size": vocab_size if uses_compact else None,
            "gpu": str(final_run["environment"]["gpu"]),
            "sources": [
                "studies/compression/reserved_confirmation_milestone.json",
                "studies/compression/calibration.json",
                str(recipe_path.relative_to(ROOT)),
                str(final_path.relative_to(ROOT)),
                str(bench_path.relative_to(ROOT)),
                str(size_path.relative_to(ROOT)),
            ],
        }
    return out
