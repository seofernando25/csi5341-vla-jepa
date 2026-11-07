"""Build final-report figures and the number ledger from committed study evidence.

Usage: python -m report.figures  (from the repository root; CPU only, no model loading)

Inputs: studies/evaluation (B16/Q8/Q4/S500 study) and studies/compression (compression study).
Outputs: report/figures/*.pdf and report/data/numbers.json. Every number quoted in
report/main.tex is printed to numbers.json.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluation.analysis import wilson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
DATA = ROOT / "report" / "data"
EVAL = ROOT / "studies" / "evaluation"
COMP = ROOT / "studies" / "compression"
RUNS = {
    "B16": "20260924T171416675676Z-B16-rollout",
    "Q8": "20260924T174749793177Z-Q8-rollout",
    "Q4": "20260924T183120067911Z-Q4-rollout",
}
FINAL = {  # compression-study reserved configurations, in report order
    "C1": "deploy-nf4-vision-text-channels-restricted",
    "C2": "deploy-hqq2-vision-action-restricted",
    "C3": "deploy-hqq2-vision-action",
}
GB = 1e9

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
COLORS = {
    "B16": "#1f2933",
    "Q8": "#d97706",
    "Q4": "#2563eb",
    "C1": "#059669",
    "C2": "#7c3aed",
    "C3": "#db2777",
    "language": "#2563eb",
    "vision": "#059669",
    "action": "#d97706",
    "joint": "#7c3aed",
    "other": "#9aa5b1",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
COL, FULL = 3.45, 7.1  # IEEE single and double column widths, inches


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
    for label, cand in [("B16c", "B16"), *FINAL.items()]:
        r, res = comp.loc[cand], reserved[cand]
        rows[label] = {
            "study": "compression",
            "candidate": cand,
            "successes": int(res["successes"]),
            "per_task": [int(res["task_successes"][str(t)]) for t in range(10)],
            "peak_allocated_gb": r.benchmark_peak_allocated_bytes / GB,
            "weights_gb": r.packed_archive_bytes / GB,
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


def divergence() -> pd.DataFrame:
    comp = pd.read_csv(COMP / "summary.csv")
    rows = []
    for cand, smoke in zip(comp.candidate, comp.smoke_successes, strict=True):
        path = COMP / "measurements" / cand / "screen" / "observations.json"
        if pd.isna(smoke) or not path.exists():
            continue
        obs = pd.DataFrame(load_json(path))
        rows.append(
            {
                "candidate": cand,
                "arm_mse": obs.physical_arm_change_mse.mean(),
                "gripper": obs.gripper_disagreement.mean(),
                "smoke": int(smoke),
                "component": component_of(cand),
            }
        )
    return pd.DataFrame(rows)


def label_of(key: str) -> str:
    return {
        "B16": "B16",
        "B16c": "B16",
        "Q8": "Q8",
        "Q4": "Q4",
        "C1": "C1",
        "C2": "C2",
        "C3": "C3",
    }[key]


def fig_tradeoff(n: dict) -> None:
    rows = n["rows"]
    keys = ["B16", "Q8", "Q4", "C3", "C2", "C1"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.25), gridspec_kw={"wspace": 0.28})
    base = rows["B16"]
    xmax = 0.75 * base["peak_allocated_gb"]
    a.axvspan(0, xmax, ymin=0, ymax=1, color="#ecfdf5", zorder=0, lw=0)
    a.axhline(base["success_pct"] - 5, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    a.axvline(xmax, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    a.text(
        xmax - 0.08,
        89.4,
        "target: ≥25% less memory,\n≤5 pp success loss",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#047857",
    )
    for k in keys:
        r = rows[k]
        lo, hi = r["wilson_pct"]
        a.errorbar(
            r["peak_allocated_gb"],
            r["success_pct"],
            yerr=[[r["success_pct"] - lo], [hi - r["success_pct"]]],
            fmt="o",
            ms=5,
            color=COLORS[k],
            ecolor=COLORS[k],
            elinewidth=0.9,
            capsize=1.8,
            zorder=3,
        )
        dx, dy, ha = {
            "B16": (-0.12, 0.5, "right"),
            "Q8": (0.12, -1.3, "left"),
            "Q4": (0.12, 0.35, "left"),
            "C1": (0.14, -1.6, "left"),
            "C2": (0.14, 0.25, "left"),
            "C3": (0.14, -1.25, "left"),
        }[k]
        a.text(
            r["peak_allocated_gb"] + dx,
            r["success_pct"] + dy,
            label_of(k),
            ha=ha,
            fontsize=7,
            color=COLORS[k],
            weight="bold",
        )
    a.set(
        xlim=(0, 6.1),
        ylim=(88, 100),
        xlabel="Peak allocated GPU memory (GB)",
        ylabel="LIBERO-Spatial success (%, 500 trials)",
    )
    a.set_title("(a) Success is retained down to 0.94 GB", loc="left")
    for k in keys:
        r = rows[k]
        b.plot([r["peak_allocated_gb"]] * 2, [r["median_ms"], r["p95_ms"]], color=COLORS[k], lw=1)
        b.scatter(r["peak_allocated_gb"], r["median_ms"], s=26, color=COLORS[k], zorder=3)
        dx, dy = {
            "B16": (-0.15, 6),
            "Q8": (0.15, -2),
            "Q4": (0.15, -5),
            "C1": (-0.07, -12),
            "C2": (0.12, 6),
            "C3": (0.13, -3),
        }[k]
        b.text(
            r["peak_allocated_gb"] + dx,
            r["median_ms"] + dy,
            label_of(k),
            ha="right" if dx < 0 else "left",
            fontsize=7,
            color=COLORS[k],
            weight="bold",
        )
    b.set(
        xlim=(0, 6.1),
        ylim=(0, 175),
        xlabel="Peak allocated GPU memory (GB)",
        ylabel="Observation-to-action latency (ms)",
    )
    b.set_title("(b) Every compressed policy is slower (median, p95 tick)", loc="left")
    fig.savefig(OUT / "tradeoff.pdf")
    plt.close(fig)


def fig_budget(census: dict, ceiling: dict, n: dict) -> None:
    params = census["parameters"]
    names = {
        "language decoder linear": ("Language decoder linear", "#2563eb"),
        "vision encoder and mergers": ("Vision encoder + mergers", "#059669"),
        "V-JEPA 2 encoder (world model)": ("V-JEPA 2 encoder (train only)", "#9aa5b1"),
        "language embeddings and norms": ("Embeddings + norms", "#93c5fd"),
        "latent predictor (world model)": ("Latent predictor (train only)", "#cbd2d9"),
        "action head": ("Action head", "#d97706"),
    }
    total = sum(params.values())
    fig, (a, b) = plt.subplots(
        2, 1, figsize=(COL, 2.75), gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.78}
    )
    keys = list(names)
    y = np.arange(len(keys))
    vals = [2 * params[k] / GB for k in keys]
    a.barh(y, vals, color=[names[k][1] for k in keys], height=0.66)
    for yy, k, v in zip(y, keys, vals, strict=True):
        a.text(
            v + 0.04,
            yy,
            f"{params[k] / 1e9:.2f}B params ({100 * params[k] / total:.0f}%)",
            va="center",
            fontsize=6.2,
            color=INK,
        )
    a.set(
        yticks=y,
        yticklabels=[names[k][0] for k in keys],
        xlim=(0, 4.2),
        xlabel="BF16 weight storage (GB)",
    )
    a.invert_yaxis()
    a.grid(axis="y", visible=False)
    a.set_title("(a) Where B16 weights live", loc="left")
    rows = n["rows"]
    measured = [
        100 * (1 - rows[k]["peak_allocated_gb"] / rows["B16"]["peak_allocated_gb"])
        for k in ["Q8", "Q4"]
    ]
    ideal = [ceiling["INT8"]["ceiling_pct"], ceiling["NF4"]["ceiling_pct"]]
    y2 = np.arange(2)
    b.barh(y2 - 0.19, ideal, height=0.36, color="#e4e7eb", label="ideal: decoder weights only")
    b.barh(
        y2 + 0.19,
        measured,
        height=0.36,
        color=[COLORS["Q8"], COLORS["Q4"]],
        label="measured peak memory",
    )
    for yy, i, m in zip(y2, ideal, measured, strict=True):
        b.text(i + 0.5, yy - 0.19, f"{i:.1f}%", va="center", fontsize=6.2, color=MUTED)
        b.text(m + 0.5, yy + 0.19, f"{m:.1f}%", va="center", fontsize=6.2, color=INK)
    b.axvline(25, color=INK, lw=0.7, ls=(0, (3, 2)))
    b.text(24.5, -0.72, "proposal target", fontsize=6, color=INK, va="center", ha="right")
    b.set(
        yticks=y2,
        yticklabels=["Q8 (INT8)", "Q4 (NF4)"],
        xlim=(0, 60),
        ylim=(1.6, -0.95),
        xlabel="Peak-memory reduction vs B16 (%)",
    )
    b.grid(axis="y", visible=False)
    b.legend(loc="upper right", handlelength=1, borderaxespad=0.1, fontsize=6)
    b.set_title("(b) Measured savings sit at the weight-only ceiling", loc="left")
    fig.savefig(OUT / "budget.pdf")
    plt.close(fig)


def smoke(comp: pd.DataFrame, name: str) -> float:
    v = comp.loc[name, "smoke_successes"] if name in comp.index else np.nan
    return float(v)


def fig_sensitivity() -> dict:
    comp = pd.read_csv(COMP / "summary.csv").set_index("candidate")
    base = smoke(comp, "B16")
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.75), gridspec_kw={"wspace": 0.32})
    # (a) weight bit width; best of tested variants at each width, all variants shown faintly.
    bits = {
        "language": {
            8: ["text-bnb8"],
            4: ["text-bnb4", "text-hqq4-axis1-pytorch"],
            3: ["text-hqq3-axis0-pytorch", "text-hqq3-axis1-pytorch"],
            2: ["text-hqq2-axis0-pytorch", "text-hqq2-axis1-pytorch", "text-hqq2-axis1-gemlite"],
        },
        "vision": {
            8: ["vision_blocks-bnb8"],
            4: ["vision_blocks-bnb4", "vision_blocks-hqq4-axis1-pytorch"],
            3: ["vision_blocks-hqq3-axis0-pytorch", "vision_blocks-hqq3-axis1-pytorch"],
            2: [
                "vision_blocks-hqq2-axis0-pytorch",
                "vision_blocks-hqq2-axis1-pytorch",
                "vision_blocks-hqq2-axis1-gemlite",
            ],
        },
        "action": {
            8: ["action_mlp-bnb8"],
            4: ["action_mlp-bnb4", "action_mlp-hqq4-axis1-pytorch"],
            3: ["action_mlp-hqq3-axis0-pytorch", "action_mlp-hqq3-axis1-pytorch"],
            2: [
                "action_mlp-hqq2-axis0-pytorch",
                "action_mlp-hqq2-axis1-pytorch",
                "action_mlp-hqq2-axis1-gemlite",
            ],
        },
    }
    ledger: dict = {"bits": {}, "structure": {}}
    a = axes[0]
    offsets = {"language": -0.12, "vision": 0.0, "action": 0.12}
    for comp_name, table in bits.items():
        xs, best = [16], [base]
        for b, names in table.items():
            vals = [smoke(comp, nm) for nm in names if not np.isnan(smoke(comp, nm))]
            a.scatter(
                [b + offsets[comp_name]] * len(vals),
                vals,
                s=7,
                color=COLORS[comp_name],
                alpha=0.35,
                lw=0,
            )
            xs.append(b)
            best.append(max(vals))
        mk, ms, fill = {
            "language": ("o", 3.2, COLORS["language"]),
            "vision": ("s", 5, "white"),
            "action": ("o", 2.6, COLORS["action"]),
        }[comp_name]
        a.plot(
            np.array(xs) + offsets[comp_name],
            best,
            "-",
            marker=mk,
            ms=ms,
            mfc=fill,
            lw=1.1,
            color=COLORS[comp_name],
            label={
                "language": "language decoder",
                "vision": "vision blocks",
                "action": "action MLP",
            }[comp_name],
        )
        ledger["bits"][comp_name] = dict(zip(xs, best, strict=True))
    a.set_xscale("log", base=2)
    a.set(
        xticks=[2, 3, 4, 8, 16],
        xticklabels=["2", "3", "4", "8", "16"],
        ylim=(-0.5, 10.5),
        xlabel="Weight bits",
        ylabel="Successes / 10 trials",
    )
    a.minorticks_off()
    a.set_title("(a) Bit width", loc="left")
    a.legend(loc="lower right", handlelength=1.2, borderaxespad=0.1, fontsize=6)
    # (b–d) structural removal per module.
    struct_series = {
        "language": {
            "MLP channels": [
                (0.125, "text_mlp-channels-125"),
                (0.25, "text_mlp-channels-250"),
                (0.5, "text_mlp-channels-500"),
                (0.75, "text_mlp-channels-750"),
            ],
            "KV heads": [
                (0.125, "text-heads-125"),
                (0.25, "text-heads-250"),
                (0.5, "text-heads-500"),
            ],
            "layers": [
                (1 / 28, "text-layers-1"),
                (4 / 28, "text-layers-4"),
                (8 / 28, "text-layers-8"),
                (20 / 28, "text-layers-20"),
            ],
        },
        "vision": {
            "MLP channels": [
                (0.125, "vision_mlp-channels-125"),
                (0.5, "vision_mlp-channels-500"),
                (0.75, "vision_mlp-channels-750"),
            ],
            "heads": [
                (0.125, "vision_attention-heads-125"),
                (0.5, "vision_attention-heads-500"),
                (0.75, "vision_attention-heads-750"),
            ],
            "layers": [
                (1 / 24, "vision-layers-1"),
                (8 / 24, "vision-layers-8"),
                (12 / 24, "vision-layers-12"),
                (16 / 24, "vision-layers-16"),
            ],
        },
        "action": {
            "MLP channels": [(0.125, "action_mlp-channels-125"), (0.75, "action_mlp-channels-750")],
            "heads": [(0.125, "action_attention-heads-125"), (0.75, "action_attention-heads-750")],
            "layers": [
                (1 / 16, "action-layers-1"),
                (8 / 16, "action-layers-8"),
                (12 / 16, "action-layers-12"),
            ],
        },
    }
    styles = {
        "MLP channels": ("-", "o"),
        "KV heads": ("--", "s"),
        "heads": ("--", "s"),
        "layers": (":", "^"),
    }
    hollow = {"KV heads", "heads"}
    for ax, (comp_name, series) in zip(axes[1:], struct_series.items(), strict=True):
        ledger["structure"][comp_name] = {}
        for name, pts in series.items():
            xs = [0.0] + [x for x, _ in pts]
            ys = [base] + [smoke(comp, c) for _, c in pts]
            ls, mk = styles[name]
            ax.plot(
                np.array(xs) * 100,
                ys,
                ls=ls,
                marker=mk,
                ms=4.6 if name in hollow else 3,
                mfc="white" if name in hollow else COLORS[comp_name],
                lw=1.1,
                color=COLORS[comp_name],
                label=name,
            )
            ledger["structure"][comp_name][name] = dict(
                zip([round(x, 3) for x in xs], ys, strict=True)
            )
        ax.set(xlim=(-3, 78), ylim=(-0.5, 10.5), xlabel="Fraction removed (%)")
        ax.set_title(
            f"({'bcd'[list(struct_series).index(comp_name)]}) {comp_name.capitalize()}", loc="left"
        )
        ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.1, fontsize=6)
    for ax in axes:
        ax.axhline(base, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=0)
    fig.savefig(OUT / "sensitivity.pdf")
    plt.close(fig)
    return ledger


def fig_divergence(div: pd.DataFrame) -> dict:
    floor = 2e-8
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    for comp_name in ["language", "vision", "action", "joint", "other"]:
        d = div[div.component == comp_name]
        x = d.arm_mse.clip(lower=floor)
        y = d.smoke + rng.uniform(-0.18, 0.18, len(d))
        ax.scatter(x, y, s=9, color=COLORS[comp_name], alpha=0.75, lw=0, label=comp_name)
    rho, p = spearmanr(div.arm_mse, div.smoke)
    ax.set_xscale("log")
    ax.set(
        ylim=(-0.6, 10.2),
        xlabel="Offline arm-action change vs B16 (MSE, 80 fixed frames)",
        ylabel="Successes / 10 trials",
    )
    ax.axvline(floor * 1.6, color=GRID, lw=0.6)
    ax.text(floor * 1.1, 0.0, "identical\nactions", fontsize=5.8, color=MUTED, ha="left")
    ax.text(
        0.98,
        0.97,
        f"Spearman ρ = {rho:.2f}, n = {len(div)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color=INK,
    )
    ax.legend(
        loc="lower left",
        ncol=5,
        handletextpad=0.1,
        columnspacing=0.6,
        fontsize=6,
        bbox_to_anchor=(0, 1.0, 1, 0.1),
        mode="expand",
        borderaxespad=0.2,
    )
    fig.savefig(OUT / "divergence.pdf")
    plt.close(fig)
    comp = pd.read_csv(COMP / "summary.csv").set_index("candidate")
    dev = div.join(comp.development_successes, on="candidate").dropna(
        subset=["development_successes"]
    )
    rho_dev, p_dev = spearmanr(dev.arm_mse, dev.development_successes)
    clean = div[div.arm_mse < 1e-4]
    noisy = div[div.arm_mse >= 1e-3]
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n": len(div),
        "rho_gripper": float(spearmanr(div.gripper, div.smoke)[0]),
        "development": {"n": len(dev), "spearman_rho": float(rho_dev), "p_value": float(p_dev)},
        "below_1e-4": {
            "n": len(clean),
            "mean_smoke": float(clean.smoke.mean()),
            "share_ge7": float((clean.smoke >= 7).mean()),
        },
        "above_1e-3": {
            "n": len(noisy),
            "mean_smoke": float(noisy.smoke.mean()),
            "share_ge7": float((noisy.smoke >= 7).mean()),
        },
    }


def fig_tasks(n: dict) -> None:
    rows = n["rows"]
    keys = ["B16", "Q8", "Q4", "C1", "C2", "C3"]
    mat = np.array([rows[k]["per_task"] for k in keys]).T * 2  # percent of 50 trials
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("s", ["#b91c1c", "#fde68a", "#f8fafc"])
    ax.imshow(mat.T, aspect="auto", cmap=cmap, vmin=55, vmax=100)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                i,
                j,
                f"{mat[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=6.4,
                color="white" if mat[i, j] < 75 else INK,
            )
    ax.set(
        xticks=range(10),
        xticklabels=[str(t) for t in range(10)],
        yticks=range(len(keys)),
        yticklabels=keys,
        xlabel="LIBERO-Spatial task",
    )
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(OUT / "tasks.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    n = collect()
    census = parameter_census()
    n["census"] = census
    n["memory_ceiling"] = memory_ceiling(census, n["rows"]["B16"]["peak_allocated_gb"])
    fig_tradeoff(n)
    fig_budget(census, n["memory_ceiling"], n)
    n["sensitivity"] = fig_sensitivity()
    n["divergence"] = fig_divergence(divergence())
    fig_tasks(n)
    (DATA / "numbers.json").write_text(json.dumps(n, indent=2, default=float) + "\n")
    print(f"Wrote {len(list(OUT.glob('*.pdf')))} figures and {DATA / 'numbers.json'}")


if __name__ == "__main__":
    main()
