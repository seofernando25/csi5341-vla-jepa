"""Parameter budget and the weight-only memory ceiling of decoder quantization (Figure 2)."""

import matplotlib.pyplot as plt
import numpy as np

from report.plots.data import GB
from report.plots.style import COL, COLORS, INK, MUTED, OUT


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
