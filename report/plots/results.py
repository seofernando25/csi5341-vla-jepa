"""Reserved-trial trade-offs and per-task success (Figures 3 and 5)."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from report.plots.style import COL, COLORS, FULL, INK, MUTED, OUT


def fig_tradeoff(n: dict) -> None:
    rows = n["rows"]
    keys = ["B16", "Q8", "Q4", "C1", "C2"]
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
            "C1": (0.14, -1.25, "left"),
            "C2": (0.12, 0.45, "left"),
        }[k]
        a.text(
            r["peak_allocated_gb"] + dx,
            r["success_pct"] + dy,
            k,
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
    a.set_title("(a) Success is retained down to 1.4 GB", loc="left")
    for k in keys:
        r = rows[k]
        b.plot([r["peak_allocated_gb"]] * 2, [r["median_ms"], r["p95_ms"]], color=COLORS[k], lw=1)
        b.scatter(r["peak_allocated_gb"], r["median_ms"], s=26, color=COLORS[k], zorder=3)
        dx, dy = {
            "B16": (-0.15, 6),
            "Q8": (0.15, -2),
            "Q4": (0.15, -5),
            "C1": (0.13, -3),
            "C2": (0.12, 6),
        }[k]
        b.text(
            r["peak_allocated_gb"] + dx,
            r["median_ms"] + dy,
            k,
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
    b.set_title("(b) Latency: stock paths slower, graph engine (C2) faster", loc="left")
    fig.savefig(OUT / "tradeoff.pdf")
    plt.close(fig)


def fig_tasks(n: dict) -> None:
    rows = n["rows"]
    keys = ["B16", "Q8", "Q4", "C1", "C2"]
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
