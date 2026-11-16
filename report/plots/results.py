"""Execution ladder: peak memory and latency by plan id (Figure 3).

Timing only. The one-step B16 arm is a control whose success is not measured; the report must
not read it as a success arm.
"""

import matplotlib.pyplot as plt

from report.plots.style import COLORS, FULL, INK, MUTED, OUT, panel_label, tint

# (plan id, label, palette key)
LADDER = [
    ("B16", "Original, 4 flow steps", "B16"),
    ("precision-lang-int8", "INT8 decoder", "Q8"),
    ("precision-lang-nf4", "NF4 decoder", "Q4"),
    ("ladder-b16-graph-4step", "Original + graph, 4 steps", "B16"),
    ("ladder-nf4-graph-4step", "NF4 + graph, 4 steps", "Q4"),
    ("ladder-b16-graph-1step", "Original + graph, 1 step", "B16"),
    ("ladder-nf4-graph-1step-dense", "NF4 unpacked + graph, 1 step", "other"),
    ("ladder-nf4-graph-1step-packed", "NF4 packed + graph, 1 step", "C2"),
]


def fig_ladder(n: dict) -> dict | None:
    """Peak memory and latency per ladder rung; None until every rung has a benchmark cell."""
    rows, ladder = n["rows"], n["latency_ladder"]
    missing = [key for key, _, _ in LADDER if key not in ladder or key not in rows]
    if missing:
        return None

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL, 2.15),
        gridspec_kw={"wspace": 0.10, "width_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.25, right=0.985, top=0.82, bottom=0.19)
    memory, latency = axes
    labels = [label for _, label, _ in LADDER]
    positions = range(len(LADDER))
    halo = {"facecolor": "white", "alpha": 0.90, "edgecolor": "none", "pad": 0.6}

    ceiling = n.get("memory_ceiling") or {}
    b16_peak = rows["B16"]["benchmark_peak_allocated_gb"]
    target = 0.75 * b16_peak
    for ax in axes:
        ax.axvline(target, color=MUTED, lw=0.75, ls=(0, (3, 2)), zorder=1)
        if "NF4" in ceiling:
            ax.axvline(
                b16_peak - ceiling["NF4"]["saved_gb"],
                color=COLORS["Q4"],
                lw=0.75,
                ls=(0, (2, 2)),
                alpha=0.85,
                zorder=1,
            )

    for index, (key, _, palette) in enumerate(LADDER):
        value = rows[key]["benchmark_peak_allocated_gb"]
        memory.barh(index, value, height=0.54, color=COLORS[palette], zorder=3)
        memory.text(
            value + 0.08,
            index,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=6.5,
            color=INK,
            bbox=halo,
            zorder=4,
        )
    memory.set(
        xlim=(0, max(rows[key]["benchmark_peak_allocated_gb"] for key, _, _ in LADDER) * 1.22)
    )
    above = memory.get_xaxis_transform()
    memory.text(
        target + 0.06,
        1.015,
        "25% target",
        fontsize=6.5,
        color=MUTED,
        va="bottom",
        ha="left",
        transform=above,
        clip_on=False,
    )
    if "NF4" in ceiling:
        memory.text(
            b16_peak - ceiling["NF4"]["saved_gb"] - 0.06,
            1.015,
            "NF4 limit (37%)",
            fontsize=6.5,
            color=COLORS["Q4"],
            va="bottom",
            ha="right",
            transform=above,
            clip_on=False,
        )

    for index, (key, _, palette) in enumerate(LADDER):
        rung = ladder[key]
        color = COLORS[palette]
        latency.barh(index, rung["median_ms"], height=0.54, color=tint(color, 0.25), zorder=2)
        latency.plot(
            [rung["median_ms"], rung["p95_ms"]],
            [index, index],
            color=color,
            lw=1.2,
            zorder=3,
        )
        latency.scatter([rung["p95_ms"]], [index], marker="|", s=28, color=color, lw=1.1, zorder=4)
        latency.scatter(
            [rung["median_ms"]],
            [index],
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        latency.text(
            rung["p95_ms"] + 1.5,
            index,
            f"{rung['median_ms']:.1f}",
            va="center",
            ha="left",
            fontsize=6.5,
            color=INK,
        )
    latency.set(xlim=(0, max(rung["p95_ms"] for rung in ladder.values()) * 1.28))

    for ax, xlabel in (
        (memory, "Peak allocated GPU memory (GB)"),
        (latency, "Observation-to-action latency (ms)"),
    ):
        ax.set(yticks=list(positions), xlim=ax.get_xlim(), xlabel=xlabel)
        ax.set_ylim(len(LADDER) - 0.40, -0.62)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=4)
    memory.set_yticklabels(labels)
    latency.set_yticklabels([])
    panel_label(memory, "a", "Peak memory", x=0.0, y=1.12)
    panel_label(latency, "b", "Latency: median (dot) to 95th percentile", x=0.0, y=1.12)

    fig.savefig(OUT / "ladder.pdf")
    plt.close(fig)
    return {
        "status": "ok",
        "rungs": {
            key: {
                "peak_allocated_gb": rows[key]["benchmark_peak_allocated_gb"],
                "median_ms": ladder[key]["median_ms"],
                "p95_ms": ladder[key]["p95_ms"],
            }
            for key, _, _ in LADDER
        },
    }
