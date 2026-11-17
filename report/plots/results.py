"""Execution ladder: peak memory against latency by plan id.

Timing only. The one-step B16 arm is a control whose success is not measured; the report must
not read it as a success arm.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from report.plots.style import COL, COLORS, INK, INK_2, MUTED, OUT, style_axes, text_ink, tint

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
    """Peak memory against latency, one point per rung; None until every rung has a cell.

    One column: the reference sits top right, the goal is the shaded bottom-left corner (smaller
    and faster than the reference), and the four CUDA-graph rungs share one label because they
    overlap at this scale.
    """
    rows, ladder = n["rows"], n["latency_ladder"]
    missing = [key for key, _, _ in LADDER if key not in ladder or key not in rows]
    if missing:
        return None

    fig, ax = plt.subplots(figsize=(COL, 2.2))
    fig.subplots_adjust(left=0.12, right=0.975, top=0.97, bottom=0.18)
    peak = {key: rows[key]["benchmark_peak_allocated_gb"] for key, _, _ in LADDER}
    b16_peak, b16_ms = peak["B16"], ladder["B16"]["median_ms"]
    target = 0.75 * b16_peak
    x_max = max(rung["p95_ms"] for rung in ladder.values()) * 1.12

    ax.add_patch(
        Rectangle((0, 0), b16_ms, target, facecolor=tint(COLORS["C2"], 0.88), lw=0, zorder=0)
    )
    ax.text(3, 0.25, "smaller and faster", fontsize=6.5, color=INK_2, va="bottom", ha="left")
    ax.axhline(target, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(
        b16_ms + 30,
        target - 0.12,
        "25% memory target",
        fontsize=6.5,
        color=INK_2,
        ha="left",
        va="top",
    )

    for key, _, palette in LADDER:
        rung, color = ladder[key], COLORS[palette]
        ax.plot(
            [rung["median_ms"], rung["p95_ms"]],
            [peak[key]] * 2,
            color=tint(color, 0.35),
            lw=1.0,
            zorder=2,
            solid_capstyle="round",
        )
        ax.scatter(
            [rung["median_ms"]],
            [peak[key]],
            s=26,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )

    def label(key: str, text: str, dx: float, dy: float, ha: str = "left", **kw) -> None:
        ax.annotate(
            text,
            (ladder[key]["median_ms"], peak[key]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.8,
            ha=ha,
            va="center",
            color=kw.pop("color", INK),
            annotation_clip=False,
            **kw,
        )

    label("B16", "original", 9, 0, fontweight="medium")
    label("precision-lang-int8", "INT8 decoder", 0, 9, ha="center")
    label("precision-lang-nf4", "NF4 decoder", 0, -9, ha="center")
    label("ladder-nf4-graph-4step", "CUDA graph (4 variants)", 2, -11)
    label(
        "ladder-nf4-graph-1step-packed",
        "packed NF4 + graph",
        22,
        0,
        color=text_ink(COLORS["C2"]),
        fontweight="bold",
    )
    ax.annotate(
        "",
        xy=(
            ladder["ladder-nf4-graph-1step-packed"]["median_ms"] + 2,
            peak["ladder-nf4-graph-1step-packed"] + 0.25,
        ),
        xytext=(b16_ms - 2, b16_peak - 0.3),
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["C2"],
            "lw": 0.8,
            "connectionstyle": "arc3,rad=0.25",
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )

    ax.set(xlim=(0, x_max), ylim=(0, 6.6))
    style_axes(
        ax,
        ylabel="Peak GPU memory (GB)",
        xlabel="Latency per action (ms), median to p95",
        yticks=[0, 2, 4, 6],
    )
    ax.xaxis.grid(True)

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
