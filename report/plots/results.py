"""Unified BF16-to-C2 ablation ladder: peak memory and latency attribution (Figure 2)."""

import matplotlib.pyplot as plt

from report.plots.style import (
    COLORS,
    FULL,
    GRID,
    INK,
    INK_2,
    MUTED,
    OUT,
    panel_label,
    tint,
)

STOCK_RUNGS = [
    ("B16c", "B16 stock (4 steps)", COLORS["B16"]),
    ("Q8", "Q8: INT8 dec.", COLORS["Q8"]),
    ("Q4", "Q4: NF4 dec.", COLORS["Q4"]),
    ("C1", "C1: NF4 + HQQ2", COLORS["C1"]),
]

GRAPH_RUNGS = [
    ("B16 graph, 4 steps", "B16 graph, 4 steps", COLORS["B16"]),
    ("NF4 graph, 4 steps", "N4: NF4 graph, 4 steps", tint(COLORS["C2"], 0.35)),
    ("B16 graph, 1 step", "B16 graph, 1 step", COLORS["B16"]),
    ("NF4 graph, 1 step, dense", "NF4 dense, 1 step", "#64748B"),
    ("NF4 graph, 1 step, packed (C2)", "C2: NF4 packed, 1 step", COLORS["C2"]),
]


def _style_barh(ax: plt.Axes, labels: list[str], xticks: list[int], xlim: tuple[float, float]):
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlim=xlim, xticks=xticks)
    ax.invert_yaxis()
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)


def _plot_latency_row(ax: plt.Axes, y: float, med: float, p95: float, color: str):
    ax.barh(y, med, height=0.52, color=tint(color, 0.25), zorder=2)
    ax.plot([med, p95], [y, y], color=color, lw=1.2, zorder=3)
    ax.scatter([p95], [y], marker="|", s=28, color=color, lw=1.1, zorder=4)
    ax.scatter([med], [y], s=18, color=color, edgecolor="white", linewidth=0.6, zorder=4)


def fig_ladder(n: dict) -> None:
    rows = n["rows"]
    ladder = n["latency_ladder"]
    ceiling = n["memory_ceiling"]
    prof = n["launch_profile"]["variants"]

    b16_peak = rows["B16"]["peak_allocated_gb"]
    h1_mem = 0.75 * b16_peak
    nf4_ceil_mem = b16_peak - ceiling["NF4"]["saved_gb"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FULL, 2.9),
        gridspec_kw={
            "height_ratios": [4, 5.4],
            "width_ratios": [1.0, 1.28],
            "hspace": 0.46,
            "wspace": 0.08,
        },
    )
    fig.subplots_adjust(left=0.205, right=0.985, top=0.86, bottom=0.145)
    a1, b1 = axes[0]
    a2, b2 = axes[1]

    # --- Left column: Peak allocated GPU memory (GB) ---
    mem_xlim = (0.0, 6.85)
    mem_xticks = [0, 1, 2, 3, 4, 5, 6]
    stock_labels = [lbl for _, lbl, _ in STOCK_RUNGS]
    graph_labels = [lbl for _, lbl, _ in GRAPH_RUNGS]

    for ax in (a1, a2):
        ax.axvline(h1_mem, color=MUTED, lw=0.75, ls=(0, (3, 2)), zorder=1)
        ax.axvline(nf4_ceil_mem, color=COLORS["Q4"], lw=0.75, ls=(0, (2, 2)), alpha=0.85, zorder=1)

    for i, (key, _, col) in enumerate(STOCK_RUNGS):
        val = rows[key]["peak_allocated_gb"]
        a1.barh(i, val, height=0.54, color=col, zorder=3)
        note = {
            "B16c": f"{val:.2f}",
            "Q8": f"{val:.2f} (-23%, fail)",
            "Q4": f"{val:.2f} (-37%)",
            "C1": f"{val:.2f} (-75%)",
        }[key]
        a1.text(val + 0.10, i, note, va="center", ha="left", fontsize=6.5, color=INK)

    _style_barh(a1, stock_labels, mem_xticks, mem_xlim)
    a1.set_ylim(3.55, -0.55)
    a1.tick_params(axis="x", labelbottom=False, length=0)
    a1.spines["bottom"].set_color(GRID)
    panel_label(a1, "a", "Peak allocated GPU memory (GB)", x=0.0, y=1.18)
    a1.text(
        0.0,
        1.03,
        "Stock path",
        transform=a1.transAxes,
        fontsize=6.5,
        fontweight="medium",
        color=INK_2,
        va="bottom",
        ha="left",
    )

    for i, (lkey, _, col) in enumerate(GRAPH_RUNGS):
        val = ladder[lkey]["peak_allocated_gb"]
        a2.barh(i, val, height=0.54, color=col, zorder=3)
        if lkey == "NF4 graph, 1 step, dense":
            note = f"{val:.2f} (dense)"
        elif lkey == "NF4 graph, 1 step, packed (C2)":
            note = f"{val:.2f} (-73%)"
        else:
            note = f"{val:.2f}"
        a2.text(val + 0.10, i, note, va="center", ha="left", fontsize=6.5, color=INK)

    # Ceiling annotations placed in the open space below row 4 (C2).
    a2.text(
        h1_mem + 0.08,
        4.92,
        f"H1 -25% ({h1_mem:.2f} GB)",
        fontsize=6.5,
        color=MUTED,
        va="center",
        ha="left",
    )
    a2.text(
        nf4_ceil_mem - 0.08,
        4.92,
        "NF4 ceiling (-37%)",
        fontsize=6.5,
        color=COLORS["Q4"],
        va="center",
        ha="right",
    )

    _style_barh(a2, graph_labels, mem_xticks, mem_xlim)
    a2.set_ylim(5.38, -0.55)
    a2.set_xlabel("Peak allocated GPU memory (GB)")
    a2.text(
        0.0,
        1.03,
        "CUDA-graph engine",
        transform=a2.transAxes,
        fontsize=6.5,
        fontweight="medium",
        color=INK_2,
        va="bottom",
        ha="left",
    )

    # --- Right column: Latency (median bar, p95 whisker) ---
    b1_xlim = (0.0, 265.0)
    b1_xticks = [0, 50, 100, 150, 200, 250]
    b1.axvspan(0, 72, color=tint(COLORS["C2"], 0.92), lw=0, zorder=0)
    b1.axvline(72, color=GRID, lw=0.7, zorder=1)

    for i, (key, _, col) in enumerate(STOCK_RUNGS):
        r = rows[key]
        _plot_latency_row(b1, i, r["median_ms"], r["p95_ms"], col)

    # Session A B16 tick overlay on B16 stock row (y=0).
    a_med, a_p95 = rows["B16"]["median_ms"], rows["B16"]["p95_ms"]
    b_med = rows["B16c"]["median_ms"]
    b1.plot([a_med, a_p95], [-0.19, -0.19], color=MUTED, lw=0.9, zorder=4)
    b1.scatter(
        [a_med], [-0.19], marker="D", s=11, color="white", edgecolor=MUTED, linewidth=0.9, zorder=5
    )

    b1.text(
        rows["B16"]["p95_ms"] + 4,
        0,
        f"A: {a_med:.1f} / B: {b_med:.1f} · {prof['B16']['kernel_launches']:,.0f} kernels, {prof['B16']['host_syncs']:.0f} syncs",
        va="center",
        ha="left",
        fontsize=6.5,
        color=INK,
    )
    b1.text(
        rows["Q8"]["p95_ms"] + 4,
        1,
        f"{rows['Q8']['median_ms']:.1f} · {prof['Q8']['kernel_launches']:,.0f}, {prof['Q8']['host_syncs']:.0f}",
        va="center",
        ha="left",
        fontsize=6.5,
        color=INK,
    )
    b1.text(
        rows["Q4"]["p95_ms"] + 4,
        2,
        f"{rows['Q4']['median_ms']:.1f} · {prof['Q4']['kernel_launches']:,.0f}, {prof['Q4']['host_syncs']:.0f}",
        va="center",
        ha="left",
        fontsize=6.5,
        color=INK,
    )
    b1.text(
        rows["C1"]["p95_ms"] + 4,
        3,
        f"{rows['C1']['median_ms']:.1f}",
        va="center",
        ha="left",
        fontsize=6.5,
        color=INK,
    )

    _style_barh(b1, stock_labels, b1_xticks, b1_xlim)
    b1.set_ylim(3.55, -0.55)
    b1.tick_params(axis="y", labelleft=False)
    panel_label(b1, "b", "Latency: median (dot) to p95 (whisker), ms", x=0.0, y=1.18)
    b1.text(
        0.0,
        1.03,
        "Stock path (shaded 0–72 ms zoomed below)",
        transform=b1.transAxes,
        fontsize=6.5,
        color=INK_2,
        va="bottom",
        ha="left",
    )

    # Bottom right: CUDA-graph engine (zoomed 0..72 ms)
    b2_xlim = (0.0, 72.0)
    b2_xticks = [0, 10, 20, 30, 40, 50, 60, 70]
    for i, (lkey, _, col) in enumerate(GRAPH_RUNGS):
        d = ladder[lkey]
        _plot_latency_row(b2, i, d["median_ms"], d["p95_ms"], col)
        txt = f"{d['median_ms']:.1f}"
        if lkey == "NF4 graph, 1 step, packed (C2)":
            txt = f"{d['median_ms']:.1f} · C2 {rows['C2']['successes']}/500"
        b2.text(d["p95_ms"] + 1.0, i, txt, va="center", ha="left", fontsize=6.5, color=INK)

    # Bracket for +6.8 ms NF4 dequant tax between 1-step B16/dense (24.8 ms) and C2 packed (31.6 ms).
    x_b16_1 = ladder["B16 graph, 1 step"]["median_ms"]
    x_c2 = ladder["NF4 graph, 1 step, packed (C2)"]["median_ms"]
    tax = x_c2 - x_b16_1
    b2.plot([x_b16_1, x_b16_1], [2.25, 3.55], color=COLORS["C2"], lw=0.75, ls=":", zorder=3)
    b2.plot([x_c2, x_c2], [3.38, 3.80], color=COLORS["C2"], lw=0.75, ls=":", zorder=3)
    b2.annotate(
        "",
        xy=(x_c2, 3.50),
        xytext=(x_b16_1, 3.50),
        arrowprops={
            "arrowstyle": "<->",
            "color": COLORS["C2"],
            "lw": 0.9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=5,
    )
    b2.text(
        x_c2 + 1.2,
        3.50,
        f"+{tax:.1f} ms NF4 dequant tax (31.6 vs 24.8)",
        va="center",
        ha="left",
        fontsize=6.5,
        color=COLORS["C2"],
        fontweight="medium",
    )

    _style_barh(b2, graph_labels, b2_xticks, b2_xlim)
    b2.set_ylim(5.38, -0.55)
    b2.tick_params(axis="y", labelleft=False)
    b2.set_xlabel("Observation-to-action latency (ms, zoomed)")
    b2.text(
        0.0,
        1.03,
        "CUDA-graph engine (zoomed 0–72 ms)",
        transform=b2.transAxes,
        fontsize=6.5,
        color=INK_2,
        va="bottom",
        ha="left",
    )
    b2.text(
        0.01,
        5.05,
        "How to read: compression cuts memory; CUDA graphs cut latency.",
        ha="left",
        va="center",
        fontsize=6.5,
        color=INK_2,
        style="italic",
    )

    fig.savefig(OUT / "ladder.pdf")
    plt.close(fig)
