"""Reserved-500 headline figure: peak memory against the paired success difference.

Every cell with a 500-episode reserved result: the reference, the two language-precision cells,
the packed-NF4 graph cell and the four search winners. The x axis is peak allocated GPU memory:
the benchmark phase for recipe cells, and the evaluator's resident-plus-peak measure for solution
cells, which is the largest figure seen before, during and after the episodes. The y axis is the
task-stratified paired difference against the reference, so it is a difference, not a success
rate, and the interval is the bootstrap 95 % interval of that difference.
"""

import math

import matplotlib.pyplot as plt

from report.plots.style import COL, COLORS, INK_2, MUTED, OUT, style_axes, tint

MARGIN_PP = -5.0  # margin, `decision_rules.reserved_gate` of the frozen plan
CELLS = [
    ("B16", "Reference", "reference"),
    ("precision-lang-int8", "INT8 decoder", "precision"),
    ("precision-lang-nf4", "NF4 decoder", "precision"),
    ("ladder-nf4-graph-1step-packed", "NF4 packed + graph", "precision"),
    ("iter_000-b0-a3", "First search, cycle 1", "search"),
    ("iter_001-b1-a3", "First search, cycle 2", "search"),
    ("iter_001-b2-a1", "Second search, cycle 2", "search"),
    ("iter_002-b2-a2", "Second search, cycle 3", "search"),
]
# One hue per group; the marker shape repeats the grouping for greyscale and colour-blind readers.
GROUPS = {
    "reference": ("s", COLORS["B16"]),
    "precision": ("o", COLORS["language"]),
    "search": ("D", COLORS["C1"]),
}
FAN_DX_GB = 0.28  # label column of the search fan, just right of the search cluster
FAN_TEXT_GB = 0.05  # gap between a leader's end and its label
FAN_GAP_PP = 0.70  # clearance above the fan's top label and below its bottom label
FAN_LINE_PP = 1.15  # label line spacing of the fan, percentage points


def _summary_cells(n: dict) -> dict:
    """Reserved-summary entries keyed by cell id."""
    return {entry["cell"]: entry for entry in (n.get("reserved_summary") or {}).get("cells") or []}


def _solution_cells(n: dict) -> dict:
    """Dream-RSI solution cells keyed by cell id, from both programs."""
    found = {}
    for key in ("dream_rsi_reserved", "dream_rsi_v2_reserved"):
        for entry in (n.get(key) or {}).get("cells") or []:
            found[entry["cell"]] = entry
    return found


def paired(n: dict, cell: str) -> tuple | None:
    """Paired difference against the reference in pp, with its 95 % interval."""
    if cell == "B16":
        values = (n["rows"].get(cell) or {}).get("final_paired_pp")
        return (values[0], values[1:3]) if values else None
    entry = _summary_cells(n).get(cell) or {}
    if entry.get("paired_vs_B16"):
        pair = entry["paired_vs_B16"]
        return pair["difference_pp"], pair["ci95_pp"]
    solution = _solution_cells(n).get(cell) or {}
    if solution.get("difference_pp") is not None:
        return solution["difference_pp"], solution["ci95_pp"]
    return None


def memory_gb(n: dict, cell: str) -> float | None:
    """Peak allocated GPU memory of the cell's weights, in GB."""
    row = n["rows"].get(cell)
    if row is not None:
        return row.get("benchmark_peak_allocated_gb")
    return (_solution_cells(n).get(cell) or {}).get("resident_gb")


def plotted_rows(n: dict) -> list[dict] | None:
    """One row per plotted cell; None until every cell has a 500-episode reserved result."""
    rows = []
    for cell, label, group in CELLS:
        memory, pair = memory_gb(n, cell), paired(n, cell)
        if memory is None or pair is None:
            return None
        rows.append(
            {
                "cell": cell,
                "label": label,
                "group": group,
                "memory_gb": memory,
                "difference_pp": pair[0],
                "ci95_pp": list(pair[1]),
            }
        )
    return rows


def annotate_cells(ax: plt.Axes, rows: list[dict]) -> None:
    """Direct labels: precision cells above their interval, search winners in a right-hand fan.

    The four search winners sit within 0.25 GB of each other, so their labels stack in one column
    right of the cluster and a thin leader line ties each label to its point. Labels are ordered by
    memory, rightmost point on top, so the leaders fan out without crossing.
    """
    search = sorted(
        (row for row in rows if row["group"] == "search"), key=lambda row: -row["memory_gb"]
    )
    column = max(row["memory_gb"] for row in search) + FAN_DX_GB
    others = [row for row in rows if row["group"] != "search"]
    # The fan starts under the reference line and under the closest interval to its right, and
    # stops short of the non-inferiority margin.
    right = [row for row in others if row["memory_gb"] > column]
    nearest = min(right, key=lambda row: row["memory_gb"]) if right else None
    top = min(0.0, nearest["ci95_pp"][0] if nearest else 0.0) - FAN_GAP_PP
    floor = MARGIN_PP + FAN_GAP_PP
    step = min(FAN_LINE_PP, (top - floor) / max(len(search) - 1, 1))
    leader = {
        "arrowstyle": "-",
        "lw": 0.4,
        "color": tint(INK_2, 0.45),
        "shrinkA": 1.5,
        "shrinkB": 2.5,
    }
    for index, row in enumerate(search):
        # A leader drawn on its own ends at the label anchor; as part of the text annotation it
        # would stop on the text's edge and point at the middle of the word.
        ax.annotate(
            "",
            (row["memory_gb"], row["difference_pp"]),
            xytext=(column, top - index * step),
            textcoords="data",
            arrowprops=dict(leader, shrinkA=0.0),
            annotation_clip=False,
        )
        ax.text(
            column + FAN_TEXT_GB,
            top - index * step,
            row["label"],
            ha="left",
            va="center",
            fontsize=6.5,
            color=COLORS["C1"],
        )
    for row in others:
        color = GROUPS[row["group"]][1]
        if row["group"] == "reference":
            ax.annotate(
                row["label"],
                (row["memory_gb"], row["difference_pp"]),
                xytext=(-5.0, 1.5),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=6.5,
                color=color,
                annotation_clip=False,
            )
            continue
        ax.annotate(
            row["label"],
            (row["memory_gb"], row["ci95_pp"][1]),
            xytext=(0.0, 4.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=color,
            annotation_clip=False,
        )


def fig_reserved(n: dict) -> dict | None:
    """Reserved success against peak memory, one point per cell; None while a cell is missing."""
    rows = plotted_rows(n)
    if rows is None:
        return None

    fig, ax = plt.subplots(figsize=(COL, 2.4))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.965, bottom=0.155)
    low = min(row["ci95_pp"][0] for row in rows)
    high = max(row["ci95_pp"][1] for row in rows)
    xlim = (0.0, float(math.ceil(max(row["memory_gb"] for row in rows))))
    ylim = (min(low, MARGIN_PP) - 1.0, high + 1.4)
    ax.set(xlim=xlim, ylim=ylim)

    ax.axhline(0.0, color=INK_2, lw=0.7, zorder=1)
    ax.axhline(MARGIN_PP, color=MUTED, lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.text(
        xlim[1] - 0.06,
        MARGIN_PP + 0.12,
        "$-5$ pp margin",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=MUTED,
    )
    for row in rows:
        marker, color = GROUPS[row["group"]]
        ax.errorbar(
            row["memory_gb"],
            row["difference_pp"],
            yerr=[
                [row["difference_pp"] - row["ci95_pp"][0]],
                [row["ci95_pp"][1] - row["difference_pp"]],
            ],
            color=color,
            marker=marker,
            ms=3.4,
            mfc=color,
            mec="white",
            mew=0.7,
            elinewidth=0.8,
            capsize=0,
            zorder=3,
        )
    annotate_cells(ax, rows)
    style_axes(
        ax,
        ylabel="Paired difference vs reference (pp)",
        xlabel="Peak allocated GPU memory (GB)",
        yticks=list(range(int(ylim[0]), int(ylim[1]) + 1, 2)),
    )

    fig.savefig(OUT / "reserved.pdf")
    plt.close(fig)
    return {
        "status": "ok",
        "memory": "peak allocated GPU memory (GB)",
        "margin_pp": MARGIN_PP,
        "cells": [
            {
                "cell": row["cell"],
                "label": row["label"],
                "memory_gb": row["memory_gb"],
                "difference_pp": row["difference_pp"],
                "ci95_pp": row["ci95_pp"],
            }
            for row in rows
        ],
    }
