"""Best-so-far timelines of the two kinds of Dream-RSI search (Figures `size-timeline`, `latency-timeline`).

Both read committed records only: `tree.json` order plus the corrected scores in the cycle
ledger (size), and `studies/program-v4/nodes.json` (latency). Each returns None when its records
are absent. Exploit nodes are drawn but never set the running best.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from report.plots.data import ROOT, optional_json
from report.plots.style import COL, COLORS, GRID, INK_2, MUTED, OUT, style_axes

RED = "#DC2626"
LABELS = {  # node id -> what changed, for the jumps that matter
    "program-v1": {
        "iter_000-b0-a1": "drop world model",
        "iter_000-b2-a2": "packed int3",
    },
    "program-v2": {
        "iter_000-b0-a0": "exact trims, int8 text",
        "iter_000-b1-a0": "int4 all",
        "iter_000-b2-a2": "int3 vision, action",
    },
}
LATENCY_LABELS = {"iter_000-b0-a0": "CUDA graph", "iter_000-b0-a1": "no per-call eval walk"}


def running_best(points: Sequence[dict]) -> list[float | None]:
    """Best valid score so far at each point; None before the first valid score."""
    best, out = None, []
    for point in points:
        if point.get("valid", True) and (best is None or point["score"] > best):
            best = point["score"]
        out.append(best)
    return out


def jumps(points: Sequence[dict]) -> list[int]:
    """Indices at which the running best rises."""
    best, out = None, []
    for index, point in enumerate(points):
        if point.get("valid", True) and (best is None or point["score"] > best):
            best = point["score"]
            out.append(index)
    return out


def _boundaries(points: Sequence[dict]) -> list[float]:
    return [i - 0.5 for i in range(1, len(points)) if points[i]["cycle"] != points[i - 1]["cycle"]]


def draw_timeline(ax: plt.Axes, points: Sequence[dict], color: str, labels: dict, scale: float):
    """Grey dots, a step line for the running best, red x for invalid nodes, direct labels."""
    x = list(range(len(points)))
    for edge in _boundaries(points):
        ax.axvline(edge, color=GRID, lw=0.8, zorder=0)
    ok = [i for i, p in enumerate(points) if p.get("valid", True)]
    ax.scatter(ok, [points[i]["score"] * scale for i in ok], s=9, color="#A8B1BD", lw=0, zorder=2)
    bad = [i for i in x if i not in ok]
    ax.scatter(
        bad,
        [points[i]["score"] * scale for i in bad],
        s=18,
        marker="x",
        color=RED,
        lw=0.9,
        zorder=3,
    )
    best = running_best(points)
    start = next(i for i, b in enumerate(best) if b is not None)
    ax.step(
        x[start:],
        [b * scale for b in best[start:]],
        where="post",
        color=color,
        lw=1.4,
        zorder=4,
    )
    ax.scatter(
        jumps(points),
        [points[i]["score"] * scale for i in jumps(points)],
        s=11,
        color=color,
        lw=0,
        zorder=5,
    )


def _label_jumps(ax, points, labels, scale, color, offsets):
    """Name the jumps that matter; a thin leader joins each label to its dot."""
    for index in jumps(points):
        text = labels.get(points[index]["cell_id"])
        if text:
            dx, dy = offsets.get(points[index]["cell_id"], (4, -9))
            ax.annotate(
                text,
                (index, points[index]["score"] * scale),
                xytext=(dx, dy),
                textcoords="offset points",
                color=color,
                fontsize=6.5,
                va="center",
                ha="left" if dx >= 0 else "right",
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "lw": 0.5,
                    "shrinkA": 1,
                    "shrinkB": 2,
                },
            )


def _tree_order(cycle: dict, source: str) -> list[str]:
    """Node ids in probe order from the cycle's `tree.json`, falling back to id order."""
    tree = optional_json(ROOT / source / cycle["cycle_id"] / "tree.json") or {}
    ranked = sorted(tree.get("nodes") or [], key=lambda n: n.get("seq", 0))
    return [n["cell_id"] for n in ranked] or [n["cell_id"] for n in cycle["nodes"]]


def study_points(cycles: dict) -> list[dict]:
    """Probes of one study in search order, with the corrected score and validity."""
    points = []
    for index, cycle in enumerate(cycles.get("cycles", [])):
        nodes = {n["cell_id"]: n for n in cycle["nodes"]}
        for cell_id in _tree_order(cycle, cycles["source"]):
            node = nodes.get(cell_id)
            if node and node.get("score") is not None:
                valid = node.get("valid", True)
                points.append(
                    {"cell_id": cell_id, "cycle": index, "score": node["score"], "valid": valid}
                )
    return points


SIZE_OFFSETS = {
    "program-v1": {"iter_000-b0-a1": (8, -26), "iter_000-b2-a2": (6, 12)},
    "program-v2": {
        "iter_000-b0-a0": (6, -10),
        "iter_000-b1-a0": (6, -14),
        "iter_000-b2-a2": (10, -12),
    },
}


def fig_size_timeline(n: dict) -> dict | None:
    """Memory removed per probe for both size searches, stacked; None without v1 and v2 cycles."""
    studies = {"program-v1": n.get("dream_rsi") or {}, "program-v2": n.get("dream_rsi_v2") or {}}
    series = {name: study_points(c) for name, c in studies.items() if c.get("cycles")}
    if len(series) < 2:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.9))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.95, bottom=0.115, hspace=0.5)
    colors = {"program-v1": COLORS["Q4"], "program-v2": COLORS["C1"]}
    names = {"program-v1": "First search", "program-v2": "Second search"}
    ledger = {}
    for ax, (name, points) in zip(axes, series.items(), strict=True):
        draw_timeline(ax, points, colors[name], LABELS[name], 100.0)
        _label_jumps(ax, points, LABELS[name], 100.0, colors[name], SIZE_OFFSETS[name])
        ax.set(xlim=(-0.7, len(points) - 0.3), ylim=(0, 105))
        ax.set_xticks(range(3, len(points), 4))
        ax.set_xticklabels([str(i + 1) for i in range(3, len(points), 4)])
        style_axes(ax, ylabel="Memory removed (%)", yticks=[0, 50, 100])
        ax.set_title(names[name], loc="left", fontsize=7, color=INK_2, pad=2)
        best = running_best(points)
        ledger[name] = {
            "probes": len(points),
            "cycle_sizes": [sum(1 for p in points if p["cycle"] == c) for c in range(3)],
            "best_pct": round(best[-1] * 100, 2),
            "exploit_nodes": [p["cell_id"] for p in points if not p["valid"]],
            "jumps": [
                {
                    "probe": i + 1,
                    "cell_id": points[i]["cell_id"],
                    "pct": round(points[i]["score"] * 100, 2),
                }
                for i in jumps(points)
            ],
        }
    axes[-1].set_xlabel("Probe, in search order")
    fig.savefig(OUT / "size-timeline.pdf")
    plt.close(fig)
    return ledger


def latency_points(table: dict) -> list[dict]:
    """Probes of the latency run in evaluation order; score is the guarded lower bound."""
    return [
        {"cell_id": row["cell_id"], "cycle": row["cycle"], "score": row["score"] or 0.0}
        for row in table["nodes"]
    ]


def fig_latency_timeline(n: dict, nodes_path: Path) -> dict | None:
    """Speedup lower bound per probe of the live latency search; None without `nodes.json`."""
    table = optional_json(nodes_path)
    if not table or not table.get("nodes"):
        return None
    rows = n["rows"]
    graph = rows.get("ladder-b16-graph-4step", {}).get("benchmark_median_ms")
    stock = rows.get("B16", {}).get("benchmark_median_ms")
    if not graph or not stock:
        return None
    reference = stock / graph
    points = latency_points(table)
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.97, bottom=0.17)
    color = COLORS["Q8"]
    draw_timeline(ax, points, color, LATENCY_LABELS, 1.0)
    for value, text in ((1.0, "untouched policy"), (reference, "hand-built graph")):
        ax.axhline(value, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.annotate(
            f"{text}, {value:.1f}x",
            (-0.4, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="left",
            fontsize=6.5,
            color=MUTED,
            annotation_clip=False,
        )
    _label_jumps(
        ax,
        points,
        LATENCY_LABELS,
        1.0,
        color,
        {"iter_000-b0-a1": (-6, -14), "iter_000-b0-a0": (6, -14)},
    )
    ax.set(xlim=(-0.5, len(points) - 0.5), ylim=(-0.1, 4.2))
    ax.set_xticks(range(len(points)))
    ax.set_xticklabels([str(i + 1) for i in range(len(points))])
    style_axes(
        ax, ylabel="Speedup, lower bound", xlabel="Probe, in search order", yticks=[0, 1, 2, 3, 4]
    )
    fig.savefig(OUT / "latency-timeline.pdf")
    plt.close(fig)
    best = running_best(points)
    return {
        "snapshot_utc": table["snapshot_utc"],
        "nodes": len(points),
        "cycles_complete": table["cycles_complete"],
        "cycles_target": table["cycles_target"],
        "best_lower_bound": round(best[-1], 3),
        "best_node": points[jumps(points)[-1]]["cell_id"],
        "hand_built_speedup": round(reference, 3),
        "source": "studies/program-v4/nodes.json",
    }
