"""Best-so-far timelines of the two kinds of Dream-RSI search (Figures `size-timeline`, `latency-timeline`).

Both read committed records only: `tree.json` order plus the corrected scores in the cycle
ledger (size), and `studies/program-v4/nodes.json` (latency). Each returns None when its records
are absent. Exploit nodes are drawn but never set the running best.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from report.plots.data import ROOT, optional_json
from report.plots.style import COL, COLORS, GRID, INK, INK_2, MUTED, OUT, style_axes, text_ink

RED = "#DC2626"
LABELS = {  # node id -> what changed, for the jumps that matter
    "program-v1": {
        "iter_000-b0-a1": "drop world model",
        "iter_000-b0-a3": "prune MLP channels",
    },
    "program-v2": {
        "iter_000-b0-a0": "trim, 8-bit text",
        "iter_000-b1-a0": "4-bit weights",
        "iter_000-b2-a2": "3-bit vision, action",
    },
    "program-v3": {
        "iter_000-b2-a0": "4-bit text, 8-bit rest",
        "iter_000-b2-a2": "2 to 3-bit text",
        "iter_002-b2-a2": "packed 3-bit",
    },
}
LATENCY_LABELS = {
    "iter_000-b0-a0": "CUDA graph",
    "iter_000-b0-a3": "compiled, 4 steps",
    "iter_001-b0-a2": "2 flow steps",
    "iter_001-b1-a0": "1 flow step",
}


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


def draw_timeline(ax: plt.Axes, points: Sequence[dict], color: str, scale: float, hollow=False):
    """Grey dots, a step line for the running best; invalid nodes are red x, or hollow rings.

    An invalid point is drawn at its `plot` value when it has one, else at its score.
    """
    x = list(range(len(points)))
    for edge in _boundaries(points):
        ax.axvline(edge, color=GRID, lw=0.8, zorder=0)
    ok = [i for i, p in enumerate(points) if p.get("valid", True)]
    ax.scatter(ok, [points[i]["score"] * scale for i in ok], s=9, color="#A8B1BD", lw=0, zorder=2)
    bad = [i for i in x if i not in ok]
    ys = [points[i].get("plot", points[i]["score"]) * scale for i in bad]
    if hollow:
        ax.scatter(bad, ys, s=14, facecolor="white", edgecolor=color, lw=0.9, zorder=3)
    else:
        ax.scatter(bad, ys, s=18, marker="x", color=RED, lw=0.9, zorder=3)
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
                color=text_ink(color),
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
    """Probes of one study in search order, scored as the search saw them.

    `score_old` is the search-time score of an audited node. Audited nodes that were deferred or
    exploited are suspect; honest ones carry their re-measured score as `remeasured`.
    """
    points = []
    for index, cycle in enumerate(cycles.get("cycles", [])):
        nodes = {n["cell_id"]: n for n in cycle["nodes"]}
        for cell_id in _tree_order(cycle, cycles["source"]):
            node = nodes.get(cell_id)
            if not node or node.get("score") is None:
                continue
            audit = node.get("audit_class")
            point = {
                "cell_id": cell_id,
                "cycle": index,
                "score": node.get("score_old", node["score"]),
                "valid": audit not in ("exploit", "deferred"),
            }
            if audit == "honest" and "score_old" in node:
                point["remeasured"] = node["score"]
            points.append(point)
    return points


SIZE_OFFSETS = {
    "program-v1": {
        "iter_000-b0-a1": (10, -22),
        "iter_000-b0-a3": (10, 12),
    },
    "program-v2": {
        "iter_000-b0-a0": (6, -10),
        "iter_000-b1-a0": (6, -14),
        "iter_000-b2-a2": (10, -12),
    },
    "program-v3": {
        "iter_000-b2-a0": (6, -12),
        "iter_000-b2-a2": (6, 11),
        "iter_002-b2-a2": (-8, -14),
    },
}


def _remeasured(ax, points):
    """Filled diamonds at the re-measured score, joined to the search-time dot by a thin line."""
    marked = [(i, p) for i, p in enumerate(points) if "remeasured" in p]
    for i, p in marked:
        ax.plot([i, i], [p["score"] * 100, p["remeasured"] * 100], color=INK_2, lw=0.6, zorder=3)
        ax.scatter([i], [p["remeasured"] * 100], s=14, marker="D", color=INK_2, lw=0, zorder=6)
    if marked:
        i, p = marked[0]
        ax.annotate(
            "re-measured",
            (i, p["remeasured"] * 100),
            xytext=(7, -12),
            textcoords="offset points",
            fontsize=6.5,
            color=INK_2,
            ha="left",
            va="center",
            annotation_clip=False,
        )
    suspect = [i for i, p in enumerate(points) if not p["valid"]]
    if suspect:
        top = max(points[i]["score"] for i in suspect) * 100
        ax.annotate(
            "suspect",
            (suspect[len(suspect) // 2], top),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=6.5,
            color=RED,
            ha="center",
            annotation_clip=False,
        )


def fig_size_timeline(n: dict) -> dict | None:
    """Memory removed per probe for the size searches, stacked; None without v1 and v2 cycles."""
    studies = {
        "program-v1": n.get("dream_rsi") or {},
        "program-v2": n.get("dream_rsi_v2") or {},
        "program-v3": n.get("dream_rsi_v3") or {},
    }
    series = {name: study_points(c) for name, c in studies.items() if c.get("cycles")}
    if len(series) < 2:
        return None
    rows = len(series)
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(COL, 2.8 if rows == 2 else 3.8),
        gridspec_kw={"height_ratios": [1.3, 1, 1][:rows]},
    )
    fig.subplots_adjust(
        left=0.105, right=0.985, top=0.965, bottom=0.085 if rows > 2 else 0.115, hspace=0.62
    )
    colors = {"program-v1": COLORS["Q4"], "program-v2": COLORS["C1"], "program-v3": COLORS["Q8"]}
    names = {
        "program-v1": "First search",
        "program-v2": "Second search",
        "program-v3": "Third search, code hidden",
    }
    ledger = {}
    for ax, (name, points) in zip(axes, series.items(), strict=True):
        draw_timeline(ax, points, colors[name], 100.0)
        _label_jumps(ax, points, LABELS[name], 100.0, colors[name], SIZE_OFFSETS[name])
        _remeasured(ax, points)
        ax.set(xlim=(-0.7, len(points) - 0.3), ylim=(0, 114))
        ax.set_xticks(range(3, len(points), 4))
        ax.set_xticklabels([str(i + 1) for i in range(3, len(points), 4)])
        style_axes(ax, yticks=[0, 50, 100])
        ax.set_title(names[name], loc="left", fontsize=7, fontweight="medium", color=INK, pad=4)
        best = running_best(points)
        ledger[name] = {
            "probes": len(points),
            "cycle_sizes": [sum(1 for p in points if p["cycle"] == c) for c in range(3)],
            "best_pct": round(best[-1] * 100, 2),
            "suspect_nodes": [p["cell_id"] for p in points if not p["valid"]],
            "remeasured": {
                p["cell_id"]: round(p["remeasured"] * 100, 2) for p in points if "remeasured" in p
            },
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
    fig.supylabel("Memory removed (%)", fontsize=7.5, color=INK_2, x=0.01)
    fig.savefig(OUT / "size-timeline.pdf")
    plt.close(fig)
    return ledger


def latency_points(table: dict) -> list[dict]:
    """Probes of the latency run in evaluation order; score is the guarded lower bound.

    A node scored 0 by the smoke gate is invalid and is drawn at its speedup point.
    """
    points = []
    for row in table["nodes"]:
        point = {"cell_id": row["cell_id"], "cycle": row["cycle"], "score": row["score"] or 0.0}
        gated = not row["score"] and (row.get("speedup") or {}).get("point")
        if gated:
            point.update(valid=False, plot=row["speedup"]["point"])
        points.append(point)
    return points


def fig_latency_timeline(n: dict, nodes_path: Path) -> dict | None:
    """Speedup lower bound per probe of the live latency search; None without `nodes.json`."""
    table = optional_json(nodes_path)
    if not table or not table.get("nodes"):
        return None
    # The hand-built graph is re-timed with the same per-task, block-interleaved protocol as the
    # search winners, so its line and the dots share one ruler.
    timing = optional_json(nodes_path.parent / "timing.json") or {}
    graph = (timing.get("entries") or {}).get("ladder-b16-graph-4step", {})
    reference = graph.get("speedup")
    if not reference:
        return None
    points = latency_points(table)
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.97, bottom=0.17)
    color = COLORS["Q8"]
    draw_timeline(ax, points, color, 1.0, hollow=True)
    for value, text in ((1.0, "untouched policy"), (reference, "hand-built graph")):
        ax.axhline(value, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.annotate(
            f"{text}, {value:.1f}x",
            (len(points) - 0.6, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
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
        {
            "iter_000-b0-a0": (4, -12),
            "iter_000-b0-a3": (-5, 8),
            "iter_001-b0-a2": (5, -11),
            "iter_001-b1-a0": (4, 9),
        },
    )
    ax.set(xlim=(-0.5, len(points) - 0.5), ylim=(-0.1, 6.0))
    ax.set_xticks(range(len(points)))
    ax.set_xticklabels([str(i + 1) for i in range(len(points))])
    style_axes(
        ax,
        ylabel="Speedup, lower bound",
        xlabel="Probe, in search order",
        yticks=[0, 1, 2, 3, 4, 5],
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
        "source": "studies/program-v4/nodes.json, studies/program-v4/timing.json",
    }
