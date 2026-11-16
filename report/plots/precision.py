"""Precision response surface and format curve (Figure 2).

Success is plotted against effective bits `8 S(x)/N(x)` per module, computed from the per-module
storage each run records, never against nominal bits. Points are measured, not smoothed.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from report.plots.style import COLORS, FULL, INK_2, MUTED, OUT, panel_label, style_axes, tint

MODULES = [
    ("language", "precision-language", "text"),
    ("vision", "precision-vision", "vision"),
    ("action", "precision-action", "action"),
]
FORMAT_FAMILIES = {"precision-language", "format-language"}
INTERACTION = [
    ("B16", "B16"),
    ("precision-vision-hqq2", "vision 2-bit"),
    ("precision-action-hqq2", "action 2-bit"),
    ("precision-joint-va-hqq2", "joint 2-bit"),
]
FORMAT_NAMES = {
    "awq": "AWQ",
    "gptq": "GPTQ",
    "hqq4": "HQQ 4-bit",
    "hqq3": "HQQ 3-bit",
    "hqq2": "HQQ 2-bit",
    "nf4": "NF4",
    "int8": "INT8",
}
LABEL_GAP_GB = 0.1  # formats closer than this in memory would put their labels on top of each other
FAN_LINE_PT = 8.1  # line height of the stacked label block, points
FAN_DX_PT = 16.0  # leader-line length from point to label block, points


def module_points(n: dict, family: str, module: str) -> list[dict]:
    """Development cells of one module family, with their effective bits and Wilson interval."""
    points = []
    for candidate, row in n["rows"].items():
        bits = (row.get("effective_bits") or {}).get(module)
        if row.get("family") != family or bits is None:
            continue
        if "development_successes" not in row:
            continue
        points.append(
            {
                "candidate": candidate,
                "bits": bits,
                "success_pct": row["development_success_pct"],
                "wilson_pct": row["development_wilson_pct"],
            }
        )
    return sorted(points, key=lambda point: point["bits"])


def _reference(n: dict, ax: plt.Axes):
    """B16 development success as a dashed line with its Wilson band."""
    b16 = n["rows"].get("B16") or {}
    if "development_success_pct" not in b16:
        return
    low, high = b16["development_wilson_pct"]
    ax.axhspan(low, high, color=tint(MUTED, 0.86), lw=0, zorder=0)
    ax.axhline(b16["development_success_pct"], color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)


def fig_precision(n: dict) -> dict | None:
    """Precision surface plus the joint 2-bit interaction; None until the curves exist."""
    series = {name: module_points(n, family, module) for name, family, module in MODULES}
    if not any(len(points) >= 2 for points in series.values()):
        return None
    interaction = [(label, n["rows"].get(candidate) or {}) for candidate, label in INTERACTION]
    interaction = [(label, row) for label, row in interaction if "development_success_pct" in row]
    panels = 2 if len(interaction) == 4 else 1
    fig, axes = plt.subplots(
        1, panels, figsize=(FULL, 1.95), gridspec_kw={"wspace": 0.22, "width_ratios": [1.55, 1.0]}
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.235)
    ax = axes[0] if panels == 2 else axes
    ledger = {"series": {}, "interaction": {}}

    _reference(n, ax)
    for name, _, _ in MODULES:
        points = series[name]
        if not points:
            continue
        x = [point["bits"] for point in points]
        y = [point["success_pct"] for point in points]
        low = [point["success_pct"] - point["wilson_pct"][0] for point in points]
        high = [point["wilson_pct"][1] - point["success_pct"] for point in points]
        ax.errorbar(
            x,
            y,
            yerr=[low, high],
            color=COLORS[name],
            marker="o",
            ms=3,
            mfc=COLORS[name],
            mec="white",
            mew=0.7,
            lw=1.35,
            elinewidth=0.7,
            capsize=0,
            label=name,
            zorder=3,
        )
        ledger["series"][name] = {
            point["candidate"]: {
                "effective_bits": point["bits"],
                "success_pct": point["success_pct"],
            }
            for point in points
        }
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=6.5,
        handlelength=1.6,
        handletextpad=0.4,
        borderaxespad=0.2,
        labelcolor=INK_2,
    )
    measured = [point["bits"] for points in series.values() for point in points]
    ax.set(xlim=(min(measured) - 0.4, max(measured) + 0.9), ylim=(-3, 104))
    style_axes(
        ax,
        ylabel="Development success (%)",
        xlabel="Effective bits per weight",
        yticks=[0, 25, 50, 75, 100],
    )
    panel_label(ax, "a", "Precision surface")

    if panels == 2:
        b = axes[1]
        labels = [label for label, _ in interaction]
        values = [row["development_success_pct"] for _, row in interaction]
        colors = [COLORS["B16"], COLORS["vision"], COLORS["action"], COLORS["joint"]]
        positions = np.arange(len(labels))
        b.bar(positions, values, width=0.58, color=colors, zorder=3)
        for index, (label, row) in enumerate(interaction):
            low, high = row["development_wilson_pct"]
            b.plot([index, index], [low, high], color=INK_2, lw=0.7, zorder=4)
            b.text(
                index,
                high + 2,
                f"{row['development_success_pct']:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK_2,
            )
            ledger["interaction"][label] = {
                "success_pct": row["development_success_pct"],
                "wilson_pct": row["development_wilson_pct"],
            }
        b.set(xticks=positions, xticklabels=labels, ylim=(0, 112))
        b.tick_params(axis="x", labelsize=6.5)
        style_axes(b, yticks=[0, 25, 50, 75, 100])
        panel_label(b, "b", "2-bit cells")

    fig.savefig(OUT / "precision.pdf")
    plt.close(fig)
    return ledger


def format_name(candidate: str) -> str:
    """Readable format name from a cell id: `format-lang-hqq4` becomes `HQQ 4-bit`."""
    suffix = candidate.removeprefix("precision-lang-").removeprefix("format-lang-")
    return FORMAT_NAMES.get(suffix, suffix.upper())


def label_groups(points: list[dict]) -> list[list[dict]]:
    """Group memory-sorted points whose labels would collide, so they can be stacked."""
    groups: list[list[dict]] = []
    for point in points:
        if groups and point["peak_gb"] - groups[-1][-1]["peak_gb"] < LABEL_GAP_GB:
            groups[-1].append(point)
        else:
            groups.append([point])
    return groups


def annotate_formats(ax: plt.Axes, points: list[dict], xlim: tuple, ylim: tuple) -> None:
    """Label each format; a memory cluster becomes one label block with a single leader line.

    Offsets are in points, so the block keeps its spacing at any output size. The data-per-point
    factors come from the axes box, which is final once the limits are set.
    """
    box = ax.get_position()
    x_per_pt = (xlim[1] - xlim[0]) / (box.width * ax.figure.get_figwidth() * 72)
    y_per_pt = (ylim[1] - ylim[0]) / (box.height * ax.figure.get_figheight() * 72)
    high = max(point["peak_gb"] for point in points)
    leader = {
        "arrowstyle": "-",
        "lw": 0.4,
        "color": tint(INK_2, 0.45),
        "shrinkA": 1.5,
        "shrinkB": 2.5,
    }
    for group in label_groups(points):
        if len(group) == 1:
            point = group[0]
            # The rightmost label goes above its point, clear of the reference band and line.
            above = point["peak_gb"] >= high
            ax.annotate(
                format_name(point["candidate"]),
                (point["peak_gb"], point["success_pct"]),
                xytext=(0.0, 8.0) if above else (4.0, 0.0),
                textcoords="offset points",
                ha="center" if above else "left",
                va="bottom" if above else "center",
                fontsize=6.5,
                color=INK_2,
                annotation_clip=False,
            )
            continue
        # The group is one blob: formats within 0.11 GB and 3 points of each other. One label
        # block and one leader line say that, where five lines would graze each other's markers.
        x_anchor = max(point["peak_gb"] for point in group) + FAN_DX_PT * x_per_pt
        mean = sum(point["success_pct"] for point in group) / len(group)
        height = len(group) * FAN_LINE_PT * y_per_pt
        centre = max(mean - 8 * y_per_pt - height / 2, ylim[0] + 12 + height / 2)
        ax.annotate(
            "\n".join(format_name(point["candidate"]) for point in group),
            (group[-1]["peak_gb"], group[-1]["success_pct"]),
            xytext=(x_anchor, centre),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=6.5,
            color=INK_2,
            linespacing=1.25,
            arrowprops=dict(leader),
            annotation_clip=False,
        )


def fig_format(n: dict) -> dict | None:
    """Decoder format curve: development success against peak memory, one point per format."""
    points = []
    for candidate, row in n["rows"].items():
        if row.get("family") not in FORMAT_FAMILIES:
            continue
        if "development_success_pct" not in row:
            continue
        peak = row.get("benchmark_peak_allocated_gb", row.get("screen_peak_allocated_gb"))
        if peak is None:
            continue
        points.append(
            {
                "candidate": candidate,
                "peak_gb": peak,
                "success_pct": row["development_success_pct"],
                "wilson_pct": row["development_wilson_pct"],
            }
        )
    if len(points) < 2:
        return None
    points.sort(key=lambda point: point["peak_gb"])
    low = min(point["peak_gb"] for point in points)
    high = max(point["peak_gb"] for point in points)
    pad = 0.14 * (high - low or high)
    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    fig.subplots_adjust(left=0.145, right=0.985, top=0.88, bottom=0.225)
    _reference(n, ax)
    for point in points:
        ax.errorbar(
            point["peak_gb"],
            point["success_pct"],
            yerr=[
                [point["success_pct"] - point["wilson_pct"][0]],
                [point["wilson_pct"][1] - point["success_pct"]],
            ],
            color=COLORS["language"],
            marker="o",
            ms=3,
            mfc=COLORS["language"],
            mec="white",
            mew=0.7,
            elinewidth=0.7,
            capsize=0,
            zorder=3,
        )
    ax.set(xlim=(low - pad, high + pad), ylim=(-3, 104))
    annotate_formats(ax, points, (low - pad, high + pad), (-3, 104))
    style_axes(
        ax,
        ylabel="Development success (%)",
        xlabel="Peak allocated GPU memory (GB)",
        yticks=[0, 25, 50, 75, 100],
    )
    ax.legend(
        handles=[
            Patch(facecolor=tint(MUTED, 0.86), edgecolor="none", label="reference 95 % interval")
        ],
        loc="upper left",
        frameon=False,
        fontsize=6.2,
        handlelength=1.2,
        handleheight=0.8,
        handletextpad=0.4,
        borderaxespad=0.3,
        labelcolor=INK_2,
    )
    fig.savefig(OUT / "format.pdf")
    plt.close(fig)
    return {"points": {point["candidate"]: point for point in points}}
