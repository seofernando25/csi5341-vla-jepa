"""Precision response surface and format curve (Figure 2).

Success is plotted against effective bits `8 S(x)/N(x)` per module, computed from the per-module
storage each run records, never against nominal bits. Points are measured, not smoothed.
"""

import matplotlib.pyplot as plt
import numpy as np

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
        xlabel="Effective bits per module, $8S(x)/N(x)$",
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
        b.tick_params(axis="y", labelleft=False)
        panel_label(b, "b", "Joint 2-bit cell")

    fig.savefig(OUT / "precision.pdf")
    plt.close(fig)
    return ledger


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
    for index, point in enumerate(points):
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
        ax.annotate(
            point["candidate"].removeprefix("precision-lang-").removeprefix("format-lang-"),
            (point["peak_gb"], point["success_pct"]),
            xytext=(3, 4 if index % 2 == 0 else -9),
            textcoords="offset points",
            fontsize=6.5,
            color=INK_2,
            annotation_clip=False,
        )
    ax.set(xlim=(low - pad, high + pad), ylim=(-3, 104))
    style_axes(
        ax,
        ylabel="Development success (%)",
        xlabel="Peak allocated GPU memory (GB)",
        yticks=[0, 25, 50, 75, 100],
    )
    fig.savefig(OUT / "format.pdf")
    plt.close(fig)
    return {"points": {point["candidate"]: point for point in points}}
