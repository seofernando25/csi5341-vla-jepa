"""Precision response surface (Figure 2) and the decoder format curve.

The surface plots development success against nominal weight bits per module: one line per module,
the uncompressed reference at 16 bits, and the joint vision+action 2-bit cell beside the
single-module 2-bit cells. Nominal bits come from the cell's recipe, or from its format suffix
when the recipe records none; where a module has several 4-bit formats, NF4 stands for the line.
Points are measured development rollouts, never pooled with smoke.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from report.plots.data import recipes
from report.plots.style import COL, COLORS, INK_2, MUTED, OUT, style_axes, tint

MODULES = ("language", "vision", "action")
FAMILIES = {
    "language": {"precision-language", "format-language"},
    "vision": {"precision-vision"},
    "action": {"precision-action"},
}
SCOPES = {
    "language": {"text"},
    "vision": {"vision_blocks", "merger"},
    "action": {"action_mlp"},
}
# Nominal bits by method or by the trailing format token of a cell id.
BITS_BY_NAME = {
    "int8": 8,
    "nf4": 4,
    "bnb": 4,
    "awq": 4,
    "awq_diagonal": 4,
    "gptq": 4,
    "hqq": 4,
    "hqq4": 4,
    "hqq3": 3,
    "hqq2": 2,
}
REFERENCE_BITS = 16  # the uncompressed bfloat16 checkpoint
BIT_TICKS = (2, 3, 4, 8, 16)
OFFSETS = {"language": -0.16, "vision": 0.0, "action": 0.16}  # category units, so bars clear
JOINT = ("precision-joint-va-hqq2", "vision + action")
JOINT_OFFSET = 0.30
LABEL_DY = {"language": 11.0, "vision": 3.0, "action": -8.0}  # points; the lines meet at 16 bits
FORMAT_FAMILIES = {"precision-language", "format-language"}
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


def nominal_bits(candidate: str, recipe: dict, module: str) -> int:
    """Nominal bit width of one module's weights: the recipe's change, else the cell's format."""
    for change in recipe.get("changes") or []:
        if change.get("scope") in SCOPES[module]:
            bits = change.get("bits")
            return int(bits) if bits is not None else BITS_BY_NAME[change["method"]]
    return BITS_BY_NAME.get(candidate.rsplit("-", 1)[-1], REFERENCE_BITS)


def _prefer(candidate: str, incumbent: str) -> bool:
    """NF4 is the reference 4-bit format, so it stands for a width that several formats share."""
    return candidate.endswith("-nf4") and not incumbent.endswith("-nf4")


def module_points(n: dict, authored: dict, module: str) -> list[dict]:
    """One development point per nominal width for a module, plus the uncompressed reference."""
    points: dict[int, dict] = {}
    for candidate, row in n["rows"].items():
        if row.get("family") not in FAMILIES[module] or "development_success_pct" not in row:
            continue
        bits = nominal_bits(candidate, authored.get(candidate) or {}, module)
        if bits in points and not _prefer(candidate, points[bits]["candidate"]):
            continue
        points[bits] = {
            "candidate": candidate,
            "bits": bits,
            "success_pct": row["development_success_pct"],
            "wilson_pct": row["development_wilson_pct"],
        }
    reference = n["rows"].get("B16") or {}
    if "development_success_pct" in reference and REFERENCE_BITS not in points:
        points[REFERENCE_BITS] = {
            "candidate": "B16",
            "bits": REFERENCE_BITS,
            "success_pct": reference["development_success_pct"],
            "wilson_pct": reference["development_wilson_pct"],
        }
    return [points[bits] for bits in sorted(points)]


def _reference(n: dict, ax: plt.Axes) -> dict | None:
    """B16 development success as a dashed line with its Wilson band; returns the record."""
    b16 = n["rows"].get("B16") or {}
    if "development_success_pct" not in b16:
        return None
    low, high = b16["development_wilson_pct"]
    ax.axhspan(low, high, color=tint(MUTED, 0.86), lw=0, zorder=0)
    ax.axhline(b16["development_success_pct"], color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
    return {
        "candidate": "B16",
        "success_pct": b16["development_success_pct"],
        "wilson_pct": b16["development_wilson_pct"],
    }


def fig_precision(n: dict) -> dict | None:
    """Development success against nominal weight bits, one line per module; None without curves."""
    authored = recipes()
    series = {module: module_points(n, authored, module) for module in MODULES}
    if not any(len(points) >= 2 for points in series.values()):
        return None
    joint = n["rows"].get(JOINT[0]) or {}
    position = {bits: float(index) for index, bits in enumerate(BIT_TICKS)}
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    fig.subplots_adjust(left=0.105, right=0.995, top=0.975, bottom=0.155)
    ledger = {"reference": _reference(n, ax), "series": {}, "joint": None}

    for module in MODULES:
        points = [point for point in series[module] if point["bits"] in position]
        if not points:
            continue
        x = [position[point["bits"]] + OFFSETS[module] for point in points]
        y = [point["success_pct"] for point in points]
        # The band already carries the reference interval, so its repeated point needs no bar.
        low = [
            0.0 if point["candidate"] == "B16" else point["success_pct"] - point["wilson_pct"][0]
            for point in points
        ]
        high = [
            0.0 if point["candidate"] == "B16" else point["wilson_pct"][1] - point["success_pct"]
            for point in points
        ]
        ax.errorbar(
            x,
            y,
            yerr=[low, high],
            color=COLORS[module],
            marker="o",
            ms=3,
            mfc=COLORS[module],
            mec="white",
            mew=0.7,
            lw=1.1,
            elinewidth=0.7,
            capsize=0,
            zorder=3,
        )
        # The three lines meet at 16 bits, so their names fan out from the shared right end.
        ax.annotate(
            module,
            (x[-1], y[-1]),
            xytext=(5.0, LABEL_DY[module]),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.5,
            color=COLORS[module],
            annotation_clip=False,
        )
        ledger["series"][module] = {
            point["candidate"]: {
                "nominal_bits": point["bits"],
                "success_pct": point["success_pct"],
                "wilson_pct": point["wilson_pct"],
            }
            for point in points
        }

    if "development_success_pct" in joint:
        x_joint = position[BIT_TICKS[0]] + JOINT_OFFSET
        y_joint = joint["development_success_pct"]
        low, high = joint["development_wilson_pct"]
        ax.errorbar(
            x_joint,
            y_joint,
            yerr=[[y_joint - low], [high - y_joint]],
            color=COLORS["joint"],
            marker="D",
            ms=3.4,
            mfc="white",
            mec=COLORS["joint"],
            mew=0.9,
            lw=0,
            elinewidth=0.7,
            capsize=0,
            zorder=4,
        )
        ax.annotate(
            JOINT[1],
            (x_joint, y_joint),
            xytext=(0.0, 14.0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=6.5,
            color=COLORS["joint"],
            annotation_clip=False,
        )
        ledger["joint"] = {
            "candidate": JOINT[0],
            "nominal_bits": BIT_TICKS[0],
            "success_pct": y_joint,
            "wilson_pct": joint["development_wilson_pct"],
        }

    ax.set(xlim=(-0.6, 5.05), ylim=(-3, 104))
    ax.set_xticks(range(len(BIT_TICKS)))
    ax.set_xticklabels([str(bits) for bits in BIT_TICKS])
    style_axes(
        ax,
        ylabel="Development success (%)",
        xlabel="Weight bits",
        yticks=[0, 25, 50, 75, 100],
    )

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
