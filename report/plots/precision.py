"""Precision response surface (Figure 1).

The surface plots development success against nominal weight bits per module: one line per module,
the uncompressed reference at 16 bits, and the joint vision+action 2-bit cell beside the
single-module 2-bit cells. Nominal bits come from the cell's recipe, or from its format suffix
when the recipe records none; where a module has several 4-bit formats, NF4 stands for the line.
Points are measured development rollouts, never pooled with smoke.
"""

import matplotlib.pyplot as plt

from report.plots.data import recipes
from report.plots.style import COL, COLORS, MUTED, OUT, style_axes, tint

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
LABEL_DY = {"language": 18.0, "vision": 6.0, "action": -12.0}  # points; the lines meet at 16 bits


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
            xytext=(-4.0, LABEL_DY[module]),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7.5,
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
        xlabel="Weight bits (categories, not to scale)",
        yticks=[0, 25, 50, 75, 100],
    )

    fig.savefig(OUT / "precision.pdf")
    plt.close(fig)
    return ledger
