"""Module sensitivity to bit width and structural removal, from 10-trial smokes (Figure 3)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from evaluation.analysis import wilson
from report.plots.data import COMP, sensitivity_dev_failures, smoke
from report.plots.style import COLORS, FULL, INK_2, MUTED, OUT, SPINE, panel_label, style_axes, tint


def fig_sensitivity(dev_fails: dict | None = None) -> dict:
    if dev_fails is None:
        dev_fails = sensitivity_dev_failures()
    comp = pd.read_csv(COMP / "summary.csv").set_index("candidate")
    base = smoke(comp, "B16")
    ci_lo, ci_hi = (10.0 * v for v in wilson(int(base), 10))

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FULL, 1.8),
        gridspec_kw={"wspace": 0.14, "width_ratios": [1.05, 1.0, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.065, right=0.80, top=0.77, bottom=0.24)

    # (a) weight bit width; best of tested variants at each width.
    bits = {
        "language": {
            8: ["text-bnb8"],
            4: ["text-bnb4", "text-hqq4-axis1-pytorch"],
            3: ["text-hqq3-axis0-pytorch", "text-hqq3-axis1-pytorch"],
            2: ["text-hqq2-axis0-pytorch", "text-hqq2-axis1-pytorch", "text-hqq2-axis1-gemlite"],
        },
        "vision": {
            8: ["vision_blocks-bnb8"],
            4: ["vision_blocks-bnb4", "vision_blocks-hqq4-axis1-pytorch"],
            3: ["vision_blocks-hqq3-axis0-pytorch", "vision_blocks-hqq3-axis1-pytorch"],
            2: [
                "vision_blocks-hqq2-axis0-pytorch",
                "vision_blocks-hqq2-axis1-pytorch",
                "vision_blocks-hqq2-axis1-gemlite",
            ],
        },
        "action": {
            8: ["action_mlp-bnb8"],
            4: ["action_mlp-bnb4", "action_mlp-hqq4-axis1-pytorch"],
            3: ["action_mlp-hqq3-axis0-pytorch", "action_mlp-hqq3-axis1-pytorch"],
            2: [
                "action_mlp-hqq2-axis0-pytorch",
                "action_mlp-hqq2-axis1-pytorch",
                "action_mlp-hqq2-axis1-gemlite",
            ],
        },
    }
    ledger: dict = {"bits": {}, "structure": {}}
    a = axes[0]
    offsets = {"language": -0.10, "vision": 0.0, "action": 0.10}
    for comp_name, table in bits.items():
        xs, best = [16], [base]
        for b, names in table.items():
            vals = [smoke(comp, nm) for nm in names if not np.isnan(smoke(comp, nm))]
            xs.append(b)
            best.append(max(vals))
        a.plot(
            np.array(xs) + offsets[comp_name],
            best,
            "-",
            marker="o",
            ms=3,
            mfc=COLORS[comp_name],
            mec="white",
            mew=0.7,
            lw=1.35,
            color=COLORS[comp_name],
            zorder=3,
        )
        ledger["bits"][comp_name] = dict(zip(xs, best, strict=True))
    a.set_xscale("log", base=2)
    a.set(
        xticks=[2, 3, 4, 8, 16],
        xticklabels=["2", "3", "4", "8", "16"],
        xlim=(1.7, 17.8),
        ylim=(-0.8, 10.6),
    )
    a.minorticks_off()
    style_axes(a, ylabel="Successes / 10 trials", xlabel="Weight bits", yticks=[0, 2, 4, 6, 8, 10])
    panel_label(a, "a", "Bit width")
    a.text(1.85, 10.5, "B16 8/10 (CI)", fontsize=6.5, color=MUTED, va="top", ha="left")
    a.text(
        16.6,
        8.35,
        "vision blocks",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=COLORS["vision"],
        fontweight="medium",
    )
    a.text(
        16.6,
        7.55,
        "action MLP",
        ha="right",
        va="top",
        fontsize=6.5,
        color=COLORS["action"],
        fontweight="medium",
    )
    a.text(
        4.0,
        6.5,
        "language\ndecoder",
        ha="left",
        va="top",
        fontsize=6.5,
        color=COLORS["language"],
        fontweight="medium",
    )
    a.text(
        17.0,
        -0.4,
        "How to read:\ndecoder breaks first;\nvision & action hold.",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=INK_2,
        style="italic",
    )

    # (b-d) structural removal per module; shared legend strip above panels b-d.
    struct_series = {
        "language": {
            "MLP channels": [
                (0.125, "text_mlp-channels-125"),
                (0.25, "text_mlp-channels-250"),
                (0.5, "text_mlp-channels-500"),
                (0.75, "text_mlp-channels-750"),
            ],
            "KV heads": [
                (0.125, "text-heads-125"),
                (0.25, "text-heads-250"),
                (0.5, "text-heads-500"),
            ],
            "layers": [
                (1 / 28, "text-layers-1"),
                (4 / 28, "text-layers-4"),
                (8 / 28, "text-layers-8"),
                (20 / 28, "text-layers-20"),
            ],
        },
        "vision": {
            "MLP channels": [
                (0.125, "vision_mlp-channels-125"),
                (0.5, "vision_mlp-channels-500"),
                (0.75, "vision_mlp-channels-750"),
            ],
            "heads": [
                (0.125, "vision_attention-heads-125"),
                (0.5, "vision_attention-heads-500"),
                (0.75, "vision_attention-heads-750"),
            ],
            "layers": [
                (1 / 24, "vision-layers-1"),
                (8 / 24, "vision-layers-8"),
                (12 / 24, "vision-layers-12"),
                (16 / 24, "vision-layers-16"),
            ],
        },
        "action": {
            "MLP channels": [(0.125, "action_mlp-channels-125"), (0.75, "action_mlp-channels-750")],
            "heads": [(0.125, "action_attention-heads-125"), (0.75, "action_attention-heads-750")],
            "layers": [
                (1 / 16, "action-layers-1"),
                (8 / 16, "action-layers-8"),
                (12 / 16, "action-layers-12"),
            ],
        },
    }
    line_style = {"MLP channels": "-", "KV heads": "--", "heads": "--", "layers": ":"}
    for ax, (comp_name, series) in zip(axes[1:], struct_series.items(), strict=True):
        ledger["structure"][comp_name] = {}
        for name, pts in series.items():
            xs = [0.0] + [x for x, _ in pts]
            ys = [base] + [smoke(comp, c) for _, c in pts]
            ax.plot(
                np.array(xs) * 100,
                ys,
                ls=line_style[name],
                marker="o",
                ms=3,
                mfc=COLORS[comp_name],
                mec="white",
                mew=0.7,
                lw=1.3,
                color=COLORS[comp_name],
                zorder=3,
            )
            ledger["structure"][comp_name][name] = dict(
                zip([round(x, 3) for x in xs], ys, strict=True)
            )
        ax.set(xticks=[0, 25, 50, 75], xlim=(-5, 82), ylim=(-0.8, 10.6))
        style_axes(ax, yticks=[0, 2, 4, 6, 8, 10])
        ax.tick_params(axis="y", labelleft=False)
        panel_label(ax, "bcd"[list(struct_series).index(comp_name)], comp_name.capitalize())

    pos_b = axes[1].get_position()
    pos_d = axes[3].get_position()
    mid_bcd = 0.5 * (pos_b.x0 + pos_d.x1)

    handles = [
        Line2D([0], [0], color=INK_2, lw=1.3, ls="-", label="MLP channels"),
        Line2D([0], [0], color=INK_2, lw=1.3, ls="--", label="heads (KV in b)"),
        Line2D([0], [0], color=INK_2, lw=1.3, ls=":", label="layers"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(mid_bcd, 0.995),
        ncol=3,
        fontsize=6.5,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.2,
    )

    # Span bracket + centered x-label across panels b-d.
    fig.add_artist(
        Line2D(
            [pos_b.x0, pos_b.x0, pos_d.x1, pos_d.x1],
            [0.145, 0.125, 0.125, 0.145],
            transform=fig.transFigure,
            color=SPINE,
            lw=0.6,
            clip_on=False,
        )
    )
    fig.text(
        mid_bcd,
        0.04,
        "Fraction removed (%)",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=INK_2,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )

    callout = (
        f"100-trial development,\ngate 77:\n"
        f"act heads −75%: {dev_fails['action_heads_75']}\n"
        f"merger −75%: {dev_fails['merger_75']}\n"
        f"comb. NF4: {dev_fails['combined_channels']} & {dev_fails['combined_depth']}"
    )
    fig.text(
        0.818,
        0.50,
        callout,
        ha="left",
        va="center",
        fontsize=6.5,
        color=INK_2,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#F8FAFC",
            "edgecolor": "#CBD5E1",
            "linewidth": 0.6,
        },
    )

    for ax in axes:
        ax.axhspan(ci_lo, ci_hi, color=tint(MUTED, 0.86), lw=0, zorder=0)
        ax.axhline(base, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)

    fig.savefig(OUT / "sensitivity.pdf")
    plt.close(fig)
    return ledger
