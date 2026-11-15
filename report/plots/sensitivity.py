"""Module sensitivity to bit width and structural removal, from 10-trial smokes (Figure 3)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.analysis import wilson
from report.plots.data import COMP, sensitivity_dev_failures, smoke
from report.plots.style import COLORS, FULL, INK_2, MUTED, OUT, panel_label, style_axes, tint


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
        gridspec_kw={"wspace": 0.14, "width_ratios": [1.08, 1.0, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.23)

    # (a) weight bit width; best of tested variants at each width, all variants shown faintly.
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
            a.scatter(
                [b + offsets[comp_name]] * len(vals),
                vals,
                s=7,
                color=COLORS[comp_name],
                alpha=0.32,
                lw=0,
                zorder=2,
            )
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
    a.text(1.9, 10.2, "B16 8/10 (95% CI)", fontsize=6.5, color=MUTED, va="top", ha="left")
    a.text(
        16.6,
        8.65,
        "vision blocks",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=COLORS["vision"],
        fontweight="medium",
    )
    a.text(
        16.6,
        7.25,
        "action MLP",
        ha="right",
        va="top",
        fontsize=6.5,
        color=COLORS["action"],
        fontweight="medium",
    )
    a.text(
        2.55,
        4.3,
        "language\ndecoder",
        ha="left",
        va="center",
        fontsize=6.5,
        color=COLORS["language"],
        fontweight="medium",
    )
    a.text(
        17.0,
        -0.2,
        "How to read: decoder breaks\nfirst; vision & action hold.",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=INK_2,
        style="italic",
    )

    # (b-d) structural removal per module; series told apart by line style + direct label.
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
    end_labels = {
        ("language", "MLP channels"): ("MLP\nchannels", 4, 0.4, "left", "center"),
        ("language", "KV heads"): ("KV\nheads", -3, -0.5, "right", "center"),
        ("language", "layers"): ("layers", 4, 0.0, "left", "center"),
        ("vision", "MLP channels"): ("MLP\nchannels", 4, 0.4, "left", "bottom"),
        ("vision", "heads"): ("heads", 4, 1.3, "left", "bottom"),
        ("vision", "layers"): ("layers", -14, 1.6, "right", "center"),
        ("action", "MLP channels"): ("MLP\nchannels", 4, 0.5, "left", "bottom"),
        ("action", "heads"): ("heads", 4, -0.3, "left", "top"),
        ("action", "layers"): ("layers", 4, 0.0, "left", "center"),
    }
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
            label, dx, dy, ha, va = end_labels[(comp_name, name)]
            ax.text(
                xs[-1] * 100 + dx,
                ys[-1] + dy,
                label,
                ha=ha,
                va=va,
                fontsize=6.5,
                color=COLORS[comp_name],
            )
            ledger["structure"][comp_name][name] = dict(
                zip([round(x, 3) for x in xs], ys, strict=True)
            )
        ax.set(xticks=[0, 25, 50, 75], xlim=(-4, 128), ylim=(-0.8, 10.6))
        style_axes(ax, yticks=[0, 2, 4, 6, 8, 10])
        ax.tick_params(axis="y", labelleft=False)
        panel_label(ax, "bcd"[list(struct_series).index(comp_name)], comp_name.capitalize())

    pos_b = axes[1].get_position()
    pos_d = axes[3].get_position()
    fig.text(
        0.5 * (pos_b.x0 + pos_d.x1),
        0.04,
        "Fraction removed (%)",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=INK_2,
    )

    callout = (
        f"100-trial dev fails (<77):\n"
        f"act heads -75%: {dev_fails['action_heads_75']}\n"
        f"merger -75%: {dev_fails['merger_75']}\n"
        f"combined NF4: {dev_fails['combined_channels']} & {dev_fails['combined_depth']}"
    )
    axes[3].text(
        0.02,
        0.04,
        callout,
        transform=axes[3].transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=INK_2,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#F8FAFC",
            "edgecolor": "#CBD5E1",
            "linewidth": 0.6,
        },
        zorder=4,
    )

    for ax in axes:
        ax.axhspan(ci_lo, ci_hi, color=tint(MUTED, 0.86), lw=0, zorder=0)
        ax.axhline(base, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)

    fig.savefig(OUT / "sensitivity.pdf")
    plt.close(fig)
    return ledger
