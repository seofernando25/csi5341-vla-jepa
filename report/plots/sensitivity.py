"""Module sensitivity to bit width and structural removal, from 10-trial smokes (Figure 4)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from report.plots.data import COMP, smoke
from report.plots.style import COLORS, FULL, MUTED, OUT


def fig_sensitivity() -> dict:
    comp = pd.read_csv(COMP / "summary.csv").set_index("candidate")
    base = smoke(comp, "B16")
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.75), gridspec_kw={"wspace": 0.32})
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
    offsets = {"language": -0.12, "vision": 0.0, "action": 0.12}
    for comp_name, table in bits.items():
        xs, best = [16], [base]
        for b, names in table.items():
            vals = [smoke(comp, nm) for nm in names if not np.isnan(smoke(comp, nm))]
            a.scatter(
                [b + offsets[comp_name]] * len(vals),
                vals,
                s=7,
                color=COLORS[comp_name],
                alpha=0.35,
                lw=0,
            )
            xs.append(b)
            best.append(max(vals))
        mk, ms, fill = {
            "language": ("o", 3.2, COLORS["language"]),
            "vision": ("s", 5, "white"),
            "action": ("o", 2.6, COLORS["action"]),
        }[comp_name]
        a.plot(
            np.array(xs) + offsets[comp_name],
            best,
            "-",
            marker=mk,
            ms=ms,
            mfc=fill,
            lw=1.1,
            color=COLORS[comp_name],
            label={
                "language": "language decoder",
                "vision": "vision blocks",
                "action": "action MLP",
            }[comp_name],
        )
        ledger["bits"][comp_name] = dict(zip(xs, best, strict=True))
    a.set_xscale("log", base=2)
    a.set(
        xticks=[2, 3, 4, 8, 16],
        xticklabels=["2", "3", "4", "8", "16"],
        ylim=(-0.5, 10.5),
        xlabel="Weight bits",
        ylabel="Successes / 10 trials",
    )
    a.minorticks_off()
    a.set_title("(a) Bit width", loc="left")
    a.legend(loc="lower right", handlelength=1.2, borderaxespad=0.1, fontsize=6)
    # (b–d) structural removal per module.
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
    styles = {
        "MLP channels": ("-", "o"),
        "KV heads": ("--", "s"),
        "heads": ("--", "s"),
        "layers": (":", "^"),
    }
    hollow = {"KV heads", "heads"}
    for ax, (comp_name, series) in zip(axes[1:], struct_series.items(), strict=True):
        ledger["structure"][comp_name] = {}
        for name, pts in series.items():
            xs = [0.0] + [x for x, _ in pts]
            ys = [base] + [smoke(comp, c) for _, c in pts]
            ls, mk = styles[name]
            ax.plot(
                np.array(xs) * 100,
                ys,
                ls=ls,
                marker=mk,
                ms=4.6 if name in hollow else 3,
                mfc="white" if name in hollow else COLORS[comp_name],
                lw=1.1,
                color=COLORS[comp_name],
                label=name,
            )
            ledger["structure"][comp_name][name] = dict(
                zip([round(x, 3) for x in xs], ys, strict=True)
            )
        ax.set(xlim=(-3, 78), ylim=(-0.5, 10.5), xlabel="Fraction removed (%)")
        ax.set_title(
            f"({'bcd'[list(struct_series).index(comp_name)]}) {comp_name.capitalize()}", loc="left"
        )
        ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.1, fontsize=6)
    for ax in axes:
        ax.axhline(base, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=0)
    fig.savefig(OUT / "sensitivity.pdf")
    plt.close(fig)
    return ledger
