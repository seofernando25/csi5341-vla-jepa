"""Offline action change versus closed-loop smoke success (Figure 6)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from report.plots.data import COMP
from report.plots.style import COL, COLORS, GRID, INK, MUTED, OUT


def fig_divergence(div: pd.DataFrame) -> dict:
    floor = 2e-8
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    for comp_name in ["language", "vision", "action", "joint", "other"]:
        d = div[div.component == comp_name]
        x = d.arm_mse.clip(lower=floor)
        y = d.smoke + rng.uniform(-0.18, 0.18, len(d))
        ax.scatter(x, y, s=9, color=COLORS[comp_name], alpha=0.75, lw=0, label=comp_name)
    rho, p = spearmanr(div.arm_mse, div.smoke)
    ax.set_xscale("log")
    ax.set(
        ylim=(-0.6, 10.2),
        xlabel="Offline arm-action change vs B16 (MSE, 80 fixed frames)",
        ylabel="Successes / 10 trials",
    )
    ax.axvline(floor * 1.6, color=GRID, lw=0.6)
    ax.text(floor * 1.1, 0.0, "identical\nactions", fontsize=5.8, color=MUTED, ha="left")
    ax.text(
        0.98,
        0.97,
        f"Spearman ρ = {rho:.2f}, n = {len(div)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color=INK,
    )
    ax.legend(
        loc="lower left",
        ncol=5,
        handletextpad=0.1,
        columnspacing=0.6,
        fontsize=6,
        bbox_to_anchor=(0, 1.0, 1, 0.1),
        mode="expand",
        borderaxespad=0.2,
    )
    fig.savefig(OUT / "divergence.pdf")
    plt.close(fig)
    comp = pd.read_csv(COMP / "summary.csv").set_index("candidate")
    dev = div.join(comp.development_successes, on="candidate").dropna(
        subset=["development_successes"]
    )
    rho_dev, p_dev = spearmanr(dev.arm_mse, dev.development_successes)
    clean = div[div.arm_mse < 1e-4]
    noisy = div[div.arm_mse >= 1e-3]
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n": len(div),
        "rho_gripper": float(spearmanr(div.gripper, div.smoke)[0]),
        "development": {"n": len(dev), "spearman_rho": float(rho_dev), "p_value": float(p_dev)},
        "below_1e-4": {
            "n": len(clean),
            "mean_smoke": float(clean.smoke.mean()),
            "share_ge7": float((clean.smoke >= 7).mean()),
        },
        "above_1e-3": {
            "n": len(noisy),
            "mean_smoke": float(noisy.smoke.mean()),
            "share_ge7": float((noisy.smoke >= 7).mean()),
        },
    }
