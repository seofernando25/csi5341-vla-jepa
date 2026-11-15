"""Offline action change versus closed-loop smoke success (Figure 4)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from report.plots.style import COL, COLORS, GRID, INK, INK_2, MUTED, OUT, style_axes


def fig_divergence(div: pd.DataFrame, rows: dict) -> dict | None:
    """Offline arm change vs closed-loop success; None until screen and smoke cells exist."""
    if len(div) < 3:  # a rank correlation needs more than two recipes
        return None
    floor = 2e-8
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    fig.subplots_adjust(left=0.145, right=0.985, top=0.89, bottom=0.225)

    groups = ["language", "vision", "action", "joint", "other"]
    for comp_name in groups:
        d = div[div.component == comp_name]
        x = d.arm_mse.clip(lower=floor)
        y = d.smoke + rng.uniform(-0.18, 0.18, len(d))
        ax.scatter(
            x,
            y,
            s=8.5,
            color=COLORS[comp_name],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )

    rho, p = spearmanr(div.arm_mse, div.smoke)
    dev = div.assign(
        development_successes=[rows.get(c, {}).get("development_successes") for c in div.candidate]
    ).dropna(subset=["development_successes"])
    if len(dev) >= 3:
        rho_dev, p_dev = spearmanr(dev.arm_mse, dev.development_successes)
        dev_note = f"dev ρ = {rho_dev:.2f} (n = {len(dev)})"
    else:
        rho_dev, p_dev = float("nan"), float("nan")
        dev_note = f"dev ρ not computed (n = {len(dev)})"
    clean = div[div.arm_mse < 1e-4]
    noisy = div[div.arm_mse >= 1e-3]
    clean_ge7 = int((clean.smoke >= 7).sum())
    noisy_ge7 = int((noisy.smoke >= 7).sum())

    ax.set_xscale("log")
    ax.set(ylim=(-0.5, 10.3), xlim=(1e-8, 3e-1), xticks=[1e-8, 1e-6, 1e-4, 1e-2])
    ax.minorticks_off()
    style_axes(
        ax,
        ylabel="Successes / 10 trials",
        xlabel="Offline arm-action change vs B16 (MSE, 80 frames)",
        yticks=[0, 2, 4, 6, 8, 10],
    )

    ax.axvline(floor * 1.6, color=GRID, lw=0.7, zorder=1)
    ax.axvline(1e-4, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axvline(1e-3, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)

    x_tags = [0.00, 0.21, 0.36, 0.51, 0.62]
    for x_frac, comp_name in zip(x_tags, groups, strict=True):
        ax.text(
            x_frac,
            1.03,
            f"● {comp_name}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            fontweight="medium",
            color=COLORS[comp_name],
        )

    halo = {"facecolor": "white", "alpha": 0.90, "edgecolor": "none", "pad": 1.0}
    ax.annotate(
        "identical actions",
        xy=(floor * 1.1, 0.42),
        xytext=(4.3e-8, 0.12),
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.7, "shrinkA": 1, "shrinkB": 0},
        fontsize=6.5,
        color=MUTED,
        ha="left",
        va="bottom",
        style="italic",
        bbox=halo,
        zorder=4,
    )
    ax.text(
        7.5e-5,
        9.95,
        f"<1e-4: {clean_ge7}/{len(clean)} ≥7/10",
        ha="right",
        va="top",
        fontsize=6.5,
        color=INK,
        fontweight="medium",
    )
    ax.text(
        1.3e-3,
        9.95,
        f"≥1e-3: {noisy_ge7}/{len(noisy)} ≥7/10",
        ha="left",
        va="top",
        fontsize=6.5,
        color=INK,
        fontweight="medium",
    )
    ax.text(
        0.085,
        0.64,
        f"Spearman ρ = {rho:.2f} (n = {len(div)})\n{dev_note}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=INK,
        bbox=halo,
        zorder=4,
    )
    ax.text(
        0.085,
        0.41,
        "How to read: screens (>1e-3),\ncannot rank near-lossless.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=INK_2,
        style="italic",
        bbox=halo,
        zorder=4,
    )

    fig.savefig(OUT / "divergence.pdf")
    plt.close(fig)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n": len(div),
        "rho_gripper": float(spearmanr(div.gripper, div.smoke)[0]),
        "development": (
            {"n": len(dev), "spearman_rho": float(rho_dev), "p_value": float(p_dev)}
            if len(dev) >= 3
            else {"n": len(dev), "spearman_rho": None, "p_value": None}
        ),
        "below_1e-4": {
            "n": len(clean),
            "mean_smoke": float(clean.smoke.mean()) if len(clean) else None,
            "share_ge7": float((clean.smoke >= 7).mean()) if len(clean) else None,
        },
        "above_1e-3": {
            "n": len(noisy),
            "mean_smoke": float(noisy.smoke.mean()) if len(noisy) else None,
            "share_ge7": float((noisy.smoke >= 7).mean()) if len(noisy) else None,
        },
    }
