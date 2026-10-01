"""Consistent vector figures for the paper; result plots require measured inputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLORS = {"B16": "#264D73", "Q8": "#007F79", "Q4": "#C77928", "S500": "#80549A"}
ORDER = list(COLORS)


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#66717C",
            "axes.linewidth": 0.6,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": "#E4E8EC",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
        }
    )


def save(fig, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png"]:
        fig.savefig(directory / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture(directory):
    style()
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ink, muted = "#253547", "#596774"
    ax.text(
        0, 0.99, "A   Policy architecture and supervision", color=ink, weight="bold", fontsize=11
    )

    def box(x, y, w, h, text, edge=ink, fill="#F2F5F7", size=9):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.009",
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.8,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=ink,
            linespacing=1.35,
        )

    def arrow(start, end, dashed=False):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                color=muted,
                linewidth=0.9,
                linestyle="--" if dashed else "-",
            )
        )

    box(0.01, 0.65, 0.17, 0.16, "Current images\n+ instruction")
    box(0.23, 0.65, 0.23, 0.16, "Vision–language\nbackbone", edge=COLORS["B16"])
    box(0.51, 0.65, 0.16, 0.16, "Conditioning\ninterface")
    box(0.72, 0.65, 0.17, 0.16, "Flow-matching\naction head")
    ax.text(0.97, 0.73, "Action\nchunk", ha="center", va="center", fontsize=9, color=ink)
    for x1, x2 in [(0.18, 0.23), (0.46, 0.51), (0.67, 0.72), (0.89, 0.94)]:
        arrow((x1, 0.73), (x2, 0.73))
    ax.text(0.345, 0.595, "Qwen3-VL / SmolVLM2", ha="center", fontsize=8, color=muted)
    ax.text(0.59, 0.88, "SmolVLM:\nlinear projection", ha="center", fontsize=8, color=muted)
    ax.text(0.805, 0.895, "Robot state", ha="center", fontsize=8.5, color=muted)
    arrow((0.805, 0.875), (0.805, 0.815))
    ax.add_patch(
        FancyBboxPatch(
            (0.003, 0.255),
            0.985,
            0.24,
            boxstyle="round,pad=0.008",
            facecolor="#FAFAFB",
            edgecolor="#B5BEC6",
            linestyle="--",
            linewidth=0.7,
        )
    )
    ax.text(0.018, 0.46, "TRAINING ONLY", fontsize=7.7, color=muted, weight="bold")
    box(0.015, 0.295, 0.17, 0.115, "Training video", size=8.5)
    box(0.23, 0.295, 0.23, 0.115, "Frozen V-JEPA 2\nvideo encoder", size=8.5)
    box(0.51, 0.295, 0.20, 0.115, "Latent-state\npredictor", size=8.5)
    box(0.77, 0.295, 0.19, 0.115, "Latent prediction\nloss", size=8.5)
    arrow((0.185, 0.352), (0.23, 0.352), True)
    arrow((0.46, 0.352), (0.51, 0.352), True)
    arrow((0.71, 0.352), (0.77, 0.352), True)
    arrow((0.615, 0.65), (0.615, 0.415), True)
    ax.plot(
        [0.46, 0.475, 0.475, 0.865],
        [0.352, 0.352, 0.445, 0.445],
        color=muted,
        linewidth=0.8,
        linestyle="--",
    )
    arrow((0.865, 0.445), (0.865, 0.415), True)
    ax.text(
        0.50,
        0.215,
        "Video context and shifted targets are used only during training.",
        ha="center",
        color=muted,
        fontsize=8.4,
    )
    ax.text(0, 0.135, "B   Matched deployment comparison", color=ink, weight="bold", fontsize=11)
    for i, key in enumerate(ORDER):
        x = 0.02 + i * 0.25
        ax.scatter([x], [0.058], s=35, color=COLORS[key])
        label = {
            "B16": "Qwen · BF16",
            "Q8": "Qwen · 8-bit",
            "Q4": "Qwen · 4-bit",
            "S500": "SmolVLM · BF16",
        }[key]
        ax.text(x + 0.02, 0.058, label, va="center", fontsize=8.9, color=ink)
    save(fig, directory, "architecture")
