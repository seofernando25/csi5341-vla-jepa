"""Shared figure style: IEEE column widths, palette and Matplotlib settings."""

import matplotlib as mpl

from evaluation.common import ROOT

OUT = ROOT / "report" / "figures"
INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
COLORS = {
    "B16": "#1f2933",
    "Q8": "#d97706",
    "Q4": "#2563eb",
    "C1": "#059669",
    "C2": "#7c3aed",
    "C3": "#db2777",
    "language": "#2563eb",
    "vision": "#059669",
    "action": "#d97706",
    "joint": "#7c3aed",
    "other": "#9aa5b1",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
COL, FULL = 3.45, 7.1  # IEEE single and double column widths, inches
