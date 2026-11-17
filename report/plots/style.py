"""Shared figure style: IEEE column widths, editorial palette, Inter typography.

Tufte/Nature-inspired: charcoal text (never pure black), light horizontal gridlines
that do the work of the y-axis, no y tick marks, bold panel letters with medium-weight
titles, and direct labels instead of legends where possible.
"""

from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgb

from evaluation.common import ROOT

OUT = ROOT / "report" / "figures"
FONT_DIR = Path(__file__).parent / "fonts"

INK, INK_2, MUTED, GRID, SPINE = "#1A1A1A", "#475569", "#64748B", "#E2E8F0", "#94A3B8"
COLORS = {
    "B16": "#334155",
    "Q8": "#D97706",
    "Q4": "#2563EB",
    "C1": "#BE185D",
    "C2": "#0891B2",
    "language": "#2563EB",
    "vision": "#059669",
    "action": "#D97706",
    "joint": "#7C3AED",
    "other": "#94A3B8",
}


def _luminance(rgb: tuple[float, ...]) -> float:
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def text_ink(color: str, contrast: float = 4.5) -> str:
    """`color` darkened until text in it reaches WCAG AA contrast on white (4.5:1)."""
    rgb = to_rgb(color)
    for step in range(21):
        shade = tuple(c * (1 - step * 0.04) for c in rgb)
        if 1.05 / (_luminance(shade) + 0.05) >= contrast:
            return to_hex(shade)
    return INK


def tint(color: str, amount: float = 0.45) -> str:
    """Mix `color` with white; `amount` is the white share in [0, 1]."""
    return to_hex(tuple(c + (1 - c) * amount for c in to_rgb(color)))


@lru_cache(maxsize=1)
def register_fonts() -> None:
    """Register the vendored Inter TTFs (OFL) with matplotlib's font manager."""
    for ttf in sorted(FONT_DIR.glob("Inter-*.ttf")):
        font_manager.fontManager.addfont(str(ttf))


register_fonts()
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Open Sans", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Inter",
        "mathtext.it": "Inter:italic",
        "mathtext.bf": "Inter:bold",
        "mathtext.cal": "Inter",
        "mathtext.sf": "Inter",
        "mathtext.tt": "Inter",
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "xtick.major.pad": 2.5,
        "ytick.major.size": 0,
        "ytick.major.pad": 3,
        "axes.labelpad": 3,
        "legend.frameon": False,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.02,
    }
)
COL, FULL = 3.45, 7.1  # IEEE single and double column widths, inches


def style_axes(ax: plt.Axes, ylabel: str | None = None, xlabel: str | None = None, yticks=None):
    """Light horizontal grid, left spine removed (gridlines do the work), labels only where given."""
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.spines["left"].set_visible(False)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def panel_label(ax: plt.Axes, letter: str, title: str = "", x: float = 0.0, y: float = 1.02):
    """Bold panel letter at axes fraction (x, y), then a medium-weight title to its right."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8.5,
        va="bottom",
        ha="left",
    )
    if title:
        ax.annotate(
            title,
            (x, y),
            xycoords="axes fraction",
            xytext=(11, 0),
            textcoords="offset points",
            fontweight="medium",
            fontsize=7.5,
            color=INK,
            va="bottom",
            ha="left",
        )


def direct_label(ax: plt.Axes, x: float, y: float, text: str, color: str, dx=4, dy=0, **kw):
    """Label right of point (x, y) in data coords, offset (dx, dy) points; never clipped."""
    kw.setdefault("fontsize", 7)
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        va="center",
        ha="left",
        annotation_clip=False,
        **kw,
    )
