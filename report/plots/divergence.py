"""Offline action error against closed-loop success, one phase at a time (Figure 4).

The question is whether the offline action error of a compressed policy — its mean physical
arm-action change against B16 on the 80 held-out screen frames — predicts LIBERO success. Cells
come from two sources and are never pooled:

- precision-surface cells: `measurements/<recipe>/screen/run.json` holds the action error and
  `measurements/<recipe>/{smoke,development}/run.json` the successes;
- Dream-RSI nodes: `cycles/<cycle>/nodes/<cell>/{score,development}.json` hold successes but no
  action error, so they are inventoried and excluded, with the reason recorded.

Phases stay separate because a 10-trial smoke result and a 100-trial development result answer
different questions. The figure draws the development panel only; the smoke record stays in
`numbers.json` and the rank statistics belong to the caption. The module takes the study directory
as an argument, so it runs unchanged on `studies/program-v2`; `divergence_record` returns the
compact record and `fig_divergence` plots it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from evaluation.analysis import wilson
from evaluation.common import ROOT
from report.plots.data import component_of, optional_json
from report.plots.style import COL, COLORS, INK_2, MUTED, OUT, style_axes

PHASES = ("smoke", "development")
PLOT_PHASE = "development"
ACTION_ERROR_FIELD = "mean_physical_arm_change_mse"
ACTION_ERROR_FRAMES = 80
DEFAULT_GATES = {"smoke": 8, "development": 77}
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
FLOOR = 2e-8  # B16 acts identically to itself; a log axis needs a finite stand-in


def gates(study_dir: Path) -> dict:
    """Pre-registered success gate per phase, from `protocol.json`, else the defaults."""
    recorded = (optional_json(study_dir / "protocol.json") or {}).get("gates") or {}
    return {
        "smoke": int(recorded.get("smoke_min_successes", DEFAULT_GATES["smoke"])),
        "development": int(recorded.get("development_min_successes", DEFAULT_GATES["development"])),
    }


def _label(study_dir: Path) -> str:
    """Repository-relative study path, or the directory name when it sits outside the repo."""
    try:
        return study_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return study_dir.name


def _source(study_dir: Path, path: Path) -> str:
    """Path relative to the study directory, so no machine path enters the record."""
    return path.relative_to(study_dir).as_posix()


def _entry(
    cell_id: str,
    phase: str,
    action_error: float | None,
    successes: int,
    trials: int,
    gate_min: int,
    component: str,
    sources: list[str],
    reason: str | None = None,
) -> dict:
    """One (cell, phase) pair with its Wilson interval and gate verdict."""
    low, high = wilson(successes, trials)
    entry = {
        "cell_id": cell_id,
        "phase": phase,
        "action_error": action_error,
        "successes": successes,
        "trials": trials,
        "success_pct": 100 * successes / trials,
        "wilson_pct": [100 * low, 100 * high],
        "gate_min_successes": gate_min,
        "gate_passed": successes >= gate_min,
        "component": component,
        "included": action_error is not None,
        "sources": sources,
    }
    if reason is not None:
        entry["reason"] = reason
    return entry


def surface_cells(study_dir: Path, gate: dict) -> list[dict]:
    """Precision-surface cells: screen action error paired with each rollout phase."""
    root = study_dir / "measurements"
    families = {
        path.stem: (optional_json(path) or {}).get("family")
        for path in sorted((study_dir / "recipes").glob("*.json"))
    }
    cells = []
    for screen_path in sorted(root.glob("*/screen/run.json")):
        recipe = screen_path.parent.parent.name
        screen = optional_json(screen_path) or {}
        summary = screen.get("summary") or {}
        error = summary.get(ACTION_ERROR_FIELD)
        if screen.get("status") != "completed" or error is None:
            continue
        for phase in PHASES:
            rollout_path = root / recipe / phase / "run.json"
            rollout = optional_json(rollout_path) or {}
            counts = rollout.get("summary") or {}
            if rollout.get("status") != "completed" or counts.get("successes") is None:
                continue
            cells.append(
                _entry(
                    recipe,
                    phase,
                    float(error),
                    int(counts["successes"]),
                    int(counts["episodes"]),
                    gate[phase],
                    "B16" if recipe == "B16" else component_of(recipe, families.get(recipe)),
                    [_source(study_dir, screen_path), _source(study_dir, rollout_path)],
                )
            )
    return cells


def node_cells(study_dir: Path, gate: dict) -> list[dict]:
    """Dream-RSI nodes: successes exist, action error does not, so they are excluded."""
    cells = []
    for score_path in sorted((study_dir / "cycles").glob("*/nodes/*/score.json")):
        node = score_path.parent
        for phase, name in (("smoke", "score.json"), ("development", "development.json")):
            path = node / name
            record = optional_json(path) or {}
            successes, trials = record.get("successes"), record.get("episodes")
            if successes is None or not trials:
                continue
            cells.append(
                _entry(
                    record.get("cell_id") or node.name,
                    phase,
                    None,
                    int(successes),
                    int(trials),
                    gate[phase],
                    "other",
                    [_source(study_dir, path)],
                    reason="no action error recorded: the evaluator's rollout phases run no screen",
                )
            )
    return cells


def rank_stats(cells: list[dict]) -> dict:
    """Spearman rho of action error against success rate, with a percentile bootstrap interval.

    Resampling is over cells, which is the unit of the correlation. A resample with constant
    error or constant success has no rank correlation and is dropped; the interval is withheld
    when more than a tenth of the resamples are dropped.
    """
    n = len(cells)
    if n < 3:
        return {"n": n, "rho": None, "ci95": None, "p_value": None, "note": "fewer than 3 cells"}
    x = np.array([cell["action_error"] for cell in cells], dtype=float)
    y = np.array([cell["successes"] / cell["trials"] for cell in cells], dtype=float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {
            "n": n,
            "rho": None,
            "ci95": None,
            "p_value": None,
            "note": "constant error or success; no ranking is possible",
        }
    rho, p_value = spearmanr(x, y)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        index = rng.integers(0, n, n)
        if np.ptp(x[index]) == 0 or np.ptp(y[index]) == 0:
            continue
        draws.append(float(spearmanr(x[index], y[index])[0]))
    ci95 = (
        np.quantile(draws, [0.025, 0.975]).tolist()
        if len(draws) >= 0.9 * BOOTSTRAP_RESAMPLES
        else None
    )
    return {
        "n": n,
        "rho": float(rho),
        "ci95": ci95,
        "p_value": float(p_value),
        "bootstrap_resamples": len(draws),
        "method": (
            f"Spearman rank correlation; percentile bootstrap over cells, "
            f"{BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}"
        ),
    }


def phase_stats(cells: list[dict], phase: str) -> dict:
    """Correlation over every included cell of a phase, then over its gate-passing cells only."""
    included = [cell for cell in cells if cell["phase"] == phase and cell["included"]]
    passing = [cell for cell in included if cell["gate_passed"]]
    return {
        "gate_min_successes": included[0]["gate_min_successes"] if included else None,
        "n_cells": len(included),
        "spearman": rank_stats(included),
        "gate_passing": {**rank_stats(passing), "n_cells": len(passing)},
    }


def divergence_record(study_dir: Path | str) -> dict:
    """Compact per-cell record and per-phase correlations for one study directory."""
    study_dir = Path(study_dir)
    gate = gates(study_dir)
    cells = surface_cells(study_dir, gate) + node_cells(study_dir, gate)
    return {
        "study": _label(study_dir),
        "action_error_field": ACTION_ERROR_FIELD,
        "action_error_frames": ACTION_ERROR_FRAMES,
        "phases": {phase: phase_stats(cells, phase) for phase in PHASES},
        "counts": {
            "included": sum(cell["included"] for cell in cells),
            "excluded": sum(not cell["included"] for cell in cells),
            "by_phase": {
                phase: {
                    "included": sum(c["included"] for c in cells if c["phase"] == phase),
                    "excluded": sum(not c["included"] for c in cells if c["phase"] == phase),
                }
                for phase in PHASES
            },
        },
        "cells": cells,
    }


def fig_divergence(record: dict) -> dict | None:
    """Development action error against success, single column.

    The smoke panel and the rank statistics belong to the caption, not the plot. The returned
    record keeps every phase with enough cells, so `numbers.json` carries both.
    """
    recorded = [phase for phase in PHASES if record["phases"][phase]["n_cells"] >= 3]
    if PLOT_PHASE not in recorded:
        return None
    stats = record["phases"][PLOT_PHASE]
    cells = [cell for cell in record["cells"] if cell["included"] and cell["phase"] == PLOT_PHASE]
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.965, bottom=0.215)
    errors = [cell["action_error"] for cell in cells]
    low, high = max(min(errors), FLOOR), max(errors)
    gate_pct = 100 * stats["gate_min_successes"] / cells[0]["trials"]

    for cell in cells:
        x = max(cell["action_error"], FLOOR)
        y = cell["success_pct"]
        interval = cell["wilson_pct"]
        ax.errorbar(
            x,
            y,
            yerr=[[y - interval[0]], [interval[1] - y]],
            color=COLORS[cell["component"]],
            marker="D" if cell["action_error"] == 0 else "o",
            ms=3.2,
            mfc="white" if cell["action_error"] == 0 else COLORS[cell["component"]],
            mec=COLORS[cell["component"]],
            mew=0.8,
            lw=0,
            elinewidth=0.6,
            capsize=0,
            zorder=3,
        )
    ax.axhline(gate_pct, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(
        high * 2.2,
        gate_pct + 1.5,
        "gate",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=MUTED,
        style="italic",
    )
    ax.set_xscale("log")
    ax.set(xlim=(low / 2.5, high * 2.6), ylim=(-4, 112))
    ax.set_xticks([FLOOR, 1e-6, 1e-4, 1e-2])
    ax.set_xticklabels(["0", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"])
    ax.minorticks_off()
    style_axes(
        ax,
        ylabel="Development success (%)",
        xlabel="Action error vs reference (MSE)",
        yticks=[0, 25, 50, 75, 100],
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="D" if name == "B16" else "o",
            ls="",
            ms=3.2,
            mfc="white" if name == "B16" else COLORS[name],
            mec=COLORS[name],
            mew=0.8,
            label=name,
        )
        for name in ("B16", "language", "vision", "action", "joint", "other")
        if any(cell["component"] == name for cell in cells)
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            frameon=False,
            fontsize=6.2,
            ncol=5,
            columnspacing=0.7,
            handlelength=0.9,
            handletextpad=0.3,
            borderaxespad=0.2,
            labelcolor=INK_2,
        )
    fig.savefig(OUT / "divergence.pdf")
    plt.close(fig)
    return {"phases": {phase: record["phases"][phase] for phase in recorded}}
