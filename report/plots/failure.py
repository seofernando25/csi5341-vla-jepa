"""Representation drift and action error of the language-decoder cells (Figure 5).

`evaluation/compression/failure.py` replays the screen's 80 fixed frames for B16 and for the four
language-decoder cells and commits `studies/program-v1/failure/failure.json`. This module turns
that record into one single-column figure: linear CKA against B16 at the decoder's last layer (all
tokens and image tokens only) and at the action head's conditioning input, then the mean physical
arm action error on the same frames. Both panels share the weight-bit axis, so a cell's
representation drift and its action error sit in one column.

Diagnostics only: neither panel is LIBERO success.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from report.plots.data import STUDY, missing, optional_json, rel
from report.plots.style import COL, COLORS, INK_2, MUTED, OUT, style_axes

ORDER = (16, 8, 4, 3, 2)  # the bfloat16 reference, then the four cells by nominal weight bits
SERIES = (
    ("cka_decoder", "decoder, all tokens", COLORS["language"]),
    ("cka_image", "decoder, image tokens", COLORS["vision"]),
    ("cka_conditioning", "action head input", COLORS["action"]),
)
FLOOR = 2e-8  # B16 acts identically to itself; a log axis needs a finite stand-in
TASK_PHASES = ("development", "reserved")


def failure_record(study: Path | None = None) -> dict:
    """Compact per-cell record of the committed failure analysis, or a status stub."""
    root = Path(study) if study is not None else STUDY
    path = root / "failure" / "failure.json"
    record = optional_json(path)
    if record is None or record.get("status") != "completed":
        return missing(path, "failure analysis")
    cells = []
    for cell in record.get("cells", []):
        cka, actions = cell.get("cka") or {}, cell.get("actions") or {}
        attention = cell.get("attention") or {}
        cells.append(
            {
                "cell": cell["cell"],
                "nominal_bits": cell["nominal_bits"],
                "cka_decoder": cka["all"][-1],
                "cka_image": cka["image"][-1],
                "cka_text": cka["text"][-1],
                "cka_conditioning": cka["conditioning"],
                "cka_layers": cka["all"],
                "mean_arm_mse": actions["mean_arm_mse"],
                "median_arm_mse": actions["median_arm_mse"],
                "max_arm_mse": actions["max_arm_mse"],
                "frames_with_gripper_flip": actions["frames_with_gripper_flip"],
                "gripper_steps_disagreeing": actions["gripper_steps_disagreeing"],
                "gripper_steps": actions["gripper_steps"],
                "attention_mean_mass": attention.get("mean_mass"),
                "attention_spearman_median": attention.get("spearman_median"),
                "reproduction": cell.get("reproduction"),
            }
        )
    return {
        "status": "ok",
        "source": rel(path),
        "frames": (record.get("protocol") or {}).get("frames"),
        "reference": record.get("reference"),
        "cells": cells,
        "tasks": {
            phase: {
                "reference": entry.get("reference"),
                "cells": entry.get("cells"),
            }
            for phase, entry in (record.get("tasks") or {}).items()
            if phase in TASK_PHASES
        },
        "does_not_show": record.get("does_not_show", []),
    }


def fig_failure(record: dict) -> dict | None:
    """CKA and action error against weight bits, single column; None without the analysis."""
    cells = [cell for cell in record.get("cells") or [] if cell["nominal_bits"] in ORDER]
    if len(cells) < 2:
        return None
    position = {bits: index for index, bits in enumerate(ORDER)}
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(COL, 2.55), sharex=True, gridspec_kw={"height_ratios": [1.0, 0.85]}
    )
    fig.subplots_adjust(left=0.135, right=0.985, top=0.975, bottom=0.185, hspace=0.22)

    ordered = sorted(cells, key=lambda c: -c["nominal_bits"])
    for key, label, color in SERIES:
        top.plot(
            [position[bits] for bits in ORDER],
            [1.0] + [cell[key] for cell in ordered],
            color=color,
            lw=1.1,
            marker="o",
            ms=3,
            mfc=color,
            mec="white",
            mew=0.7,
        )

    errors = [cell["mean_arm_mse"] for cell in cells]
    low, high = max(min(errors), FLOOR), max(errors)
    bottom.plot(
        [position[bits] for bits in ORDER[1:]],
        [cell["mean_arm_mse"] for cell in ordered],
        color=MUTED,
        lw=1.1,
        marker="o",
        ms=3,
        mfc=MUTED,
        mec="white",
        mew=0.7,
    )
    bottom.set_yscale("log")

    top.set(
        ylim=(
            min(0.9, min(cell[key] for cell in cells for key, _, _ in SERIES) - 0.03),
            1.005,
        )
    )
    bottom.set(xlim=(-0.4, len(ORDER) - 0.6), ylim=(low / 3, high * 3))
    bottom.set_xticks(range(len(ORDER)))
    bottom.set_xticklabels(["16"] + [str(bits) for bits in ORDER[1:]])
    style_axes(top, ylabel="Similarity (CKA)")
    style_axes(bottom, ylabel="Action error (MSE)", xlabel="Language-decoder weight bits")
    top.legend(
        handles=[
            plt.Line2D([], [], color=color, lw=1.1, marker="o", ms=3, label=label)
            for _, label, color in SERIES
        ],
        loc="lower left",
        frameon=False,
        fontsize=6.2,
        handlelength=1.1,
        handletextpad=0.35,
        borderaxespad=0.2,
        labelcolor=INK_2,
    )
    fig.savefig(OUT / "failure.pdf")
    plt.close(fig)
    return {"cells": [{key: cell[key] for key in ("cell", "nominal_bits")} for cell in cells]}
