"""Rebuild the compact offline action-accuracy figure from measured CSVs."""

from __future__ import annotations

import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.common import ROOT


def main():
    root = ROOT / "studies/recovery"
    names = ["b16-action-audit", "n0008-10k-action-audit"]
    if (root / 'diagnostics/rgb-2k-action-audit/actions.csv').exists():
        names.append('rgb-2k-action-audit')
    rows = {
        name: [r for r in csv.DictReader((root / "diagnostics" / name / "actions.csv").open())
               if r["split"] == "heldout"]
        for name in names
    }
    if [r["row"] for r in rows["b16-action-audit"]] != [r["row"] for r in rows["n0008-10k-action-audit"]]:
        raise ValueError("Action diagnostics must use identical selected frames")
    values = [np.sqrt([float(r["arm_mse"]) for r in rows["b16-action-audit"]]),
              np.sqrt([float(r["arm_mse"]) for r in rows["n0008-10k-action-audit"]])]
    labels, colors = ['Qwen\nB16', 'n0008\nlegacy 10k'], ['#1f5f94', '#bd5e3b']
    if 'rgb-2k-action-audit' in rows:
        if [(r['row'], r['episode'], r['frame'], r['valid_actions']) for r in rows['rgb-2k-action-audit']] != [
                (r['row'], r['episode'], r['frame'], r['valid_actions']) for r in rows['n0008-10k-action-audit']]:
            raise ValueError('Recovery and legacy action frames differ')
        values.append(np.sqrt([float(r['arm_mse']) for r in rows['rgb-2k-action-audit']]))
        labels.append('n0008\nRGB +2k')
        colors.append('#65528c')
    values.append(np.sqrt([float(r['mean_arm_mse']) for r in rows['n0008-10k-action-audit']]))
    labels.append('Constant\nmean action')
    colors.append('#77838f')
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for i, (vals, color) in enumerate(zip(values, colors)):
        ax.scatter(i + np.linspace(-.12, .12, len(vals)), vals, s=22, color=color,
                   alpha=.65, linewidths=.4, edgecolors="white", zorder=3)
        rms = np.sqrt(np.mean(vals**2))
        ax.plot([i - .18, i + .18], [rms, rms], color=color, lw=2.6, zorder=4)
        ax.annotate(f"{rms:.4f}", (i + .21, rms), ha="left", va="center", fontsize=10, color=color)
    ax.set_yscale("log")
    ax.set_ylabel("Arm-command RMSE (simulator inputs)")
    ax.set_xticks(range(len(values)), labels)
    ax.set_xlim(-.5, len(values) - .35)
    ax.grid(axis="y", which="major", color="#dce1e7", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#bac4ce")
    ax.tick_params(length=3, color="#bac4ce")
    fig.text(.12, .018, "20 matched held-out frames · bars: RMSE over frame-wise MSE · no task-success inference",
             fontsize=9, color="#596776")
    fig.tight_layout(rect=[0, .055, 1, 1])
    output = root / "figures"
    output.mkdir(exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        path = output / f"F7_action_audit.{suffix}"
        fig.savefig(path, dpi=180)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
