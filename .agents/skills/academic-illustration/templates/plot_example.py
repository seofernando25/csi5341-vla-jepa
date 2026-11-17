"""Template: a column-width line plot in the house style (direct labels, no legend)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.style.use(HERE / "paper.mplstyle")
BLUE, ORANGE, ORANGE_TXT = "#0072B2", "#D55E00", "#A84800"  # *_TXT passes 4.5:1 for text

fig, ax = plt.subplots(figsize=(3.3, 2.2))  # column width; height 2.0-2.6 in
x = [1, 2, 3, 4, 5]
ax.plot(x, [2, 3, 5, 6, 8], color=BLUE, marker="o", mec="white", mew=0.6)
ax.plot(x, [1, 2, 2, 3, 4], color=ORANGE, marker="o", mec="white", mew=0.6)
for y, text, color in ((8, "method A", BLUE), (4, "method B", ORANGE_TXT)):
    ax.annotate(text, (x[-1], y), xytext=(5, 0), textcoords="offset points", va="center",
                color=color, annotation_clip=False)
ax.set(ylim=(0, 9), xlabel="Budget (hours)", ylabel="Bugs found")
fig.savefig(HERE / "plot_example.pdf")
