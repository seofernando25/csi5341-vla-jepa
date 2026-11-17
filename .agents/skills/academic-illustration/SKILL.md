---
name: academic-illustration
description: Design and draw research-paper illustrations (TikZ explainers, overview figures, matplotlib plots) in a clean research-lab style, and avoid the layout mistakes this kit was built to prevent. Use whenever creating, redrawing or polishing a figure for a report or paper, when a figure looks bland, crowded or misaligned, or when turning a table into a chart.
---

# Academic illustration

Figures explain the method faster than prose. A reader should get the idea of a figure from its
shapes, numbers and arrows before reading the caption. This skill gives the house style, a TikZ
kit, templates, and the mistakes to check for before calling a figure done.

Templates in `templates/` (copy next to the figure source):

| File | Use |
|---|---|
| `figkit.tex` | Palette, fonts, badges, chips, and the explainer kit (`\StagePanel`, `\CompareFooter`, `item`, `outcome`, `flow`, `elab`, `cyl`, `pill`) |
| `explainer.tex` | Three numbered stages (baseline, derivation, variant) feeding one comparison |
| `overview.tex` | Page-1 overview: input, numbered levers, one result card per lever |
| `paper.mplstyle`, `plot_example.py` | Column-width plot with direct labels |

`figkit.tex` uses `fontspec` (build with XeLaTeX). Under pdfLaTeX, drop its four font lines and
load `newtxtext` in the document instead; the TikZ styles are engine-independent.

## Style

- **Structure first.** Number the stages with badges (1, 2, 3) in reading order. A tinted panel
  per stage, a bold title on one header row, an italic one-line subtitle, then content.
- **Role colours, fixed across all figures of a paper.** Baseline = blue, derivation or neutral
  step = grey, transformed variant = orange. Good outcome = forest green, failure/bug = garnet, and
  garnet means only that.
- **Two nesting levels at most.** Tinted panel, then white items. Never a white card inside a
  dashed white card inside a panel.
- **One width per column.** All items in a panel share `text width` (set `\itemw` once), so left
  and right edges line up. Centre-align items in the panel.
- **Place by relation, not by guess.** First item `at (P-in)`, the rest `below=1.6mm of prev`.
  Shared coordinates with `calc`: `($(A)!0.5!(B)$)`, `(A |- B)`. Hardcoded y values drift when a
  badge or subtitle changes height.
- **Arrows carry flow.** Shaft at least 3 mm; head `Stealth[length=1.8mm]`; route between outer
  card edges (east to west, south to north), never through text. Edge labels are one or two
  words and must fit the gap they sit in.
- **One decision point per comparison.** Fan in with one bus, one `equal?` diamond, two pills.
- **Icons** (`fontawesome5`): one per concept, in the accent blue or ink, next to a word. Never
  decorative, never alone.
- **Fonts.** Figure labels sans (Liberation Sans or Helvetica), code in mono, captions and
  prose in the document serif. Labels at least 7 pt at final size.
- **Contrast.** All text at least 4.5:1 on its fill. No grey text lighter than ink at 75 %.
  Orange text uses the dark variant (`vartext` `#A84800`), never `#D9822B`.
- **Plots.** Horizontal grid replaces the y spine; no y tick marks; labels at line ends instead
  of legends; panel letters `a`, `b` with a short title; step curves for cumulative counts;
  column width (3.3-3.5 in), height 2.0-2.6 in. Bars start at zero.
- **Tables or charts.** A chart when the shape is the point (a trend, a trade-off such as tests
  against plans). Keep a table for a few exact small counts.
- **Captions.** Bold first sentence that names what is shown, not a claim. Then source and
  caveat. Do not restate the numbers already printed in the figure.

## Mistakes caught while building this kit

Check each one on the rendered PNG, not the source.

1. **Text overflow.** A subtitle or code line wider than its panel. Fix with `text width`
   derived from the panel width, or shorten the words. Long subtitles wrap and push content
   down, so place content below the subtitle node, not at a fixed y.
2. **Badges pushing titles into content.** A badge taller than the title line moves the title
   down onto the first item. Use the small `stepbadge` and the kit's fixed header row.
3. **Nested boxes.** A dashed wrapper around white cards reads as clutter. Delete the wrapper.
4. **Ragged item widths.** Cards with natural widths in one column look misaligned. One
   `text width` per column.
5. **Short arrows with big heads.** Gaps under 3 mm leave only an arrowhead. Space the boxes.
6. **Elements outside their panel.** A table or chip placed at an absolute y can poke out of a
   panel bottom. Re-check panel heights after every content change; size siblings equally.
7. **Edge labels wider than the gap** ("substitute" in a 1 cm gap). Use one short word.
8. **Shared colour names redefined locally.** A figure that redefines `accent` turns every badge
   orange. Use `stepblue` for badges and never redefine kit colours in a figure.
9. **Garnet used for a neutral fact** ("removed"). Garnet is the bug colour only.
10. **Legends over data, labels on top of points, labels clipped at the figure edge.** Use
    direct labels with offsets, then render and look.
11. **Two close points with one label between them.** Put each label on the outside of its own
    point.
12. **Stats under a diagram** ("5.6 GB, 73 ms") that the text already says. Cut them.
13. **Slogan captions** ("Only X breaks."). Describe what the figure shows.
14. **Truncated bar axes** to inflate small gains. Bars start at zero; zoom only on dot or line
    plots, and say so.
15. **Build traps.** `\big` is a TeX macro (do not name a style or macro after it); a blank line
    inside `tikzpicture[...]` options ends the paragraph; `xcolor` mixes like `ink!55!4` fail;
    `\n` written by a script into LaTeX becomes an undefined command; a greedy regex edit can
    delete half a file, so restore from git and edit exact strings.
16. **Stale output.** XeLaTeX can write a PDF after an error. Build with `&&`, check the exit
    code and `grep '^!' *.log`, then re-render the page.

## Review loop

1. Build the figure, render at 120-150 dpi, and look at the whole figure.
2. Zoom to 250-300 dpi on titles, the smallest text and every arrow.
3. Rebuild the document, check page count and overfull boxes, and look at the figure on its page.
4. Run the checklist above. Fix every hit before moving to the next figure.
