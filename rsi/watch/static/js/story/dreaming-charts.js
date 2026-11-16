/* Chart figures for the Dreaming chapter: probes vs attainment, and V. */

import { el, svg } from "../dom.js";
import { PALETTE, swatch, shape, fixed, verdictWord } from "./dreaming-figures.js";

const DASH = ["none", "6 3", "1 4", "7 3 1.5 3"];
const OFFSET = [0, 2, 1, 4];

export function curvesFigure(model, worldId, selected) {
  const candidates = model.candidates || [];
  const series = candidates
    .map((candidate, index) => ({
      candidate,
      index,
      points: (model.curves?.[candidate.slot]?.[worldId] || [])
        .filter((point) => point && Number.isFinite(Number(point.probes)) && point.attainment != null)
        .map((point) => ({ probes: Number(point.probes), attainment: Number(point.attainment), beta: Number(point.beta) }))
        .sort((a, b) => a.beta - b.beta),
    }))
    .filter((item) => item.points.length);
  const figure = el("figure", { class: "figure dream-curves" }, [
    el("div", { class: "figure-head" }, [
      el("span", {}, "Probes vs attainment"),
      el("span", { class: "dream-head-meta" }, `${worldId} · ${series.length} candidates`),
    ]),
  ]);
  if (!series.length) {
    figure.append(el("p", { class: "dream-note" }, "No replay curves recorded for this world."));
    return figure;
  }
  const width = 760;
  const height = 280;
  const box = { left: 46, right: 16, top: 18, bottom: 34 };
  const maxProbes = Math.max(1, ...series.flatMap((item) => item.points.map((point) => point.probes)));
  const x = (value) => box.left + (value / maxProbes) * (width - box.left - box.right);
  const y = (value) => height - box.bottom - Math.max(0, Math.min(1, value)) * (height - box.top - box.bottom);
  const parts = [];
  for (const level of [0, 0.5, 1]) {
    parts.push(svg("line", { class: "dream-grid", x1: box.left, x2: width - box.right, y1: y(level), y2: y(level) }));
    parts.push(svg("text", { class: "dream-tick", x: box.left - 8, y: y(level) + 3, "text-anchor": "end" }, level.toFixed(1)));
  }
  for (let value = 0; value <= maxProbes; value += maxProbes <= 6 ? 1 : 2) {
    parts.push(svg("text", { class: "dream-tick", x: x(value), y: height - box.bottom + 16, "text-anchor": "middle" }, String(value)));
  }
  parts.push(svg("line", { class: "dream-axis", x1: box.left, x2: width - box.right, y1: height - box.bottom, y2: height - box.bottom }));
  parts.push(svg("line", { class: "dream-axis", x1: box.left, x2: box.left, y1: box.top, y2: height - box.bottom }));
  parts.push(svg("text", { class: "dream-axis-label", x: width - box.right, y: height - 4, "text-anchor": "end" }, "probes used"));
  parts.push(svg("text", { class: "dream-axis-label", x: box.left, y: 10 }, "attainment"));
  for (const item of [...series].sort((a, b) => Number(a.candidate.slot === selected) - Number(b.candidate.slot === selected))) {
    const on = item.candidate.slot === selected;
    parts.push(
      svg("polyline", {
        class: "dream-curve",
        points: item.points.map((point) => `${x(point.probes)},${y(point.attainment)}`).join(" "),
        style: {
          stroke: PALETTE[item.index % PALETTE.length],
          strokeWidth: on ? 2.8 : 1.4,
          opacity: on ? 1 : 0.8,
          strokeDasharray: DASH[item.index % DASH.length],
          strokeDashoffset: OFFSET[item.index % OFFSET.length],
        },
      }),
    );
    for (const point of item.points) {
      parts.push(shape(item.index, x(point.probes), y(point.attainment), on ? 5.8 : 5.2 - item.index * 0.8, on));
    }
  }
  figure.append(
    el("div", { class: "dream-inline-legend" }, series.map((item) =>
      el("span", { class: "dream-inline-item", dataset: { selected: String(item.candidate.slot === selected) } }, [
        swatch(item.index, 10),
        el("span", {}, item.candidate.slot || "—"),
      ]),
    )),
  );
  figure.append(el("div", { class: "dream-plot" }, svg("svg", { class: "dream-chart-svg", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Probes used against attainment, one line per candidate" }, parts)));
  figure.append(el("p", { class: "figure-cap" }, "Each point is one beta: probes spent against attainment. Markers and dashes separate candidates; higher and further left is better."));
  return figure;
}

/* --- V ------------------------------------------------------------------ */

export function vFigure(model, selected) {
  const candidates = model.candidates || [];
  const max = candidates.reduce((top, candidate) => Math.max(top, Number(candidate.mean_v) || 0), 0);
  const scale = Math.max(0.1, Math.ceil(max * 10) / 10);
  const rows = candidates.map((candidate) => {
    const value = Number(candidate.mean_v);
    const width = Number.isFinite(value) ? Math.max(0, Math.min(1, value / scale)) * 100 : 0;
    const delta = candidate.delta_vs_best == null ? "—" : `${Number(candidate.delta_vs_best) > 0 ? "+" : ""}${fixed(candidate.delta_vs_best)}`;
    return el("div", { class: "dream-vrow", dataset: { selected: String(candidate.slot === selected), status: candidate.status || "" } }, [
      el("span", { class: "dream-vslot" }, candidate.slot || "—"),
      el("span", { class: "dream-vtrack" }, el("i", { class: "dream-vfill", style: { width: `${width}%` } })),
      el("span", { class: "dream-vval" }, fixed(candidate.mean_v)),
      el("span", { class: "dream-vdelta" }, delta),
      el("span", { class: "dream-vtag" }, verdictWord(candidate.status)),
    ]);
  });
  const worlds = (model.worlds || []).length;
  const lambda = Number(model.protocol?.lambda_parallel);
  const penalty = Number.isFinite(lambda) ? ` − ${lambda} × parallel penalty` : "";
  return el("figure", { class: "figure dream-v" }, [
    el("div", { class: "figure-head" }, [el("span", {}, "V"), el("span", { class: "dream-head-meta" }, `mean over ${worlds} worlds`)]),
    el("div", { class: "dream-vhead" }, ["candidate", "", "V", "Δ at step", "verdict"].map((label) => el("span", {}, label))),
    ...rows,
    el("p", { class: "figure-cap" }, `V = hypervolume${penalty}, mean over ${worlds} worlds. Delta against the best V when the step was replayed.`),
  ]);
}
