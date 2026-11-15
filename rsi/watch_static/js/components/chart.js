/* Dependency-free SVG score chart: one dot per scored node, best-so-far step line.
   update(container, {points:[{x,score,best,cell}], highlight, gate?}); gate is a 0..1
   threshold in score units. Sizes to the container width; rebuilds only when data changes. */

import { svg } from "../dom.js";
import * as fmt from "../format.js";

const HEIGHT = 120;
const PAD = { top: 10, right: 8, bottom: 18, left: 30 };
const TICKS = [0, 0.25, 0.5, 0.75, 1];
const MIN_WIDTH = 180;
const LABEL_STEP = 38;

export function createChart() {
  let host = null;
  let node = null;
  let data = null;
  let sig = "";
  let width = 0;
  let observer = null;

  function attach(container) {
    if (observer) observer.disconnect();
    observer = null;
    host = container || null;
    node = null;
    sig = "";
    if (!host || typeof ResizeObserver !== "function") return;
    observer = new ResizeObserver((entries) => {
      const next = Math.round(entries[0]?.contentRect.width || 0);
      if (next && next !== width) draw(data);
    });
    observer.observe(host);
  }

  function draw(payload) {
    if (!host) return;
    const d = payload || { points: [] };
    const points = Array.isArray(d.points) ? d.points.filter(Boolean) : [];
    const w = Math.max(MIN_WIDTH, Math.round(host.clientWidth || 0));
    const key = [
      w,
      d.highlight || "",
      d.gate == null ? "" : d.gate,
      points.map((p) => `${p.x}|${p.score}|${p.best}|${p.cell}`).join(","),
    ].join("~");
    if (key === sig && node) return;
    sig = key;
    width = w;

    const plotW = w - PAD.left - PAD.right;
    const plotH = HEIGHT - PAD.top - PAD.bottom;
    const xs = points.map((p, i) => (Number.isFinite(Number(p.x)) ? Number(p.x) : i));
    const lo = xs.length ? Math.min(...xs) : 0;
    const hi = xs.length ? Math.max(...xs) : 0;
    const px = (i) => (hi === lo ? PAD.left + plotW / 2 : PAD.left + (plotW * (xs[i] - lo)) / (hi - lo));
    const py = (value) => PAD.top + plotH * (1 - clamp01(value));
    const scored = points.filter((p) => p.score != null && !Number.isNaN(Number(p.score)));
    const best = scored.reduce((acc, p) => Math.max(acc, Number(p.best ?? p.score)), 0);

    const root = svg("svg", {
      class: "chart-svg",
      viewBox: `0 0 ${w} ${HEIGHT}`,
      width: "100%",
      height: String(HEIGHT),
      preserveAspectRatio: "xMidYMid meet",
      role: "img",
      "aria-label": scored.length
        ? `Score chart: ${scored.length} scored of ${points.length} nodes, best ${fmt.pct(best)}`
        : "Score chart: no scored nodes",
    });

    for (const tick of TICKS) {
      const y = py(tick);
      root.append(
        svg("line", {
          x1: PAD.left,
          x2: w - PAD.right,
          y1: y,
          y2: y,
          class: tick === 0 ? "chart-axis" : "chart-grid",
        }),
      );
      root.append(
        svg(
          "text",
          { x: PAD.left - 5, y: y + 3, class: "chart-tick", "text-anchor": "end" },
          `${Math.round(tick * 100)}%`,
        ),
      );
    }

    if (d.gate != null && Number.isFinite(Number(d.gate))) {
      const gy = py(d.gate);
      const line = svg("line", { x1: PAD.left, x2: w - PAD.right, y1: gy, y2: gy, class: "chart-gate" });
      line.append(svg("title", {}, `gate ${fmt.pct(d.gate)}`));
      root.append(line);
      root.append(
        svg(
          "text",
          { x: w - PAD.right, y: gy - 3, class: "chart-gate-label", "text-anchor": "end" },
          `gate ${fmt.pct(d.gate)}`,
        ),
      );
    }

    let path = "";
    let prevY = null;
    points.forEach((point, index) => {
      if (point.best == null) return;
      const x = px(index);
      const y = py(point.best);
      path = prevY == null ? `M${x} ${y}` : `${path} L${x} ${prevY} L${x} ${y}`;
      prevY = y;
    });
    if (path) root.append(svg("path", { d: path, class: "chart-best" }));

    const hlIndex = points.findIndex((p) => d.highlight && p.cell === d.highlight);
    if (hlIndex >= 0) {
      const point = points[hlIndex];
      const hasScore = point.score != null && !Number.isNaN(Number(point.score));
      const x = px(hlIndex);
      root.append(
        svg("line", {
          x1: x,
          x2: x,
          y1: py(0),
          y2: hasScore ? py(point.score) : PAD.top,
          class: "chart-guide",
        }),
      );
      if (!hasScore) {
        const marker = svg("circle", { cx: x, cy: py(0), r: 3.4, class: "chart-dot is-missing" });
        marker.append(svg("title", {}, `${point.cell} · not scored`));
        root.append(marker);
      }
    }

    points.forEach((point, index) => {
      if (point.score == null || Number.isNaN(Number(point.score))) return;
      const isHighlight = Boolean(d.highlight) && point.cell === d.highlight;
      const cx = px(index);
      const cy = py(point.score);
      if (isHighlight) root.append(svg("circle", { cx, cy, r: 7.5, class: "chart-halo" }));
      const dot = svg("circle", {
        cx,
        cy,
        r: isHighlight ? 4 : 2.4,
        class: isHighlight ? "chart-dot is-hl" : "chart-dot",
      });
      const bestNote = point.best != null ? ` · best ${fmt.pct(point.best)}` : "";
      dot.append(svg("title", {}, `${point.cell} · ${fmt.pct(point.score)}${bestNote}`));
      root.append(dot);
    });

    const step = points.length > 1 ? Math.abs(px(1) - px(0)) : plotW;
    if (points.length && step >= LABEL_STEP) {
      const single = points.length === 1;
      points.forEach((point, index) => {
        const first = !single && index === 0;
        const last = !single && index === points.length - 1;
        root.append(
          svg(
            "text",
            {
              x: single ? px(index) : first ? PAD.left : last ? w - PAD.right : px(index),
              y: HEIGHT - 5,
              class: "chart-xtick",
              "text-anchor": single ? "middle" : first ? "start" : last ? "end" : "middle",
            },
            point.cell,
          ),
        );
      });
    }

    if (!scored.length) {
      root.append(
        svg(
          "text",
          { x: PAD.left + plotW / 2, y: PAD.top + plotH / 2, class: "chart-empty", "text-anchor": "middle" },
          "no scored nodes",
        ),
      );
    }

    if (node && node.parentNode === host) node.replaceWith(root);
    else if (host) host.append(root);
    node = root;
  }

  return {
    update(container, payload) {
      data = payload || { points: [] };
      if (container !== host) attach(container);
      draw(data);
    },
    destroy() {
      if (observer) observer.disconnect();
      observer = null;
      host = null;
      node = null;
      data = null;
      sig = "";
    },
  };
}

function clamp01(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}
