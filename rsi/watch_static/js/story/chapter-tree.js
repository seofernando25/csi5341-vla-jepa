/* Chapter 03: the discovery tree as a replay, with the frontier line beside it. */

import { clear, el, icon, prefersReducedMotion, setText, svg } from "../dom.js";
import { cycleOf, frontierSeries, nodeOrder } from "./model.js";

export const id = "tree";
export const number = "03";
export const title = "The tree grows";
export const dek = "The agent spends its probe budget on branches; each scored probe is a LIBERO rollout.";

const STEP_MS = 450;
const SHORT = { pending: "queued", agent: "agent", evaluating: "eval", stopped: "stopped" };

export function mount(root, ctx) {
  const { fmt } = ctx;
  let cycle = null, order = [], rows = [], scores = [], prefix = [];
  let budget = null, gate = null, count = 0, frame = 0, sig = "", timer = null, playing = false;

  const rowsBox = el("ol", { class: "disc-rows", attrs: { "aria-label": "Discovery tree" } });
  const axis = el("div", { class: "disc-axis", attrs: { "aria-hidden": "true" } }, [
    el("span", { class: "disc-tick" }, "100 %"), el("span", { class: "disc-tick" }, "50 %"), el("span", { class: "disc-tick" }, "0"),
  ]);
  const line = svg("polyline", { class: "disc-line", "vector-effect": "non-scaling-stroke" });
  const dots = el("div", { class: "disc-dots" });
  const empty = el("p", { class: "disc-empty" }, "no scored probes");
  const svgBox = svg("svg", { class: "disc-svg", viewBox: "0 0 100 100", preserveAspectRatio: "none", "aria-hidden": "true" }, [line]);
  const plot = el("div", { class: "disc-plot", attrs: { role: "img" } }, [svgBox, dots, empty]);
  const cycleLabel = el("span", { class: "disc-cycle" });
  const side = el("div", { class: "disc-side" }, [
    el("div", { class: "disc-side-head" }, [el("span", {}, "Best score"), el("span", {}, "probe index")]),
    el("div", { class: "disc-chart" }, [axis, plot]),
    el("div", { class: "disc-legend" }, [
      el("span", { class: "disc-key" }, [el("span", { class: "disc-key-mark" }), "best"]),
    ]),
  ]);

  const playIcon = icon("play", 12);
  const pauseIcon = pauseGlyph();
  const playText = el("span", {}, "Play");
  const playBtn = el("button", { class: "disc-play", type: "button", attrs: { "aria-label": "Play replay" } }, [playIcon, pauseIcon, playText]);
  const range = el("input", {
    class: "disc-range", type: "range", min: "0", max: "0", step: "1", value: "0",
    attrs: { "aria-label": "Replay frame" },
  });
  const counter = el("span", { class: "disc-count" }, "probe 0 / 0");
  const cap = el("p", { class: "figure-cap" });
  const figure = el("div", { class: "figure disc-figure" }, [
    el("div", { class: "figure-head" }, [el("span", {}, "Discovery tree"), cycleLabel]),
    el("div", { class: "disc-split" }, [rowsBox, side]),
    el("div", { class: "disc-controls" }, [playBtn, range, counter]), cap,
  ]);
  const stateLine = el("p", { class: "disc-state" });
  root.append(el("p", { class: "chapter-kicker" }, number), el("h2", { class: "chapter-title" }, title),
    el("p", { class: "chapter-dek" }, dek), figure, stateLine);

  function update(state) {
    const next = cycleOf(state);
    const nextOrder = next ? nodeOrder(next) || [] : [];
    const nextSig = signature(next, nextOrder);
    const atEnd = frame >= count - 1;
    if (nextSig !== sig) {
      cycle = next; order = nextOrder; sig = nextSig; count = order.length;
      buildRows();
      frame = atEnd ? Math.max(count - 1, 0) : Math.min(frame, Math.max(count - 1, 0));
      if (count < 2) halt();
    }
    budget = ctx.sel.probeBudget(state);
    gate = gateOf(state);
    scores = cleanSeries(next ? frontierSeries(next) : []);
    render(state);
  }

  function buildRows() {
    clear(rowsBox);
    rows = [];
    prefix = [];
    const depth = depthMap(cycle);
    let scored = 0;
    order.forEach((node, i) => {
      if (isScored(node)) scored += 1;
      prefix[i] = scored;
      const row = rowFor(node, depth.get(node.cell_id) || 0);
      rows.push(row);
      rowsBox.append(row);
    });
  }

  function rowFor(node, depth) {
    const scored = isScored(node);
    const width = scored ? clamp01(node.score) * 100 : 0;
    const state = node.cell_id === "root" ? "root" : scored ? "scored" : "hollow";
    return el("li", { class: "disc-row", dataset: { state, revealed: "false" }, style: { paddingLeft: `${Math.min(depth, 6) * 14}px` } }, [
      el("span", { class: "disc-dot", dataset: { tone: toneOf(node) }, attrs: { "aria-hidden": "true" } }),
      el("span", { class: "disc-id", title: node.cell_id }, node.cell_id),
      el("span", { class: "disc-bar" }, [el("span", { class: "disc-fill", style: { width: `${width}%` } })]),
      el("span", { class: "disc-score" }, scoreLabel(node, scored)),
      el("span", { class: "disc-smoke" }, smokeLabel(node)),
    ]);
  }

  function render(state) {
    const live = count >= 2;
    figure.hidden = !live;
    stateLine.hidden = live;
    if (!live) return setText(stateLine, stateText(state));
    const label = cycle ? (cycle.iteration != null ? `cycle ${cycle.iteration + 1}` : cycle.cycle_id) : "";
    setText(cycleLabel, `${label} · ${count} ${count === 1 ? "node" : "nodes"}`);
    const smoke = gate ? `Smoke is the ${gate.label.replace("/", "-of-")} gate.` : "Smoke is shown beside each bar.";
    setText(cap, `Root is the baseline; its score is not recorded. Bars are score: gate-qualified size reduction. ${smoke}`);
    paint();
  }

  function paint() {
    const shown = frame + 1;
    rows.forEach((row, i) => {
      const on = i < shown;
      row.dataset.revealed = String(on);
      row.setAttribute("aria-hidden", String(!on));
    });

    const visible = new Set(order.slice(0, shown).map((node) => node.cell_id));
    const points = scores.filter((point) => visible.has(point.cell));
    const domain = Math.max(budget || 0, scores.length, 1);
    const x = (i) => (domain > 1 ? (i / (domain - 1)) * 100 : 0);
    const y = (value) => clamp01(value) * 100;
    const coords = points.map((point, i) => [x(i), y(point.best)]);
    const path = coords.flatMap(([a, b], i) => (i ? [[a, coords[i - 1][1]], [a, b]] : [[a, b]]));
    line.setAttribute("points", path.map(([a, b]) => `${a.toFixed(2)},${(100 - b).toFixed(2)}`).join(" "));

    clear(dots);
    coords.forEach(([a, b], i) => {
      const last = String(i === coords.length - 1);
      dots.append(el("span", { class: "disc-point", dataset: { last }, style: { left: `${a}%`, bottom: `${b}%` } }));
    });
    empty.hidden = points.length > 0;
    const best = points.length ? points[points.length - 1].best : null;
    const label = points.length
      ? `Best score by probe: ${fmt.pct(best, 0)} after ${points.length} scored ${points.length === 1 ? "probe" : "probes"}.`
      : "Best score by probe: no scored probes.";
    plot.setAttribute("aria-label", label);

    const scored = prefix[frame] || 0;
    setText(counter, `probe ${scored} / ${Math.max(budget ?? prefix[count - 1] ?? 0, scored)}`);
    range.max = String(Math.max(count - 1, 0));
    range.value = String(frame);
    range.setAttribute("aria-valuetext", count ? `frame ${frame + 1} of ${count}, ${order[frame].cell_id}` : "no frames");
    playBtn.setAttribute("aria-label", playing ? "Pause replay" : "Play replay");
    setText(playText, playing ? "Pause" : "Play");
    showIcon(playIcon, !playing);
    showIcon(pauseIcon, playing);
  }

  function setFrame(next) {
    frame = Math.max(0, Math.min(Math.round(next), Math.max(count - 1, 0)));
    paint();
  }

  function start() {
    if (count < 2) return;
    if (reduced()) return setFrame(count - 1);
    if (frame >= count - 1) setFrame(0);
    playing = true;
    paint();
    timer = setTimeout(tick, STEP_MS);
  }

  function tick() {
    timer = null;
    if (!playing) return;
    if (frame >= count - 1) return stop();
    setFrame(frame + 1);
    if (frame >= count - 1) stop();
    else timer = setTimeout(tick, STEP_MS);
  }

  function halt() {
    playing = false;
    if (timer) clearTimeout(timer);
    timer = null;
  }

  function stop() {
    halt();
    paint();
  }

  function reduced() {
    const flag = ctx.story?.reduced;
    const value = typeof flag === "function" ? flag() : flag;
    return value === true || prefersReducedMotion();
  }

  function stateText(state) {
    if (!cycle) return "No cycle recorded.";
    if (!(cycle.nodes || []).length) return "No tree recorded for this cycle.";
    const line = fmt.titleCase(ctx.sel.statusLine(state) || "waiting for state");
    const label = cycle.iteration != null ? `Cycle ${cycle.iteration + 1}` : cycle.cycle_id;
    return `${label}: ${line}.`;
  }

  playBtn.addEventListener("click", () => (playing ? stop() : start()));
  range.addEventListener("input", () => {
    const next = Number(range.value);
    halt();
    setFrame(next);
  });

  return { update, destroy() { halt(); clear(root); } };
}

/* --- pure helpers ------------------------------------------------------- */

function depthMap(cycle) {
  const map = new Map();
  const byId = new Map((cycle?.nodes || []).map((node) => [node.cell_id, node]));
  const walk = (node, seen) => {
    if (!node) return 0;
    if (map.has(node.cell_id)) return map.get(node.cell_id);
    if (!node.parent_id || node.cell_id === "root" || seen.has(node.cell_id)) return map.set(node.cell_id, 0).get(node.cell_id);
    seen.add(node.cell_id);
    const parent = byId.get(node.parent_id);
    return map.set(node.cell_id, parent ? walk(parent, seen) + 1 : 0).get(node.cell_id);
  };
  (cycle?.nodes || []).forEach((node) => walk(node, new Set()));
  return map;
}

function isScored(node) {
  return Boolean(node) && node.cell_id !== "root" && node.score != null && Number.isFinite(Number(node.score));
}

function cleanSeries(series) {
  return (series || []).filter((p) => p && p.cell !== "root" && p.score != null && Number.isFinite(Number(p.score)))
    .map((p) => ({ cell: p.cell, score: Number(p.score), best: Number(p.best ?? p.score) }));
}

function signature(cycle, order) {
  if (!cycle) return "";
  const nodes = order.map((n) => [n.cell_id, n.parent_id, n.status, n.score, n.successes, n.episodes].join(":"));
  return [cycle.cycle_id, ...nodes].join("|");
}

function gateOf(state) {
  const gate = Number(state.server.data?.protocol?.smoke_gate), trials = Number(state.server.data?.protocol?.smoke_trials);
  return Number.isFinite(gate) && Number.isFinite(trials) && trials > 0 ? { value: gate / trials, label: `${gate}/${trials}` } : null;
}

function toneOf(node) {
  if (node.status === "agent" || node.status === "evaluating") return "live";
  if (node.status === "stopped") return "stopped";
  if (node.status === "pending") return "queued";
  return node.cell_id === "root" ? "root" : node.fail_class && node.fail_class !== "ok" ? "failed" : "ok";
}

function scoreLabel(node, scored) {
  if (node.cell_id === "root") return "baseline";
  return scored ? `${(clamp01(node.score) * 100).toFixed(0)} %` : SHORT[node.status] || "—";
}

function smokeLabel(node) {
  if (node.cell_id === "root" || node.successes == null || node.episodes == null) return "";
  return `${node.successes} / ${node.episodes}`;
}

function clamp01(value) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : 0;
}

function pauseGlyph() {
  const rect = (x) => svg("rect", { x, y: "3", width: "3", height: "10", rx: "1", fill: "currentColor" });
  return svg("svg", { class: "icon", width: 12, height: 12, viewBox: "0 0 16 16", "aria-hidden": "true", hidden: true }, [rect("4"), rect("9")]);
}

function showIcon(node, show) {
  if (show) node.removeAttribute("hidden");
  else node.setAttribute("hidden", "");
}
