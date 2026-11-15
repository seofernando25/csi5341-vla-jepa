/* Figures for the Dreaming chapter: the recorded world tree, the candidates'
 * replay paths, the probes/attainment curves and the V bars. Geometry and DOM
 * only; every number is read through model.js. */

import { el, svg, clear, setText, icon } from "../dom.js";
import { dreamFrame } from "./model.js";

export const PALETTE = ["var(--accent)", "var(--ok)", "var(--warn)", "var(--err)"];
const SHAPES = ["circle", "square", "diamond", "triangle"];

export function swatch(index, size = 12) {
  const body = shape(index, size / 2, size / 2, size * 0.36);
  return svg("svg", { class: "dream-swatch", width: size, height: size, viewBox: `0 0 ${size} ${size}`, "aria-hidden": "true" }, body);
}

export function shape(index, x, y, r, hollow = false) {
  const style = hollow
    ? { fill: "none", stroke: PALETTE[index % PALETTE.length], strokeWidth: 1.6 }
    : { fill: PALETTE[index % PALETTE.length] };
  const kind = SHAPES[index % SHAPES.length];
  if (kind === "circle") return svg("circle", { cx: x, cy: y, r, style });
  if (kind === "square") return svg("rect", { x: x - r * 0.85, y: y - r * 0.85, width: r * 1.7, height: r * 1.7, rx: 1, style });
  if (kind === "diamond") {
    return svg("rect", { x: x - r * 0.7, y: y - r * 0.7, width: r * 1.4, height: r * 1.4, rx: 1, style, transform: `rotate(45 ${x} ${y})` });
  }
  return svg("path", { d: `M${x} ${y - r * 1.15}L${x + r * 1.05} ${y + r * 0.85}L${x - r * 1.05} ${y + r * 0.85}Z`, style });
}

function marker(index, x, y, n, selected) {
  return svg("g", { class: "dream-marker" }, [
    shape(index, x, y, selected ? 7 : 5.5),
    svg("text", { x, y: y + 2.6, "text-anchor": "middle", class: "dream-marker-num" }, String(n)),
  ]);
}

export function fixed(value, digits = 3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

export function betaLabel(value) {
  const n = Number(value);
  return Number.isFinite(n) ? String(n) : "—";
}

export function verdictWord(status) {
  return status === "not_better" ? "not better" : status || "—";
}

/* Depth from the parent chain; a node whose parent is the implicit root is 0. */
export function treeLayout(tree) {
  const nodes = (tree?.nodes || []).filter((node) => node && node.cell_id);
  const byId = new Map(nodes.map((node) => [node.cell_id, node]));
  const depthOf = (node) => {
    let depth = 0;
    let parent = node.parent_id;
    const seen = new Set([node.cell_id]);
    while (parent && byId.has(parent) && !seen.has(parent) && depth < 20) {
      seen.add(parent);
      depth += 1;
      parent = byId.get(parent).parent_id;
    }
    return depth;
  };
  const items = nodes.map((node) => ({
    node,
    depth: depthOf(node),
    branch: Number.isFinite(Number(node.branch)) ? Number(node.branch) : 0,
  }));
  const branches = [...new Set(items.map((item) => item.branch))].sort((a, b) => a - b);
  const rows = items.reduce((top, item) => Math.max(top, item.depth + 1), 1);
  return { items, branches, rows };
}

/* --- the world ---------------------------------------------------------- */

export function worldFigure(world, tree) {
  const layout = treeLayout(tree);
  const gain = Number(world?.max_gain);
  const maxGain = Number.isFinite(gain) && gain > 0 ? gain : null;
  const columns = layout.branches.map((branch) => {
    const rows = layout.items
      .filter((item) => item.branch === branch)
      .sort((a, b) => a.depth - b.depth || (a.node.attempt ?? 0) - (b.node.attempt ?? 0));
    return el("div", { class: "dream-branch" }, [
      el("div", { class: "dream-branch-label" }, `b${branch}`),
      ...rows.map((item) => nodeRow(item.node, item.depth, maxGain)),
    ]);
  });
  const meta = [
    el("span", {}, ["baseline ", el("b", {}, fixed(world?.baseline_score))]),
    el("span", {}, ["max gain ", el("b", {}, maxGain == null ? "—" : fixed(maxGain))]),
    el("span", {}, `${layout.items.length} nodes`),
    world?.degenerate ? el("span", { class: "dream-degenerate" }, "degenerate") : null,
  ];
  return el("figure", { class: "figure dream-world" }, [
    el("div", { class: "figure-head" }, [
      el("span", {}, "The world"),
      el("span", { class: "dream-head-meta" }, `${world?.id ?? ""} · ${layout.branches.length} branches`),
    ]),
    el("div", { class: "dream-world-meta" }, meta),
    layout.items.length
      ? el("div", { class: "dream-branches", style: { "--cols": String(Math.max(1, layout.branches.length)) } }, columns)
      : el("p", { class: "dream-note" }, "No tree recorded for this world."),
    el("p", { class: "figure-cap" }, "The recorded tree: every child the cycle opened, scored. Attainment is measured against the max gain."),
  ]);
}

function nodeRow(node, depth, maxGain) {
  const score = Number(node.score);
  const width = maxGain && Number.isFinite(score) ? Math.max(0, Math.min(1, score / maxGain)) * 100 : 0;
  const fail = node.fail_class && node.fail_class !== "ok" ? node.fail_class : null;
  return el("div", { class: "dream-node", style: { "--d": String(depth) }, dataset: { fail: fail ? "true" : "false" } }, [
    el("span", { class: "dream-node-id" }, node.cell_id),
    el("span", { class: "dream-node-bar" }, el("i", { style: { width: `${width}%` } })),
    el("span", { class: "dream-node-score" }, fixed(node.score)),
    fail ? el("span", { class: "dream-node-fail" }, fail) : null,
  ]);
}

/* --- the paths ---------------------------------------------------------- */

export function pathsFigure({ payload, model, worldId, beta, selected, reduced }) {
  const tree = model.tree(worldId);
  const layout = treeLayout(tree);
  const candidates = model.candidates || [];
  const figure = el("figure", { class: "figure dream-paths" }, [
    el("div", { class: "figure-head" }, [
      el("span", {}, "The paths"),
      el("span", { class: "dream-head-meta" }, `${worldId} · beta ${betaLabel(beta)}`),
    ]),
  ]);
  if (!tree || !layout.items.length) {
    figure.append(el("p", { class: "dream-note" }, "No tree recorded for this world."));
    return { node: figure, setFrame() {}, destroy() {} };
  }

  const width = 760;
  const top = 40;
  const rowH = 44;
  const height = top + Math.max(0, layout.rows - 1) * rowH + 34;
  const colW = (width - 92) / Math.max(1, layout.branches.length);
  const pos = new Map(
    layout.items.map((item) => [
      item.node.cell_id,
      { x: 46 + colW * (layout.branches.indexOf(item.branch) + 0.5), y: top + item.depth * rowH },
    ]),
  );
  const root = { x: width / 2, y: 20 };
  const statics = layout.items.flatMap((item) => {
    const to = pos.get(item.node.cell_id);
    const from = pos.get(item.node.parent_id) || root;
    const mid = (from.y + to.y) / 2;
    return [
      svg("path", { class: "dream-edge", d: `M${from.x} ${from.y}C${from.x} ${mid} ${to.x} ${mid} ${to.x} ${to.y}` }),
      svg("circle", { class: "dream-node-dot", cx: to.x, cy: to.y, r: 3 }),
      svg("text", { class: "dream-node-label", x: to.x, y: to.y + 12, "text-anchor": "middle" }, item.node.cell_id),
      svg("text", { class: "dream-node-value", x: to.x, y: to.y + 21, "text-anchor": "middle" }, fixed(item.node.score)),
    ];
  });
  statics.push(
    svg("circle", { class: "dream-node-dot", cx: root.x, cy: root.y, r: 3 }),
    svg("text", { class: "dream-node-label", x: root.x, y: root.y - 8, "text-anchor": "middle" }, "root"),
  );
  const pathsGroup = svg("g");
  figure.append(
    el("div", { class: "dream-plot" }, svg("svg", { class: "dream-paths-svg", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Replay paths over the recorded tree" }, [
      ...statics,
      pathsGroup,
    ])),
  );

  const liveNodes = new Map();
  figure.append(
    el("div", { class: "dream-legend" }, candidates.map((candidate, index) => {
      const live = el("span", { class: "dream-legend-live" });
      liveNodes.set(candidate.slot, live);
      return el("div", { class: "dream-legend-row", dataset: { selected: String(candidate.slot === selected) } }, [
        swatch(index, 13),
        el("span", { class: "dream-legend-main" }, [el("span", { class: "dream-legend-slot" }, candidate.slot || "—"), live]),
        el("span", { class: "dream-legend-status" }, verdictWord(candidate.status)),
        el("span", { class: "dream-legend-sep", "aria-hidden": "true" }, "·"),
        el("span", { class: "dream-legend-v" }, `V ${fixed(candidate.mean_v)}`),
      ]);
    })),
  );

  const last = dreamFrame(payload, worldId, beta, null).last;
  let timer = null;
  const counter = el("span", { class: "dream-counter" }, "round 0 / 0");
  const range = el("input", {
    class: "dream-range",
    type: "range",
    min: 1,
    max: Math.max(1, last),
    step: 1,
    value: Math.max(1, last),
    aria: { label: "Replay round" },
  });
  const playLabel = el("span", {}, "Play");
  const play = el("button", { class: "dream-play", type: "button", aria: { label: "Play the replay" } }, [icon("play", 11), playLabel]);
  if (last > 0) figure.append(el("div", { class: "dream-scrub" }, [play, range, counter]));
  else figure.append(el("p", { class: "dream-note" }, "No replay rounds recorded for this beta."));
  figure.append(el("p", { class: "figure-cap" }, "Numbers are probe order; each candidate's markers sit on the cells it revealed."));

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    play.dataset.on = "false";
    setText(playLabel, "Play");
  }

  function setFrame(round) {
    const frame = dreamFrame(payload, worldId, beta, round);
    clear(pathsGroup);
    const order = candidates
      .map((candidate, index) => ({ candidate, index, path: frame.paths.find((item) => item.slot === candidate.slot) }))
      .sort((a, b) => Number(a.candidate.slot === selected) - Number(b.candidate.slot === selected));
    for (const { candidate, index, path } of order) {
      if (!path) continue;
      const cells = path.cells.map((cell) => (typeof cell === "string" ? cell : cell?.cell_id)).filter((cell) => pos.has(cell));
      if (!cells.length) continue;
      const offset = (index - (candidates.length - 1) / 2) * 16;
      const on = candidate.slot === selected;
      cells.forEach((cell, step) => {
        const point = pos.get(cell);
        pathsGroup.append(marker(index, point.x + offset, point.y, step + 1, on));
      });
    }
    for (const path of frame.paths) {
      const live = liveNodes.get(path.slot);
      if (!live) continue;
      const attainment = path.attainment == null ? "attainment —" : `${Number(path.attainment).toFixed(2)} attainment`;
      setText(live, `${path.probes} probes · ${attainment}`);
    }
    setText(counter, `round ${last > 0 ? frame.rounds : 0} / ${Math.max(last, 0)}`);
    if (last > 0) range.value = String(frame.rounds);
  }

  function tick() {
    const next = Number(range.value) + 1;
    if (next > last) return stop();
    range.value = String(next);
    setFrame(next);
  }

  play.addEventListener("click", () => {
    if (timer) return stop();
    if (last < 1) return;
    if (reduced) {
      range.value = String(last);
      setFrame(last);
      return;
    }
    if (Number(range.value) >= last) {
      range.value = "1";
      setFrame(1);
    }
    play.dataset.on = "true";
    setText(playLabel, "Pause");
    timer = setInterval(tick, 900);
  });
  range.addEventListener("input", () => {
    stop();
    setFrame(Number(range.value));
  });

  setFrame(last);
  return { node: figure, setFrame, destroy: stop };
}
