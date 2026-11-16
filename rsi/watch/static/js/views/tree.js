/* Explorer: one collapsible section per cycle, a real tree with lineage guides. */

import { el, clear, icon, setText } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";

const ROW_MOVE = { ArrowDown: 1, j: 1, ArrowUp: -1, k: -1, Home: "first", End: "last" };

export function mount(root, ctx) {
  const { dispatch, act, store } = ctx;
  const count = el("span", { class: "count" });
  const head = el("div", { class: "section-head" }, [el("span", {}, "Explorer"), count]);
  const body = el("div", { class: "sidebar-body tree-scroll" });
  root.append(head, body);

  let signature = "";

  const rowEls = () => Array.from(body.querySelectorAll(".tree-row"));
  const findRow = (source) => rowEls().find((row) => row.dataset.source === source) || null;
  const findHead = (cycleId) =>
    Array.from(body.querySelectorAll(".tree-cycle-head")).find((node) => node.dataset.cycle === cycleId) || null;

  function focusRow(source, scroll = true) {
    const row = findRow(source);
    if (!row) return;
    row.focus({ preventScroll: true });
    if (scroll) row.scrollIntoView({ block: "nearest" });
  }

  function moveTo(source) {
    if (!source) return;
    if (store.getState().ui.treeCursor !== source) dispatch(act.treeCursor(source));
    focusRow(source);
  }

  function step(row, where) {
    const list = rowEls();
    const index = list.indexOf(row);
    if (index < 0) return;
    const next = where === "first" ? list[0] : where === "last" ? list[list.length - 1] : list[index + where];
    if (next) moveTo(next.dataset.source);
  }

  function enterCycle(header, last) {
    const section = header.closest(".tree-cycle");
    if (!section) return;
    if (!last) {
      const first = section.querySelector(".tree-row");
      if (first) moveTo(first.dataset.source);
      return;
    }
    const previous = section.previousElementSibling;
    const rows = previous ? previous.querySelectorAll(".tree-row") : [];
    if (rows.length) moveTo(rows[rows.length - 1].dataset.source);
  }

  function onHeaderKey(event, header) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const collapsed = header.getAttribute("aria-expanded") === "false";
    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      enterCycle(header, event.key === "ArrowUp");
      return;
    }
    if ((event.key === "ArrowLeft" && !collapsed) || (event.key === "ArrowRight" && collapsed)) {
      event.preventDefault();
      dispatch(act.treeToggle(header.dataset.cycle));
      findHead(header.dataset.cycle)?.focus({ preventScroll: true });
    }
  }

  function onRowKey(event, row) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const move = ROW_MOVE[event.key];
    if (move != null) {
      event.preventDefault();
      step(row, move);
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      dispatch(act.open(row.dataset.source));
      focusRow(row.dataset.source, false);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      if (row.dataset.children === "1") step(row, 1);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      const section = row.closest(".tree-cycle");
      if (!section) return;
      if (row.dataset.parent) moveTo(sel.sourceOf("node", section.dataset.cycle, row.dataset.parent));
      else findHead(section.dataset.cycle)?.focus({ preventScroll: true });
    }
  }

  function onKeyDown(event) {
    const header = event.target.closest?.(".tree-cycle-head");
    if (header) return onHeaderKey(event, header);
    const row = event.target.closest?.(".tree-row");
    if (row) return onRowKey(event, row);
  }

  function onClick(event) {
    const row = event.target.closest?.(".tree-row");
    if (!row) return;
    dispatch(act.open(row.dataset.source));
    focusRow(row.dataset.source, false);
  }

  function cycleHeader(state, cycle, collapsed) {
    const scored = sel.scoredCount(cycle);
    const total = (cycle.nodes || []).filter((node) => node.cell_id !== "root").length;
    const best = sel.bestIn(state, cycle);
    const active = sel.parseSource(state.ui.activeId)?.cycle === cycle.cycle_id;
    const status = cycleStatus(state, cycle);
    const recorded = cycle.status && cycle.status !== status ? ` · recorded ${cycle.status}` : "";
    const title = `${scored} of ${total} nodes scored${best ? ` · best ${fmt.pct(best.score)}` : ""}${recorded}`;
    const label =
      `Cycle ${cycleLabel(cycle)}, ${status}, ${scored} of ${total} scored` +
      `${best ? `, best ${fmt.pct(best.score)}` : ""}${recorded}`;
    const dataset = {
      cycle: cycle.cycle_id,
      current: state.ui.cycleId === cycle.cycle_id ? "1" : "0",
      active: active ? "1" : "0",
    };
    const button = el(
      "button",
      {
        class: "tree-cycle-head",
        type: "button",
        "aria-expanded": collapsed ? "false" : "true",
        "aria-label": label,
        dataset,
        title,
      },
      [
        icon(collapsed ? "chevronRight" : "chevronDown", 11),
        el("span", { class: "tree-cycle-name truncate" }, `Cycle ${cycleLabel(cycle)}`),
        el("span", { class: "pill", dataset: { tone: cycleTone(state, cycle) } }, status),
        el("span", { class: "tree-cycle-count num" }, `${scored}/${total}`),
        best ? el("span", { class: "tree-cycle-best num", title: "best score" }, fmt.pct(best.score, 0)) : null,
      ],
    );
    button.addEventListener("click", () => dispatch(act.treeToggle(cycle.cycle_id)));
    return button;
  }

  function cycleSection(state, cycle, tabStop) {
    const collapsed = state.ui.collapsedCycles.includes(cycle.cycle_id);
    const section = el("section", {
      class: "tree-cycle",
      dataset: { cycle: cycle.cycle_id, collapsed: collapsed ? "1" : "0" },
    });
    section.append(cycleHeader(state, cycle, collapsed));
    if (collapsed) return section;
    const tree = el("div", { class: "tree-rows", role: "tree", "aria-label": `Cycle ${cycleLabel(cycle)} nodes` });
    for (const row of sel.treeRows(state, cycle)) tree.append(nodeRow(state, cycle, row, tabStop));
    section.append(tree);
    return section;
  }

  function render(state) {
    const next = signatureOf(state);
    if (next === signature) return;
    signature = next;

    const focused = document.activeElement;
    const focusKey =
      focused && body.contains(focused)
        ? focused.dataset.source
          ? `row:${focused.dataset.source}`
          : focused.dataset.cycle
            ? `head:${focused.dataset.cycle}`
            : null
        : null;

    const cycles = state.server.data?.cycles || [];
    const scrollTop = body.scrollTop;
    clear(body);
    if (!cycles.length) {
      setText(count, "");
      body.append(emptyState(state));
      return;
    }
    const tabStop = pickCursor(state, cycles);
    let nodes = 0;
    for (const cycle of [...cycles].reverse()) {
      nodes += (cycle.nodes || []).length;
      body.append(cycleSection(state, cycle, tabStop));
    }
    setText(count, fmt.plural(nodes, "node"));
    body.scrollTop = scrollTop;
    if (!focusKey) return;
    if (focusKey.startsWith("row:")) {
      const source = focusKey.slice(4);
      if (findRow(source)) focusRow(source, false);
      else findHead(sel.parseSource(source)?.cycle)?.focus({ preventScroll: true });
    } else {
      findHead(focusKey.slice(5))?.focus({ preventScroll: true });
    }
  }

  body.addEventListener("keydown", onKeyDown);
  body.addEventListener("click", onClick);
  render(store.getState());

  return {
    update: render,
    destroy() {
      body.removeEventListener("keydown", onKeyDown);
      body.removeEventListener("click", onClick);
    },
  };
}

/* --- pure renderers ------------------------------------------------------ */

const cycleLabel = (cycle) => (cycle.iteration != null ? cycle.iteration : cycle.cycle_id);

/* A cycle record that says "running" while the loop is dead is a stopped cycle. */
function cycleStatus(state, cycle) {
  if (cycle.status === "running" && !sel.loopLive(state)) return "stopped";
  return cycle.status || "unknown";
}

function cycleTone(state, cycle) {
  const status = cycleStatus(state, cycle);
  if (status === "running") return "live";
  if (status === "done") return "ok";
  if (status === "error") return "err";
  if (status === "stopped") return "warn";
  return "idle";
}

function statusTone(phase, node) {
  if (phase === "agent" || phase === "evaluating") return "live";
  if (phase === "done") {
    if (node.fail_class && node.fail_class !== "ok") return "err";
    return node.score != null || node.cell_id === "root" ? "ok" : "idle";
  }
  return phase === "stopped" ? "warn" : "idle";
}

function indent(depth, guides, isLast) {
  const wrap = el("span", { class: "tree-indent", "aria-hidden": "true" });
  for (let level = 0; level < depth; level += 1) {
    wrap.append(
      el("i", {
        class: level === depth - 1 ? "tree-elbow" : "tree-guide",
        dataset: { on: guides[level] ? "1" : "0", last: isLast ? "1" : "0" },
      }),
    );
  }
  return wrap;
}

function shortDur(seconds) {
  const total = Math.max(0, Math.round(Number(seconds)));
  if (total < 60) return `${total}s`;
  if (total < 3600) return `${Math.floor(total / 60)}m`;
  if (total < 86_400) return `${(total / 3600).toFixed(total < 36_000 ? 1 : 0)}h`;
  return `${Math.floor(total / 86_400)}d`;
}

function rowTitle(node, phase) {
  const parts = [node.cell_id === "root" ? "baseline" : node.cell_id, fmt.nodeStatusLabel(phase)];
  if (node.fail_class) parts.push(fmt.failLabel(node.fail_class, true));
  if (node.score != null) parts.push(fmt.pct(node.score));
  if (node.successes != null && node.episodes != null) parts.push(`smoke ${node.successes}/${node.episodes}`);
  if (node.promoted) parts.push("promoted");
  if (node.error) parts.push(fmt.firstLine(node.error, 80));
  return parts.join(" · ");
}

function metaBlock(node, phase, score, tone) {
  const line = el("span", { class: "tree-meta-line" });
  if (score != null) line.append(el("b", { class: "tree-score num" }, fmt.pct(score, 0)));
  if (node.successes != null && node.episodes != null) {
    line.append(el("span", { class: "m-smoke num", title: "smoke trials" }, `${node.successes}/${node.episodes}`));
  }
  const seconds = node.elapsed_s ?? node.agent_seconds;
  if (seconds != null) line.append(el("span", { class: "m-dur num", title: "duration" }, shortDur(seconds)));
  if (node.agent_cost_usd > 0) {
    line.append(el("span", { class: "m-cost num", title: "agent cost" }, fmt.usd(node.agent_cost_usd)));
  }
  if (!line.childNodes.length) {
    line.append(el("span", { class: "m-state" }, phase === "pending" ? "queued" : fmt.nodeStatusLabel(phase)));
  }
  const width = score == null ? 0 : Math.max(0, Math.min(1, score)) * 100;
  return el("span", { class: "tree-meta" }, [
    line,
    el("span", { class: "tree-bar", "aria-hidden": "true", dataset: { tone } }, [
      el("i", { style: { width: `${width}%` } }),
    ]),
  ]);
}

function nodeRow(state, cycle, row, tabStop) {
  const { node, depth, guides, isLast, hasChildren, phase } = row;
  const source = sel.sourceOf("node", cycle.cycle_id, node.cell_id);
  const active = state.ui.activeId === source;
  const tone = statusTone(phase, node);
  const score = typeof node.score === "number" ? node.score : null;
  const dataset = {
    source,
    phase,
    tone,
    cell: node.cell_id,
    depth,
    parent: node.parent_id || "",
    children: hasChildren ? "1" : "0",
    active: active ? "1" : "0",
    cursor: state.ui.treeCursor === source ? "1" : "0",
  };
  const label = rowTitle(node, phase);
  const props = {
    class: "tree-row",
    role: "treeitem",
    tabIndex: source === tabStop ? 0 : -1,
    "aria-level": depth + 1,
    "aria-selected": active ? "true" : "false",
    "aria-label": label,
    dataset,
  };
  return el("div", props, [
    indent(depth, guides, isLast),
    el("span", { class: "tree-caret", "aria-hidden": "true" }, hasChildren ? icon("chevronDown", 10) : null),
    el("span", { class: "dot", dataset: { tone } }),
    el("span", { class: "tree-name truncate" }, node.cell_id === "root" ? "baseline" : node.cell_id),
    metaBlock(node, phase, score, tone),
  ]);
}

function emptyState(state) {
  const status = state.server.status;
  if (status === "loading" || status === "idle") {
    return el("div", { class: "tree-loading", role: "status" }, [
      el("span", { class: "spinner", "aria-hidden": "true" }),
      el("span", {}, status === "loading" ? "Loading state" : "Waiting for state"),
    ]);
  }
  if (status === "error") {
    return el("div", { class: "empty" }, [
      el("strong", {}, "No state"),
      state.server.error || "The watch server did not answer.",
    ]);
  }
  return el("div", { class: "empty" }, [el("strong", {}, "No cycles yet"), "The loop has not written a cycle."]);
}

function firstSource(state, cycles) {
  for (const cycle of [...cycles].reverse()) {
    if (state.ui.collapsedCycles.includes(cycle.cycle_id)) continue;
    const rows = sel.treeRows(state, cycle);
    if (rows.length) return sel.sourceOf("node", cycle.cycle_id, rows[0].node.cell_id);
  }
  return null;
}

function pickCursor(state, cycles) {
  for (const wanted of [state.ui.treeCursor, state.ui.activeId]) {
    const parsed = wanted ? sel.parseSource(wanted) : null;
    const cycle = parsed?.kind === "node" ? cycles.find((item) => item.cycle_id === parsed.cycle) : null;
    const visible = cycle && !state.ui.collapsedCycles.includes(cycle.cycle_id);
    if (visible && (cycle.nodes || []).some((node) => node.cell_id === parsed.target)) return wanted;
  }
  return firstSource(state, cycles);
}

const nodeSignature = (node) => [
  node.cell_id,
  node.parent_id,
  node.branch,
  node.attempt,
  node.status,
  node.fail_class,
  node.score,
  node.successes,
  node.episodes,
  node.elapsed_s,
  node.agent_seconds,
  node.agent_cost_usd,
  node.promoted,
  Boolean(node.error),
];

function signatureOf(state) {
  const ui = state.ui;
  return JSON.stringify([
    state.server.status,
    state.server.error,
    sel.loopLive(state),
    ui.cycleId,
    ui.activeId,
    ui.treeCursor,
    ui.collapsedCycles,
    (state.server.data?.cycles || []).map((cycle) => [
      cycle.cycle_id,
      cycle.iteration,
      cycle.status,
      (cycle.nodes || []).map(nodeSignature),
    ]),
  ]);
}
