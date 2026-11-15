/* Pure selectors over the store. No DOM, no side effects, importable under node. */

import { cleanText, firstLine, firstSentence } from "./format.js";

const SOURCE_RE = /^(node|policy):([^:]+):(.+)$/;

export function serverData(state) {
  return state.server.data;
}

export function loop(state) {
  return state.server.data?.loop || null;
}

export function isLoopLive(value) {
  return Boolean(value && value.running === true && value.status === "running");
}

export function loopLive(state) {
  return isLoopLive(loop(state));
}

export function loopEnded(state) {
  const status = loop(state)?.status;
  return status === "stopped" || status === "done" || status === "error";
}

export function loopStatusLabel(state) {
  const value = loop(state);
  if (!value) return "no state";
  if (value.status === "running") return value.stage === "policy_dev" ? "policy dev" : "running";
  if (value.status === "idle") return "not started";
  return value.status;
}

/* A node is live only while the loop is live; a dead loop demotes an in-flight node. */
export function nodePhase(state, node) {
  if (!node) return "pending";
  if (node.status === "done" || node.status === "pending" || node.status === "stopped") {
    return node.status;
  }
  return loopLive(state) ? node.status : "stopped";
}

export function isNodeLive(state, node) {
  const phase = nodePhase(state, node);
  return phase === "agent" || phase === "evaluating";
}

export function parseSource(source) {
  const match = SOURCE_RE.exec(String(source || ""));
  if (!match) return null;
  return { kind: match[1], cycle: match[2], target: match[3] };
}

export function sourceOf(kind, cycle, target) {
  return `${kind}:${cycle}:${target}`;
}

export function cycles(state) {
  return state.server.data?.cycles || [];
}

export function cycleById(state, id) {
  return cycles(state).find((c) => c.cycle_id === id) || null;
}

export function currentCycle(state) {
  return cycleById(state, state.ui.cycleId) || cycles(state)[cycles(state).length - 1] || null;
}

export function nodeById(state, cycleId, cellId) {
  const cycle = cycleById(state, cycleId);
  return cycle?.nodes.find((n) => n.cell_id === cellId) || null;
}

export function nodeForSource(state, source) {
  const parsed = parseSource(source);
  if (!parsed || parsed.kind !== "node") return null;
  return nodeById(state, parsed.cycle, parsed.target);
}

export function stepForSource(state, source) {
  const parsed = parseSource(source);
  if (!parsed || parsed.kind !== "policy") return null;
  const cycle = cycleById(state, parsed.cycle);
  return cycle?.policy_dev.steps.find((s) => String(s.step) === parsed.target) || null;
}

export function isRoot(node) {
  return node?.cell_id === "root";
}

export function bestIn(state, cycle) {
  let best = null;
  for (const node of cycle?.nodes || []) {
    if (node.cell_id === "root") continue;
    if (node.fail_class !== "ok" || node.score == null) continue;
    if (!best || node.score > best.score) best = node;
  }
  return best;
}

export function bestOf(state) {
  let out = null;
  for (const cycle of cycles(state)) {
    const node = bestIn(state, cycle);
    if (node && (!out || node.score > out.node.score)) out = { node, cycle };
  }
  return out;
}

export function scoredCount(cycle) {
  return (cycle?.nodes || []).filter((n) => n.cell_id !== "root" && n.status === "done").length;
}

export function probeBudget(state) {
  const value = loop(state)?.probe_budget;
  return typeof value === "number" ? value : null;
}

/* DFS over parent_id, one row per node, with indent guides and sibling order. */
export function treeRows(state, cycle) {
  if (!cycle) return [];
  const nodes = cycle.nodes || [];
  const children = new Map();
  for (const node of nodes) {
    const key = node.parent_id || "__orphan__";
    if (!children.has(key)) children.set(key, []);
    children.get(key).push(node);
  }
  for (const list of children.values()) {
    list.sort((a, b) => (a.branch ?? 99) - (b.branch ?? 99) || (a.attempt ?? 99) - (b.attempt ?? 99));
  }
  const rows = [];
  const seen = new Set();
  const walk = (node, depth, guides, isLast) => {
    if (seen.has(node.cell_id)) return;
    seen.add(node.cell_id);
    const kids = children.get(node.cell_id) || [];
    rows.push({ node, depth, guides, isLast, hasChildren: kids.length > 0, phase: nodePhase(state, node) });
    kids.forEach((kid, index) => {
      walk(kid, depth + 1, [...guides, !isLast && depth > 0], index === kids.length - 1);
    });
  };
  const roots = nodes.filter((n) => n.cell_id === "root");
  for (const root of roots) walk(root, 0, [], true);
  const orphans = (children.get("__orphan__") || []).filter((n) => !seen.has(n.cell_id));
  orphans.forEach((node, index) => walk(node, 0, [], index === orphans.length - 1));
  return rows;
}

export function policyRows(state, cycle) {
  const dev = cycle?.policy_dev;
  if (!dev) return [];
  return (dev.steps || []).map((step) => ({
    step,
    selected: dev.selected_step === step.step,
    source: sourceOf("policy", cycle.cycle_id, String(step.step)),
  }));
}

export function lineage(state, cycleId, cellId) {
  const cycle = cycleById(state, cycleId);
  const out = [];
  let node = cycle?.nodes.find((n) => n.cell_id === cellId) || null;
  const seen = new Set();
  while (node && !seen.has(node.cell_id)) {
    seen.add(node.cell_id);
    out.unshift(node);
    node = node.parent_id ? cycle.nodes.find((n) => n.cell_id === node.parent_id) : null;
  }
  return out;
}

export function activeTab(state) {
  return state.ui.tabs.find((t) => t.id === state.ui.activeId) || null;
}

export function activeSource(state) {
  return activeTab(state)?.id || null;
}

export function activeView(state) {
  return activeTab(state)?.view || "transcript";
}

export function tabTitle(state, tab) {
  if (!tab) return "";
  if (tab.kind === "policy") return `Policy step ${tab.target}`;
  if (tab.target === "root") return "baseline";
  return tab.target;
}

export function tabSubtitle(state, tab) {
  if (!tab) return "";
  const cycle = cycleById(state, tab.cycle);
  const label = cycle ? `cycle ${cycle.iteration ?? cycle.cycle_id}` : tab.cycle;
  if (tab.kind === "policy") return `${label} · policy development`;
  const node = nodeById(state, tab.cycle, tab.target);
  if (!node) return label;
  if (node.cell_id === "root") return `${label} · root`;
  return `${label} · branch ${node.branch ?? "—"} attempt ${node.attempt ?? "—"}`;
}

/* ---------------------------------------------------------------------------
 * Transcript grouping: one block per reasoning run, message, tool call or
 * result. Deltas are merged; a tool_result attaches to its tool_call.
 * ------------------------------------------------------------------------- */
const BLOCK_CACHE = new WeakMap();

export function transcriptBlocks(state) {
  const events = state.transcript.events;
  const cached = BLOCK_CACHE.get(events);
  if (cached) return cached;
  const blocks = groupEvents(events);
  BLOCK_CACHE.set(events, blocks);
  return blocks;
}

export function groupEvents(events) {
  const blocks = [];
  const open = new Map();
  let last = null;
  const push = (block) => {
    block.index = blocks.length;
    blocks.push(block);
    last = block;
    return block;
  };
  for (const event of events || []) {
    const kind = event.kind;
    if (kind === "text" || kind === "thinking") {
      const text = typeof event.text === "string" ? cleanText(event.text) : "";
      if (!text) continue;
      if (last && last.kind === kind && last.open) {
        last.text += text;
        last.i1 = event.i;
        last.chars = last.text.length;
        last.truncated = last.truncated || event.truncated === true;
      } else {
        push({
          kind,
          text,
          chars: text.length,
          i0: event.i,
          i1: event.i,
          open: true,
          truncated: event.truncated === true,
        });
      }
      continue;
    }
    if (last) last.open = false;
    if (kind === "tool_call") {
      const input = event.input || {};
      const block = push({
        kind: "tool",
        tool: event.tool || "tool",
        summary: input.summary || event.tool || "tool",
        input,
        output: null,
        isError: false,
        truncated: false,
        done: false,
        callId: event.tool_use_id || null,
        i0: event.i,
        i1: event.i,
      });
      if (block.callId) open.set(block.callId, block);
      continue;
    }
    if (kind === "tool_result") {
      const target = event.tool_use_id ? open.get(event.tool_use_id) : null;
      if (target) {
        target.output = cleanText(event.output);
        target.isError = event.is_error === true;
        target.truncated = event.truncated === true;
        target.done = true;
        target.i1 = event.i;
        if (!target.tool && event.tool) target.tool = event.tool;
        open.delete(event.tool_use_id);
      } else {
        push({
          kind: "toolresult",
          tool: event.tool || "tool",
          output: cleanText(event.output),
          isError: event.is_error === true,
          truncated: event.truncated === true,
          i0: event.i,
          i1: event.i,
        });
      }
      continue;
    }
    if (kind === "system") {
      push({ kind: "system", text: cleanText(event.text), i0: event.i, i1: event.i });
      continue;
    }
    if (kind === "result") {
      push({
        kind: "result",
        text: cleanText(event.text),
        meta: event.meta || {},
        isError: event.is_error === true,
        i0: event.i,
        i1: event.i,
      });
    }
  }
  for (const block of blocks) block.open = false;
  return blocks;
}

export function blockText(block) {
  if (!block) return "";
  if (block.kind === "tool") {
    return [block.tool, block.summary, block.output].filter(Boolean).join("\n");
  }
  return block.text || block.output || "";
}

export function blockTitle(block) {
  if (!block) return "";
  if (block.kind === "thinking") return firstSentence(block.text, 120) || "reasoning";
  if (block.kind === "text") return firstLine(block.text, 120) || "message";
  if (block.kind === "tool") {
    const name = block.tool || "tool";
    const summary = String(block.summary || "").trim();
    const lower = summary.toLowerCase();
    const stripped = lower.startsWith(name.toLowerCase())
      ? summary.slice(name.length).replace(/^[:\s]+/, "")
      : summary;
    return stripped ? `${name} · ${stripped}` : name;
  }
  if (block.kind === "toolresult") return `${block.tool} output`;
  if (block.kind === "result") return block.isError ? "result (error)" : "result";
  if (block.kind === "system") return block.text || "session";
  return block.kind;
}

export function filterBlocks(blocks, filter, query) {
  const q = String(query || "").trim().toLowerCase();
  return (blocks || []).filter((block) => {
    if (filter === "messages" && block.kind !== "text" && block.kind !== "result") return false;
    if (filter === "reasoning" && block.kind !== "thinking") return false;
    if (filter === "tools" && block.kind !== "tool" && block.kind !== "toolresult") return false;
    if (filter === "errors" && !block.isError) return false;
    if (q && !blockText(block).toLowerCase().includes(q)) return false;
    return true;
  });
}

export function searchHits(blocks, query) {
  const q = String(query || "").trim().toLowerCase();
  if (!q) return [];
  return (blocks || [])
    .map((block, index) => (blockText(block).toLowerCase().includes(q) ? index : -1))
    .filter((index) => index >= 0);
}

export function scoreSeries(cycle) {
  const points = [];
  for (const node of cycle?.nodes || []) {
    if (node.cell_id === "root") continue;
    if (node.fail_class !== "ok" || node.score == null) continue;
    points.push({ cell: node.cell_id, score: node.score, attempt: node.attempt, branch: node.branch });
  }
  return points;
}

export function frontier(cycle) {
  const points = scoreSeries(cycle);
  let best = 0;
  return points.map((point, index) => {
    best = Math.max(best, point.score);
    return { x: index, score: point.score, best };
  });
}

export function flattenNodes(state) {
  const out = [];
  for (const cycle of cycles(state)) {
    for (const row of treeRows(state, cycle)) {
      out.push({ cycle, node: row.node, source: sourceOf("node", cycle.cycle_id, row.node.cell_id) });
    }
  }
  return out;
}

export function nextSource(state, direction) {
  const flat = flattenNodes(state);
  if (!flat.length) return null;
  const active = activeSource(state);
  const index = flat.findIndex((item) => item.source === active);
  if (index < 0) return flat[0].source;
  const next = flat[(index + direction + flat.length) % flat.length];
  return next.source;
}

export function defaultSource(state) {
  const list = cycles(state);
  if (!list.length) return null;
  const cycle = currentCycle(state) || list[list.length - 1];
  const live = (cycle.nodes || []).find((node) => isNodeLive(state, node));
  if (live) return sourceOf("node", cycle.cycle_id, live.cell_id);
  const best = bestIn(state, cycle);
  if (best) return sourceOf("node", cycle.cycle_id, best.cell_id);
  const rows = treeRows(state, cycle);
  const stopped = [...rows].reverse().find((row) => row.phase === "stopped");
  if (stopped) return sourceOf("node", cycle.cycle_id, stopped.node.cell_id);
  const root = (cycle.nodes || []).find((node) => node.cell_id === "root");
  if (root) return sourceOf("node", cycle.cycle_id, root.cell_id);
  return null;
}

export function connectionLabel(state) {
  const t = state.transport;
  if (!t.source) return "no stream";
  if (t.status === "open") return t.frozen ? "final" : "live";
  if (t.status === "connecting") return "connecting";
  if (t.status === "gone") return "gone";
  if (t.status === "error") return "stream error";
  if (t.status === "closed") return "closed";
  return "idle";
}

export function artifactKey(source, name) {
  return `${source}|${name}`;
}

export function artifact(state, source, name) {
  return state.artifacts.byKey[artifactKey(source, name)] || null;
}

export function contextFor(state, source) {
  return state.context.bySource[source] || null;
}

export function statusLine(state) {
  const value = loop(state);
  if (!value) return "waiting for state";
  if (value.status === "running") {
    return value.stage === "policy_dev" ? "policy development" : "expanding the tree";
  }
  if (value.status === "idle") return "not started";
  if (value.stop_reason) return `${value.status}: ${value.stop_reason}`;
  return value.status;
}
