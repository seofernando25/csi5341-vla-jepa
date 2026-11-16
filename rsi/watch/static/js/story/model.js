/* Story model: pure reads over the watch state and the probe transcript.
 * No DOM, no clock, no randomness; importable under node.
 *
 * Chapters import only from this module. Dream, changed-policy and diff
 * helpers live in model_dream.js and are re-exported here, so there is one
 * import path and one place to read.
 */

import { cleanText } from "../format.js";

const MIN_SENTENCE = 20;
const MAX_SENTENCE = 6;
const MAX_OUTPUT = 400;

export { dreamModel, dreamFrame, changedModel, diffLines } from "./model_dream.js";

function num(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function list(value) {
  return Array.isArray(value) ? value : [];
}

function clip(text, max) {
  const value = String(text ?? "");
  return value.length > max ? value.slice(0, max) : value;
}

/* ---------------------------------------------------------------------------
 * Run-level facts
 * ------------------------------------------------------------------------- */

export function baseline(state) {
  const data = state?.server?.data || {};
  const loop = data.loop || {};
  const protocol = data.protocol || {};
  const agent = protocol.agent || {};
  const base = loop.baseline || {};
  return {
    bytes: num(base.resident_bytes),
    score: num(base.score),
    gate: num(protocol.smoke_gate),
    trials: num(protocol.smoke_trials),
    devGate: num(protocol.dev_gate),
    devTrials: num(protocol.dev_trials),
    budget: num(loop.probe_budget),
    cyclesTarget: num(loop.cycles_target),
    model: agent.model || null,
    backend: agent.backend || null,
  };
}

export function cycleOf(state, id) {
  const cycles = list(state?.server?.data?.cycles);
  const wanted = id ?? state?.ui?.cycleId ?? null;
  if (wanted != null) {
    const found = cycles.find((cycle) => cycle?.cycle_id === wanted);
    if (found) return found;
    if (id != null) return null; // an explicit id that is not recorded stays missing
  }
  return cycles[cycles.length - 1] || null;
}

/* ---------------------------------------------------------------------------
 * Nodes
 * ------------------------------------------------------------------------- */

function startTime(node) {
  const raw = node?.started_utc;
  if (!raw) return null;
  const ms = Date.parse(raw);
  return Number.isFinite(ms) ? ms : null;
}

/* Recorded order. The root is the baseline and always comes first. Timestamps
 * are only trusted when every node has a parseable started_utc; one missing
 * stamp (the live run leaves the first scored node unstamped) falls back to
 * the recorded grid order, (branch, attempt). Ties keep the array order. */
export function nodeOrder(cycle) {
  const items = list(cycle?.nodes).map((node, index) => ({ node, index, start: startTime(node) }));
  const timed = items.every((item) => item.start != null);
  return items
    .sort((a, b) => {
      const rootA = a.node?.cell_id === "root";
      const rootB = b.node?.cell_id === "root";
      if (rootA !== rootB) return rootA ? -1 : 1;
      if (timed && a.start !== b.start) return a.start - b.start;
      const branch = (a.node?.branch ?? 99) - (b.node?.branch ?? 99);
      if (branch) return branch;
      const attempt = (a.node?.attempt ?? 99) - (b.node?.attempt ?? 99);
      return attempt || a.index - b.index;
    })
    .map((item) => item.node);
}

/* The probe the chapter shows: the best scored done node. Proposal artifacts
 * are not in the state, so a score is the only availability signal there is.
 * With nothing scored, fall back to the latest node still being worked on. */
export function probeNode(state) {
  const cycle = cycleOf(state);
  if (!cycle) return null;
  const order = nodeOrder(cycle);
  const rank = new Map(order.map((node, index) => [node?.cell_id, index]));
  const scored = order.filter(
    (node) => node?.cell_id !== "root" && node?.status === "done" && node?.score != null,
  );
  if (scored.length) {
    const best = scored.reduce((a, b) => {
      if (b.score !== a.score) return b.score > a.score ? b : a;
      return (rank.get(b.cell_id) ?? 0) < (rank.get(a.cell_id) ?? 0) ? b : a;
    });
    return { cycle, node: best };
  }
  const live = order.filter((node) => node?.status === "agent" || node?.status === "evaluating");
  if (live.length) return { cycle, node: live[live.length - 1] };
  return null;
}

export function frontierSeries(cycle) {
  const points = [];
  let best = 0;
  for (const node of nodeOrder(cycle)) {
    if (node?.cell_id === "root" || node?.score == null) continue;
    best = Math.max(best, node.score);
    points.push({ i: points.length, cell: node.cell_id, score: node.score, best });
  }
  return points;
}

/* ---------------------------------------------------------------------------
 * Probe transcript
 * ------------------------------------------------------------------------- */

function splitSentences(text) {
  const out = [];
  for (const line of String(text || "").split("\n")) {
    let rest = line.trim();
    while (rest) {
      const cut = rest.indexOf(". ");
      if (cut < 0) {
        out.push(rest);
        break;
      }
      out.push(rest.slice(0, cut + 1));
      rest = rest.slice(cut + 2).trim();
    }
  }
  return out;
}

/* Thinking deltas joined in i order, split at `. ` and newlines. Fragments
 * under 20 characters are noise. The longest distinct ones are kept and
 * returned in reading order, so the chapter reads as prose. */
export function reasoningLines(payload, max = MAX_SENTENCE) {
  const thinking = list(payload?.events)
    .filter((event) => event?.kind === "thinking")
    .sort((a, b) => (a?.i ?? 0) - (b?.i ?? 0))
    .map((event) => cleanText(event.text))
    .join("");
  const seen = new Set();
  const items = [];
  for (const sentence of splitSentences(thinking)) {
    const text = sentence.replace(/\s+/g, " ").trim();
    if (text.length < MIN_SENTENCE || seen.has(text)) continue;
    seen.add(text);
    items.push({ text, index: items.length });
  }
  const keep = [...items]
    .sort((a, b) => b.text.length - a.text.length || a.index - b.index)
    .slice(0, Math.max(0, max));
  keep.sort((a, b) => a.index - b.index);
  return keep.map((item) => item.text);
}

export function probeTimeline(payload) {
  const events = list(payload?.events);
  const counts = {
    events: events.length,
    system: 0,
    text: 0,
    thinking: 0,
    tool_call: 0,
    tool_result: 0,
    result: 0,
  };
  for (const event of events) {
    const kind = event?.kind;
    if (typeof kind === "string") counts[kind] = (counts[kind] || 0) + 1;
  }
  const ordered = [...events].sort((a, b) => (a?.i ?? 0) - (b?.i ?? 0));
  const results = new Map();
  for (const event of ordered) {
    if (event?.kind === "tool_result" && event.tool_use_id) results.set(event.tool_use_id, event);
  }
  const tools = [];
  for (const event of ordered) {
    if (event?.kind !== "tool_call") continue;
    const input = event.input || {};
    const result = event.tool_use_id ? results.get(event.tool_use_id) || null : null;
    tools.push({
      i: event.i ?? 0,
      name: event.tool || "tool",
      summary: cleanText(input.summary) || null,
      command: cleanText(input.command) || null,
      output: result ? clip(cleanText(result.output), MAX_OUTPUT) : null,
      isError: result ? result.is_error === true : event.is_error === true,
    });
  }
  const last = [...ordered].reverse().find((event) => event?.kind === "result") || null;
  return {
    counts,
    reasoning: reasoningLines(payload, MAX_SENTENCE),
    tools,
    result: last ? { text: cleanText(last.text), meta: last.meta || {} } : null,
  };
}
