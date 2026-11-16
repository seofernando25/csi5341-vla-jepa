/* Node harness for the watch store, reducer and selectors.
   Usage: node tests/js/store_harness.mjs <rsi/watch/static>
   Prints one JSON object: { checks: { name: { ok, detail } } }. No DOM, no view modules. */

import { deepStrictEqual, ok, strictEqual } from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = process.argv[2];
if (!root) throw new Error("usage: store_harness.mjs <rsi/watch/static>");
const load = (rel) => import(pathToFileURL(path.join(root, rel)).href);

const { createStore } = await load("js/store.js");
const { makeInitialState, reducer, MAX_TABS } = await load("js/reducer.js");
const sel = await load("js/selectors.js");

const checks = {};
function check(name, fn) {
  try {
    fn();
    checks[name] = { ok: true, detail: "" };
  } catch (error) {
    checks[name] = { ok: false, detail: String(error?.stack || error) };
  }
}

const C1 = "c1";
const ev = (i, kind, extra = {}) => ({
  i,
  kind,
  text: null,
  tool: null,
  tool_use_id: null,
  input: null,
  output: null,
  truncated: false,
  is_error: false,
  meta: {},
  ...extra,
});
const run = (state, ...actions) => actions.reduce((s, action) => reducer(s, action), state);
const withData = (data, ui = {}) => {
  const state = makeInitialState();
  return { ...state, server: { ...state.server, status: "ready", data }, ui: { ...state.ui, ...ui } };
};
const loopState = (loop) => withData({ loop, cycles: [] });
const cycleWith = (nodes, extra = {}) => ({
  cycle_id: C1,
  iteration: 1,
  status: "running",
  nodes,
  policy_dev: { status: "done", selected_step: 1, steps: [] },
  ...extra,
});
const node = (cell_id, parent_id, branch, attempt, status, extra = {}) => ({
  cell_id,
  parent_id,
  branch,
  attempt,
  status,
  fail_class: null,
  score: null,
  ...extra,
});

check("loopLive", () => {
  const cases = [
    [{ running: true, status: "running" }, true],
    [{ running: false, status: "running" }, false],
    [{ running: true, status: "stopped" }, false],
    [{ running: true, status: "idle" }, false],
    [{ running: true, status: "done" }, false],
    [{ running: true, status: "error" }, false],
  ];
  for (const [loop, expected] of cases) {
    strictEqual(sel.loopLive(loopState(loop)), expected, `loopLive(${JSON.stringify(loop)})`);
  }
  strictEqual(sel.loopLive(withData(null)), false, "loopLive without data");
  strictEqual(sel.loopEnded(loopState({ status: "stopped", running: false })), true);
  strictEqual(sel.loopEnded(loopState({ status: "running", running: true })), false);
});

check("nodePhase", () => {
  const live = loopState({ running: true, status: "running" });
  const dead = ["stopped", "idle", "error", "done"].map((status) => loopState({ running: false, status }));
  for (const status of ["agent", "evaluating"]) {
    strictEqual(sel.nodePhase(live, { status }), status, `${status} while live`);
    for (const state of dead) strictEqual(sel.nodePhase(state, { status }), "stopped", `${status} while dead`);
  }
  for (const status of ["done", "pending", "stopped"]) {
    strictEqual(sel.nodePhase(live, { status }), status, `${status} while live`);
    for (const state of dead) strictEqual(sel.nodePhase(state, { status }), status, `${status} while dead`);
  }
  strictEqual(sel.nodePhase(live, null), "pending");
});

check("stateLoadedFreezes", () => {
  const open = run(
    makeInitialState(),
    { type: "transport/attach", source: "node:c1:a", after: 0 },
    { type: "transport/hello", payload: { source: "node:c1:a", cursor: 0 } },
  );
  strictEqual(open.transport.status, "open");
  strictEqual(open.transport.frozen, false);
  const stopped = reducer(open, {
    type: "state/loaded",
    data: { loop: { status: "stopped", running: false }, cycles: [] },
    at: 1,
  });
  strictEqual(stopped.transport.frozen, true, "a stopped loop freezes an open transport");
  strictEqual(stopped.transport.status, "closed");
  const running = reducer(open, {
    type: "state/loaded",
    data: { loop: { status: "running", running: true }, cycles: [] },
    at: 1,
  });
  strictEqual(running.transport.frozen, false, "a running loop keeps the transport open");
  const idle = reducer(makeInitialState(), {
    type: "state/loaded",
    data: { loop: { status: "stopped", running: false }, cycles: [] },
    at: 1,
  });
  strictEqual(idle.transport.status, "idle");
  strictEqual(idle.transport.frozen, false);
});

check("doneFreezes", () => {
  const open = run(
    makeInitialState(),
    { type: "transport/attach", source: "node:c1:a", after: 0 },
    { type: "transport/hello", payload: { source: "node:c1:a", cursor: 0 } },
    { type: "transport/events", source: "node:c1:a", events: [ev(0, "text", { text: "hi" })], next: 1, at: 1 },
  );
  const done = reducer(open, { type: "transport/done", source: "node:c1:a", reason: "score", next: 1 });
  strictEqual(done.transport.frozen, true);
  strictEqual(done.transport.done, true);
  strictEqual(done.transport.status, "closed");
  strictEqual(done.transport.reason, "score");
  const gone = reducer(open, { type: "transport/done", source: "node:c1:a", reason: "gone", next: 1 });
  strictEqual(gone.transport.status, "gone");
  const closed = reducer(open, { type: "transport/closed", source: "node:c1:a" });
  strictEqual(closed.transport.frozen, true);
  strictEqual(closed.transport.status, "closed");
});

check("helloAfterDone", () => {
  const done = run(
    makeInitialState(),
    { type: "transport/attach", source: "node:c1:a", after: 0 },
    { type: "transport/hello", payload: { source: "node:c1:a", cursor: 0 } },
    { type: "transport/events", source: "node:c1:a", events: [ev(0, "text", { text: "hi" })], next: 1, at: 1 },
    { type: "transport/done", source: "node:c1:a", reason: "score", next: 1 },
  );
  const after = reducer(done, { type: "transport/hello", payload: { source: "node:c1:a", cursor: 7 } });
  strictEqual(after.transport.frozen, true, "hello must not thaw a finished transcript");
  strictEqual(after.transport.done, true);
  deepStrictEqual(after.transcript.events, done.transcript.events, "hello must not clear events");
  strictEqual(after.transcript.events.length, 1);
  const other = reducer(done, { type: "transport/hello", payload: { source: "node:c1:b", cursor: 7 } });
  strictEqual(other, done, "hello for another source is ignored");
});

check("eventsAppend", () => {
  const attached = run(
    makeInitialState(),
    { type: "transport/attach", source: "node:c1:a", after: 0 },
    { type: "transport/hello", payload: { source: "node:c1:a", cursor: 0 } },
  );
  const foreign = reducer(attached, {
    type: "transport/events",
    source: "node:c1:b",
    events: [ev(0, "text", { text: "x" })],
    next: 1,
    at: 1,
  });
  strictEqual(foreign, attached, "events for another source are ignored");
  const first = reducer(attached, {
    type: "transport/events",
    source: "node:c1:a",
    events: [ev(0, "text", { text: "x" })],
    next: 1,
    at: 1,
  });
  strictEqual(first.transcript.source, "node:c1:a");
  strictEqual(first.transcript.events.length, 1);
  strictEqual(first.transcript.next, 1);
  strictEqual(first.transport.cursor, 1);
  const second = reducer(first, {
    type: "transport/events",
    source: "node:c1:a",
    events: [ev(1, "text", { text: "y" })],
    next: 2,
    at: 2,
  });
  deepStrictEqual(second.transcript.events.map((event) => event.i), [0, 1]);
  strictEqual(second.transport.cursor, 2);
  const frozen = reducer(run(second, { type: "transport/done", source: "node:c1:a", reason: "score", next: 2 }), {
    type: "transport/events",
    source: "node:c1:a",
    events: [ev(2, "text", { text: "z" })],
    next: 3,
    at: 3,
  });
  strictEqual(frozen.transcript.events.length, 2, "events while frozen are ignored");
  strictEqual(frozen.transport.cursor, 2);
});

check("reset", () => {
  const loaded = run(
    makeInitialState(),
    { type: "transport/attach", source: "node:c1:a", after: 0 },
    { type: "transport/hello", payload: { source: "node:c1:a", cursor: 0 } },
    { type: "transport/events", source: "node:c1:a", events: [ev(0, "text", { text: "x" })], next: 1, at: 1 },
  );
  const reset = reducer(loaded, { type: "transport/reset", source: "node:c1:a", reason: "truncated" });
  deepStrictEqual(reset.transcript.events, []);
  strictEqual(reset.transcript.next, 0);
  strictEqual(reset.transport.cursor, 0);
  strictEqual(reset.transport.reason, "truncated");
  strictEqual(reducer(loaded, { type: "transport/reset", source: "node:c1:b" }), loaded);
  const done = run(loaded, { type: "transport/done", source: "node:c1:a", reason: "score", next: 1 });
  const frozen = reducer(done, { type: "transport/reset", source: "node:c1:a" });
  strictEqual(frozen, done, "reset while frozen is ignored");
  strictEqual(frozen.transcript.events.length, 1);
});

check("transcriptLoaded", () => {
  const three = [ev(0, "text", { text: "a" }), ev(1, "text", { text: "b" }), ev(2, "text", { text: "c" })];
  const loaded = reducer(makeInitialState(), {
    type: "transcript/loaded",
    source: "node:c1:a",
    events: three,
    next: 3,
    at: 1,
  });
  strictEqual(loaded.transcript.events.length, 3);
  const shorter = reducer(loaded, {
    type: "transcript/loaded",
    source: "node:c1:a",
    events: three.slice(0, 1),
    next: 1,
    at: 2,
  });
  strictEqual(shorter, loaded, "a shorter list must not replace a longer one");
  const longer = reducer(loaded, {
    type: "transcript/loaded",
    source: "node:c1:a",
    events: [...three, ev(3, "text", { text: "d" })],
    next: 4,
    at: 3,
  });
  strictEqual(longer.transcript.events.length, 4);
  const other = reducer(longer, {
    type: "transcript/loaded",
    source: "node:c1:b",
    events: [ev(0, "text", { text: "z" })],
    next: 1,
    at: 4,
  });
  strictEqual(other.transcript.source, "node:c1:b");
  strictEqual(other.transcript.events.length, 1);
  const loading = reducer(other, { type: "transcript/loading", source: "node:c1:b", at: 5 });
  strictEqual(loading.transcript.status, "loading");
  deepStrictEqual(loading.transcript.events, []);
});

check("tabs", () => {
  const empty = makeInitialState();
  strictEqual(reducer(empty, { type: "ui/open", source: "bogus" }), empty, "an invalid source opens nothing");
  let state = reducer(empty, { type: "ui/open", source: "node:c1:a" });
  strictEqual(state.ui.tabs.length, 1);
  strictEqual(state.ui.activeId, "node:c1:a");
  deepStrictEqual(state.ui.tabs[0], {
    id: "node:c1:a",
    kind: "node",
    cycle: "c1",
    target: "a",
    view: "transcript",
  });
  strictEqual(state.ui.cycleId, "c1");
  const focused = reducer(state, { type: "ui/open", source: "node:c1:a", view: "diff" });
  strictEqual(focused.ui.tabs.length, 1, "opening the same source focuses it");
  strictEqual(focused.ui.tabs[0].view, "diff");
  const kept = reducer(focused, { type: "ui/open", source: "node:c1:a" });
  strictEqual(kept.ui.tabs[0].view, "diff", "focusing keeps the tab's view");
  const policy = reducer(empty, { type: "ui/open", source: "policy:c1:2" });
  strictEqual(policy.ui.tabs[0].kind, "policy");
  strictEqual(policy.ui.tabs[0].view, "prompt");
  let cap = empty;
  for (let i = 0; i < MAX_TABS; i += 1) cap = reducer(cap, { type: "ui/open", source: `node:c1:n${i}` });
  strictEqual(cap.ui.tabs.length, MAX_TABS);
  const over = reducer(cap, { type: "ui/open", source: "node:c1:extra" });
  strictEqual(over.ui.tabs.length, MAX_TABS);
  strictEqual(over.ui.tabs[0].id, "node:c1:n1", "the oldest inactive tab is dropped");
  strictEqual(over.ui.activeId, "node:c1:extra");
  let three = empty;
  for (const id of ["node:c1:a", "node:c1:b", "node:c1:c"]) three = reducer(three, { type: "ui/open", source: id });
  three = reducer(three, { type: "ui/activate", id: "node:c1:b" });
  strictEqual(three.ui.activeId, "node:c1:b");
  const closed = reducer(three, { type: "ui/close", id: "node:c1:b" });
  deepStrictEqual(closed.ui.tabs.map((tab) => tab.id), ["node:c1:a", "node:c1:c"]);
  strictEqual(closed.ui.activeId, "node:c1:c", "closing the active tab activates its neighbour");
  const last = reducer(closed, { type: "ui/close", id: "node:c1:c" });
  strictEqual(last.ui.activeId, "node:c1:a", "closing the last tab activates the previous");
  const none = reducer(last, { type: "ui/close", id: "node:c1:a" });
  strictEqual(none.ui.activeId, null);
  strictEqual(none.ui.tabs.length, 0);
  strictEqual(reducer(three, { type: "ui/close", id: "nope" }), three);
  strictEqual(reducer(three, { type: "ui/activate", id: "nope" }), three);
});

check("view", () => {
  const empty = makeInitialState();
  strictEqual(reducer(empty, { type: "ui/view", view: "diff" }), empty, "no active tab, no view change");
  let state = reducer(empty, { type: "ui/open", source: "node:c1:a" });
  state = reducer(state, { type: "ui/view", view: "diff" });
  strictEqual(state.ui.tabs[0].view, "diff");
  strictEqual(reducer(state, { type: "ui/view", view: "prompt" }), state, "a policy view is invalid for a node");
  let policy = reducer(empty, { type: "ui/open", source: "policy:c1:1" });
  policy = reducer(policy, { type: "ui/view", view: "replay" });
  strictEqual(policy.ui.tabs[0].view, "replay");
  strictEqual(reducer(policy, { type: "ui/view", view: "solution" }), policy, "a node-only view is invalid for a step");
  const stepTranscript = reducer(policy, { type: "ui/view", view: "transcript" });
  strictEqual(stepTranscript.ui.tabs[0].view, "transcript", "a policy step has a transcript view");
});

check("treeRows", () => {
  const nodes = [
    node("root", null, null, null, "done", { fail_class: "ok", score: 0 }),
    node("b1", "root", 1, 1, "done", { fail_class: "ok", score: 0.4 }),
    node("b1a1", "b1", 1, 1, "agent"),
    node("b1a1x1", "b1a1", 1, 1, "pending"),
    node("b2", "root", 2, 1, "pending"),
    node("loose", null, 1, 1, "pending"),
  ];
  const cycle = cycleWith(nodes);
  const state = withData({ loop: { status: "running", running: true }, cycles: [cycle] }, { cycleId: C1 });
  const rows = sel.treeRows(state, cycle);
  deepStrictEqual(
    rows.map((row) => row.node.cell_id),
    ["root", "b1", "b1a1", "b1a1x1", "b2", "loose"],
  );
  deepStrictEqual(
    rows.map((row) => row.depth),
    [0, 1, 2, 3, 1, 0],
  );
  strictEqual(rows[0].isLast, true);
  strictEqual(rows[1].isLast, false);
  strictEqual(rows[1].hasChildren, true);
  strictEqual(rows[3].hasChildren, false);
  strictEqual(rows[5].depth, 0, "a parentless orphan sits at the root");
  deepStrictEqual(rows[2].guides, [false, true]);
  for (const row of rows) {
    strictEqual(row.guides.length, row.depth, `${row.node.cell_id} guide count`);
    ok(
      row.guides.every((guide) => typeof guide === "boolean"),
      `${row.node.cell_id} guides are booleans`,
    );
    ok(["pending", "agent", "evaluating", "stopped", "done"].includes(row.phase), `${row.node.cell_id} phase`);
  }
  strictEqual(new Set(rows.map((row) => row.node.cell_id)).size, rows.length, "no duplicate rows");
});

const EVENTS = [
  ev(0, "thinking", { text: "We need " }),
  ev(1, "thinking", { text: "to check." }),
  ev(2, "text", { text: "Hello" }),
  ev(3, "tool_call", { tool: "bash", tool_use_id: "t1", input: { summary: "run tests" } }),
  ev(4, "tool_result", { tool: "bash", tool_use_id: "t1", output: "boom", is_error: true }),
  ev(5, "tool_result", { tool: "read", tool_use_id: "zzz", output: "orphan output" }),
  ev(6, "result", { text: "done", is_error: true, meta: { turns: 3 } }),
  ev(7, "system", { text: "session start" }),
];
const blocksFor = (events) => sel.groupEvents(events);

check("transcriptBlocks", () => {
  const blocks = blocksFor(EVENTS);
  deepStrictEqual(
    blocks.map((block) => block.kind),
    ["thinking", "text", "tool", "toolresult", "result", "system"],
  );
  deepStrictEqual(
    blocks.map((block) => block.index),
    [0, 1, 2, 3, 4, 5],
  );
  strictEqual(blocks[0].text, "We need to check.");
  strictEqual(blocks[0].chars, blocks[0].text.length);
  strictEqual(blocks[0].i0, 0);
  strictEqual(blocks[0].i1, 1);
  strictEqual(blocks[2].output, "boom");
  strictEqual(blocks[2].isError, true);
  strictEqual(blocks[2].done, true);
  strictEqual(blocks[2].i1, 4);
  strictEqual(blocks[3].output, "orphan output");
  strictEqual(blocks[3].isError, false);
  strictEqual(blocks[4].isError, true);
  strictEqual(blocks[5].text, "session start");
  const split = blocksFor([
    ev(0, "text", { text: "a" }),
    ev(1, "tool_call", { tool: "x", tool_use_id: "q", input: {} }),
    ev(2, "text", { text: "b" }),
  ]);
  deepStrictEqual(
    split.map((block) => block.kind),
    ["text", "tool", "text"],
    "a delta run does not merge across a tool call",
  );
});

check("filters", () => {
  const blocks = blocksFor(EVENTS);
  const kinds = (list) => list.map((block) => block.kind);
  deepStrictEqual(kinds(sel.filterBlocks(blocks, "reasoning", "")), ["thinking"]);
  deepStrictEqual(kinds(sel.filterBlocks(blocks, "tools", "")), ["tool", "toolresult"]);
  deepStrictEqual(kinds(sel.filterBlocks(blocks, "errors", "")), ["tool", "result"]);
  deepStrictEqual(kinds(sel.filterBlocks(blocks, "messages", "")), ["text", "result"]);
  deepStrictEqual(kinds(sel.filterBlocks(blocks, "all", "BOOM")), ["tool"]);
  deepStrictEqual(sel.searchHits(blocks, "boom"), [2]);
  deepStrictEqual(sel.searchHits(blocks, "orphan"), [3]);
  deepStrictEqual(sel.searchHits(blocks, "   "), []);
  deepStrictEqual(sel.searchHits(blocks, "absent"), []);
});

check("scores", () => {
  const cycle = cycleWith([
    node("root", null, null, null, "done", { fail_class: "ok", score: 0 }),
    node("a", "root", 1, 1, "done", { fail_class: "ok", score: 0.4 }),
    node("b", "root", 2, 1, "done", { fail_class: "ok", score: 0.5 }),
    node("c", "root", 3, 1, "done", { fail_class: "gate_failed", score: 0.9 }),
    node("d", "root", 4, 1, "done", { fail_class: "ok", score: null }),
  ]);
  const series = sel.scoreSeries(cycle);
  deepStrictEqual(
    series.map((point) => point.cell),
    ["a", "b"],
  );
  deepStrictEqual(
    series.map((point) => point.score),
    [0.4, 0.5],
  );
  deepStrictEqual(sel.frontier(cycle), [
    { x: 0, score: 0.4, best: 0.4 },
    { x: 1, score: 0.5, best: 0.5 },
  ]);
  deepStrictEqual(sel.scoreSeries(null), []);
});

check("nextSource", () => {
  const cycle = cycleWith([
    node("root", null, null, null, "done", { fail_class: "ok", score: 0 }),
    node("a", "root", 1, 1, "done", { fail_class: "ok", score: 0.4 }),
    node("b", "root", 2, 1, "pending"),
  ]);
  const state = withData({ loop: { status: "running", running: true }, cycles: [cycle] }, { cycleId: C1 });
  strictEqual(sel.nextSource(state, 1), "node:c1:root", "no active tab starts at the first row");
  const atA = reducer(state, { type: "ui/open", source: "node:c1:a" });
  strictEqual(sel.nextSource(atA, 1), "node:c1:b");
  const atB = reducer(state, { type: "ui/open", source: "node:c1:b" });
  strictEqual(sel.nextSource(atB, 1), "node:c1:root", "forward wraps");
  strictEqual(sel.nextSource(atB, -1), "node:c1:a");
  const atRoot = reducer(state, { type: "ui/open", source: "node:c1:root" });
  strictEqual(sel.nextSource(atRoot, -1), "node:c1:b", "backward wraps");
  strictEqual(sel.nextSource(withData({ loop: {}, cycles: [] }), 1), null);
});

check("defaultSource", () => {
  const cycle = cycleWith([
    node("root", null, null, null, "done", { fail_class: "ok", score: 0 }),
    node("a", "root", 1, 1, "agent"),
    node("b", "root", 2, 1, "done", { fail_class: "ok", score: 0.5 }),
    node("s", "root", 3, 1, "stopped"),
  ]);
  const live = withData({ loop: { status: "running", running: true }, cycles: [cycle] }, { cycleId: C1 });
  strictEqual(sel.defaultSource(live), "node:c1:a", "a live node wins");
  const stopped = withData({ loop: { status: "stopped", running: false }, cycles: [cycle] }, { cycleId: C1 });
  strictEqual(sel.defaultSource(stopped), "node:c1:b", "a demoted live node falls back to the best");
  const noBest = withData(
    {
      loop: { status: "stopped", running: false },
      cycles: [
        cycleWith([
          node("root", null, null, null, "done", { fail_class: "ok", score: 0 }),
          node("a", "root", 1, 1, "stopped"),
        ]),
      ],
    },
    { cycleId: C1 },
  );
  strictEqual(sel.defaultSource(noBest), "node:c1:a", "a stopped node is preferred over the root");
  const onlyRoot = withData(
    {
      loop: { status: "stopped", running: false },
      cycles: [cycleWith([node("root", null, null, null, "done", { fail_class: "ok", score: 0 })])],
    },
    { cycleId: C1 },
  );
  strictEqual(sel.defaultSource(onlyRoot), "node:c1:root");
  strictEqual(sel.defaultSource(withData({ loop: { status: "idle", running: false }, cycles: [] })), null);
});

check("artifactGuard", () => {
  const source = "node:c1:a";
  const loaded = reducer(makeInitialState(), {
    type: "artifact/loaded",
    source,
    name: "proposal",
    text: "v2",
    at: 200,
  });
  strictEqual(sel.artifact(loaded, source, "proposal").text, "v2");
  strictEqual(sel.artifactKey(source, "proposal"), `${source}|proposal`);
  strictEqual(sel.artifact(loaded, source, "diff"), null);
  const older = reducer(loaded, { type: "artifact/loaded", source, name: "proposal", text: "v1", at: 100 });
  strictEqual(older, loaded, "an older result must not overwrite a newer one");
  const olderError = reducer(loaded, { type: "artifact/error", source, name: "proposal", error: "late", at: 100 });
  strictEqual(olderError, loaded);
  const newer = reducer(loaded, { type: "artifact/loaded", source, name: "proposal", text: "v3", at: 300 });
  strictEqual(newer.artifacts.byKey[`${source}|proposal`].text, "v3");
  strictEqual(reducer(loaded, { type: "artifact/loading", source, name: "proposal", at: 400 }), loaded);
});

check("purity", () => {
  const actions = [
    {
      type: "state/loaded",
      data: {
        loop: { status: "running", running: true },
        cycles: [{ cycle_id: C1, iteration: 1, status: "running", nodes: [], policy_dev: { steps: [] } }],
      },
      at: 1,
      seq: 2,
    },
    { type: "transport/attach", source: "node:c1:a", after: 0 },
    { type: "transport/hello", payload: { source: "node:c1:a", cursor: 0 } },
    { type: "transport/events", source: "node:c1:a", events: [ev(0, "text", { text: "x" })], next: 1, at: 2 },
    { type: "transport/reset", source: "node:c1:a" },
    { type: "transport/done", source: "node:c1:a", reason: "score", next: 1 },
    { type: "ui/open", source: "node:c1:a", view: "diff" },
    { type: "ui/view", view: "evaluation" },
    { type: "ui/close", id: "node:c1:a" },
    { type: "artifact/loaded", source: "node:c1:a", name: "diff", text: "x", at: 3 },
    { type: "context/loaded", source: "node:c1:a", payload: { sections: [] }, at: 3 },
  ];
  const first = actions.reduce(reducer, makeInitialState());
  const second = actions.reduce(reducer, makeInitialState());
  deepStrictEqual(first, second, "the same action sequence must produce the same state");
  for (const action of actions) {
    const state = makeInitialState();
    const snapshot = structuredClone(state);
    reducer(state, action);
    deepStrictEqual(state, snapshot, `${action.type} mutated its input`);
  }
});

check("store", () => {
  const store = createStore(reducer, makeInitialState());
  let calls = 0;
  let last = null;
  const unsubscribe = store.subscribe((state) => {
    calls += 1;
    last = state;
  });
  const next = store.dispatch({ type: "ui/theme", theme: "dark" });
  strictEqual(next.ui.theme, "dark");
  strictEqual(store.getState(), next);
  strictEqual(calls, 1);
  strictEqual(last, next);
  strictEqual(store.dispatch({ type: "unknown/action" }), next);
  strictEqual(calls, 1, "an ignored action does not notify");
  store.dispatch(null);
  strictEqual(calls, 1);
  unsubscribe();
  store.dispatch({ type: "ui/theme", theme: "light" });
  strictEqual(calls, 1, "unsubscribe stops notifications");
  strictEqual(store.getState().ui.theme, "light");
  let seen = null;
  const hooked = createStore(reducer, makeInitialState(), {
    onDispatch: (action, prev, after) => {
      seen = { action, prev, after };
    },
  });
  hooked.dispatch({ type: "ui/theme", theme: "dark" });
  strictEqual(seen.action.type, "ui/theme");
  strictEqual(seen.prev.ui.theme, "system");
  strictEqual(seen.after.ui.theme, "dark");
});

process.stdout.write(JSON.stringify({ checks }));
