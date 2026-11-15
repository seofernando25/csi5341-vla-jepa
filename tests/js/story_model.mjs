/* Story model checks. Plain node script, no dependencies:
 *   node tests/js/story_model.mjs
 * Fixtures are hand-written; no run numbers from studies/ are used. */

import assert from "node:assert/strict";

const url = new URL("../../rsi/watch_static/js/story/model.js", import.meta.url);
const model = await import(url.href);

const {
  baseline,
  cycleOf,
  probeNode,
  nodeOrder,
  frontierSeries,
  probeTimeline,
  reasoningLines,
  dreamModel,
  dreamFrame,
  changedModel,
  diffLines,
} = model;

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`ok ${passed} - ${name}`);
  } catch (error) {
    failures.push(name);
    console.log(`not ok - ${name}`);
    console.log(String(error?.stack || error).replace(/^/gm, "  "));
  }
}

function almost(actual, expected, epsilon = 1e-9) {
  assert.equal(typeof actual, "number", `expected a number, got ${actual}`);
  assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} is not ${expected}`);
}

function ids(nodes) {
  return nodes.map((node) => node.cell_id);
}

/* ---------------------------------------------------------------------------
 * Fixtures
 * ------------------------------------------------------------------------- */

const NODES = [
  { cell_id: "root", branch: 0, attempt: 0, status: "done", score: 0, started_utc: "2026-01-01T00:00:00Z" },
  { cell_id: "b1_a0", branch: 1, attempt: 0, status: "evaluating", score: null, started_utc: "2026-01-01T00:04:00Z" },
  { cell_id: "b0_a1", branch: 0, attempt: 1, status: "done", score: 0.5, started_utc: "2026-01-01T00:02:00Z" },
  { cell_id: "b0_a0", branch: 0, attempt: 0, status: "done", score: 0.2, started_utc: "2026-01-01T00:01:00Z" },
  { cell_id: "b0_a2", branch: 0, attempt: 2, status: "done", score: 0.35, started_utc: "2026-01-01T00:03:00Z" },
  { cell_id: "b2_a0", branch: 2, attempt: 0, status: "pending", score: null },
  { cell_id: "b1_a1", branch: 1, attempt: 1, status: "pending", score: null },
];

const STATE = {
  ui: { cycleId: "cyc_b" },
  server: {
    data: {
      loop: {
        probe_budget: 40,
        cycles_target: 3,
        baseline: { score: null, resident_bytes: 1_900_000_000 },
      },
      protocol: {
        smoke_gate: 0.4,
        smoke_trials: 10,
        dev_gate: 0.35,
        dev_trials: 100,
        agent: { backend: "pi", model: "test-model", effort: "max" },
      },
      cycles: [
        { cycle_id: "cyc_a", iteration: 0, nodes: [NODES[0]] },
        { cycle_id: "cyc_b", iteration: 1, nodes: NODES },
      ],
    },
  },
};

function stateWith(nodes) {
  return {
    ui: { cycleId: "c1" },
    server: { data: { cycles: [{ cycle_id: "c1", iteration: 1, nodes }] } },
  };
}

const EVENTS = {
  events: [
    { i: 7, kind: "result", text: "finished", meta: { turns: 3 } },
    { i: 2, kind: "thinking", text: "policy and read its gate.\nThis second sentence is comfortably longer than the first one." },
    { i: 5, kind: "tool_call", tool: "read", tool_use_id: "t2", input: { summary: "open file", command: null } },
    { i: 0, kind: "system", text: "session started" },
    { i: 4, kind: "tool_result", tool_use_id: "t1", output: "x".repeat(500), is_error: false },
    { i: 6, kind: "tool_result", tool_use_id: "t2", output: "boom", is_error: true },
    { i: 1, kind: "thinking", text: "We start from the baseline " },
    { i: 3, kind: "tool_call", tool: "bash", tool_use_id: "t1", input: { summary: "run smoke", command: "python -m eval" } },
    { i: 8, kind: "thinking", text: "\nTiny.\nA third sentence that sits between the two in length." },
  ],
};

const DREAM = {
  cycle: "cyc_b",
  status: "ready",
  protocol: { m: 4, beta_grid: [0, 0.5, 1] },
  worlds: [{ id: "w0", nodes: 4, baseline_score: 0, max_gain: 0.5, degenerate: false }],
  tree: {
    world_id: "w0",
    nodes: [{ cell_id: "b0_a0", parent_id: null, branch: 0, attempt: 0, seq: 1, score: 0.1 }],
  },
  candidates: [
    { slot: "m0_incumbent", step: 0, status: "incumbent", mean_v: 0.4, delta_vs_best: null, attempts: 0, error: null },
    { slot: "m1_llm", step: 1, status: "selected", mean_v: 0.5, delta_vs_best: 0.1, attempts: 2, error: null },
  ],
  curves: {
    m0_incumbent: {
      w0: [
        { beta: 0.5, probes: 1, attainment: 0.2, rounds: 1 },
        { beta: 1, probes: 2, attainment: 0.3, rounds: 2 },
      ],
    },
    m1_llm: { w0: [{ beta: 0.5, probes: 2, attainment: 0.35, rounds: 2 }] },
  },
  rounds: {
    m0_incumbent: {
      w0: {
        0.5: [
          { round: 1, batch: ["b0_a0"], prefix_size: 0, outcomes: [{ cell_id: "b0_a0", score: 0.1, fail_class: "ok" }] },
        ],
      },
    },
    m1_llm: {
      w0: {
        0.5: [
          { round: 1, batch: ["b0_a1"], prefix_size: 0, outcomes: [{ cell_id: "b0_a1", score: 0.2, fail_class: "ok" }] },
          {
            round: 2,
            batch: ["b1_a0", "b1_a1"],
            prefix_size: 1,
            outcomes: [
              { cell_id: "b1_a0", score: 0.3, fail_class: "ok" },
              { cell_id: "b1_a1", score: 0, fail_class: "gate_failed" },
            ],
          },
        ],
      },
    },
  },
  heldout: { num_worlds: 24, mean_V_by_slot: {}, delta_vs_incumbent: 0.05 },
  verdict: { selected_slot: "m1_llm", incumbent_in_sample_V: 0.4, selected_in_sample_V: 0.52 },
  agent: { backend: "pi", model: "test-model", effort: "max", calls: 2, num_turns: 10, cost_usd: 0.1 },
};

/* ---------------------------------------------------------------------------
 * Surface
 * ------------------------------------------------------------------------- */

test("exports every contract function", () => {
  const names = [
    "baseline",
    "cycleOf",
    "probeNode",
    "nodeOrder",
    "frontierSeries",
    "probeTimeline",
    "reasoningLines",
    "dreamModel",
    "dreamFrame",
    "changedModel",
    "diffLines",
  ];
  for (const name of names) assert.equal(typeof model[name], "function", name);
});

/* ---------------------------------------------------------------------------
 * baseline / cycleOf
 * ------------------------------------------------------------------------- */

test("baseline reads the loop, protocol and agent", () => {
  const out = baseline(STATE);
  assert.equal(out.bytes, 1_900_000_000);
  assert.equal(out.score, null);
  assert.equal(out.gate, 0.4);
  assert.equal(out.trials, 10);
  assert.equal(out.devGate, 0.35);
  assert.equal(out.devTrials, 100);
  assert.equal(out.budget, 40);
  assert.equal(out.cyclesTarget, 3);
  assert.equal(out.model, "test-model");
  assert.equal(out.backend, "pi");
});

test("baseline tolerates an empty state", () => {
  const out = baseline({});
  for (const [key, value] of Object.entries(out)) assert.equal(value, null, key);
});

test("cycleOf defaults to the ui cycle", () => {
  assert.equal(cycleOf(STATE).cycle_id, "cyc_b");
  assert.equal(cycleOf(STATE, "cyc_a").cycle_id, "cyc_a");
});

test("cycleOf falls back to the last cycle without a ui id", () => {
  assert.equal(cycleOf({ server: { data: { cycles: [{ cycle_id: "a" }, { cycle_id: "b" }] } } }).cycle_id, "b");
});

test("cycleOf returns null for an unrecorded explicit id", () => {
  assert.equal(cycleOf(STATE, "cyc_z"), null);
  assert.equal(cycleOf({}), null);
});

/* ---------------------------------------------------------------------------
 * Nodes
 * ------------------------------------------------------------------------- */

test("nodeOrder falls back to the grid when any node lacks a timestamp", () => {
  const nodes = [
    { cell_id: "root", branch: null, attempt: null, status: "done", started_utc: null },
    { cell_id: "b0_a0", branch: 0, attempt: 0, status: "done", started_utc: null },
    { cell_id: "b2_a0", branch: 2, attempt: 0, status: "agent", started_utc: "2026-01-01T00:00:01Z" },
    { cell_id: "b1_a0", branch: 1, attempt: 0, status: "evaluating", started_utc: "2026-01-01T00:00:02Z" },
  ];
  assert.deepEqual(ids(nodeOrder({ nodes })), ["root", "b0_a0", "b1_a0", "b2_a0"]);
  assert.deepEqual(ids(nodeOrder({ nodes: NODES })), [
    "root",
    "b0_a0",
    "b0_a1",
    "b0_a2",
    "b1_a0",
    "b1_a1",
    "b2_a0",
  ]);
});

test("nodeOrder sorts by started_utc when every node has one", () => {
  const nodes = [
    { cell_id: "b1_a0", branch: 1, attempt: 0, started_utc: "2026-01-01T00:00:02Z" },
    { cell_id: "b0_a0", branch: 0, attempt: 0, started_utc: "2026-01-01T00:00:03Z" },
    { cell_id: "b2_a0", branch: 2, attempt: 0, started_utc: "2026-01-01T00:00:01Z" },
    { cell_id: "root", branch: null, attempt: null, started_utc: "2026-01-01T00:00:00Z" },
  ];
  assert.deepEqual(ids(nodeOrder({ nodes })), ["root", "b2_a0", "b1_a0", "b0_a0"]);
});

test("nodeOrder is stable and keeps the root first", () => {
  const nodes = [
    { cell_id: "x", branch: 1, attempt: 0, status: "pending" },
    { cell_id: "y", branch: 1, attempt: 0, status: "pending" },
  ];
  assert.deepEqual(ids(nodeOrder({ nodes })), ["x", "y"]);
  assert.deepEqual(
    ids(nodeOrder({ nodes: [{ cell_id: "b0_a0", branch: 0, attempt: 0 }, { cell_id: "root" }] })),
    ["root", "b0_a0"],
  );
  assert.deepEqual(nodeOrder({}), []);
});

test("probeNode prefers the best scored done node", () => {
  const out = probeNode(STATE);
  assert.equal(out.cycle.cycle_id, "cyc_b");
  assert.equal(out.node.cell_id, "b0_a1");
});

test("probeNode falls back to the latest live node", () => {
  const state = stateWith([
    { cell_id: "root", branch: 0, attempt: 0, status: "done", score: 0, started_utc: "2026-01-01T00:00:00Z" },
    { cell_id: "b0_a0", branch: 0, attempt: 0, status: "evaluating", started_utc: "2026-01-01T00:01:00Z" },
    { cell_id: "b1_a0", branch: 1, attempt: 0, status: "agent", started_utc: "2026-01-01T00:02:00Z" },
  ]);
  assert.equal(probeNode(state).node.cell_id, "b1_a0");
});

test("probeNode returns null when nothing qualifies", () => {
  const state = stateWith([
    { cell_id: "root", branch: 0, attempt: 0, status: "done", score: 0 },
    { cell_id: "b0_a0", branch: 0, attempt: 0, status: "pending" },
  ]);
  assert.equal(probeNode(state), null);
  assert.equal(probeNode({}), null);
});

test("frontierSeries keeps scored nodes in order with a running best", () => {
  const out = frontierSeries({ nodes: NODES });
  assert.deepEqual(out, [
    { i: 0, cell: "b0_a0", score: 0.2, best: 0.2 },
    { i: 1, cell: "b0_a1", score: 0.5, best: 0.5 },
    { i: 2, cell: "b0_a2", score: 0.35, best: 0.5 },
  ]);
});

test("frontierSeries skips the root and unscored nodes", () => {
  const out = frontierSeries({
    nodes: [
      { cell_id: "root", status: "done", score: 0 },
      { cell_id: "a", status: "done", score: null },
      { cell_id: "b", status: "done", score: 0.1 },
    ],
  });
  assert.deepEqual(out, [{ i: 0, cell: "b", score: 0.1, best: 0.1 }]);
  assert.deepEqual(frontierSeries(null), []);
});

/* ---------------------------------------------------------------------------
 * Transcript
 * ------------------------------------------------------------------------- */

test("probeTimeline counts events by kind", () => {
  const out = probeTimeline(EVENTS);
  assert.equal(out.counts.events, 9);
  assert.equal(out.counts.thinking, 3);
  assert.equal(out.counts.tool_call, 2);
  assert.equal(out.counts.tool_result, 2);
  assert.equal(out.counts.result, 1);
  assert.equal(out.counts.system, 1);
  assert.equal(probeTimeline(null).counts.events, 0);
});

test("reasoningLines joins deltas in i order and keeps the longest in reading order", () => {
  const lines = reasoningLines(EVENTS, 6);
  assert.equal(lines.length, 3);
  assert.ok(lines[0].startsWith("We start from the baseline policy"));
  assert.ok(lines[1].startsWith("This second sentence"));
  assert.ok(lines[2].startsWith("A third sentence"));
  for (const line of lines) assert.ok(line.length >= 20, line);
  assert.equal(new Set(lines).size, lines.length);
});

test("reasoningLines respects max and drops short fragments", () => {
  const lines = reasoningLines(EVENTS, 2);
  assert.equal(lines.length, 2);
  assert.ok(lines[0].startsWith("This second sentence"));
  assert.ok(lines[1].startsWith("A third sentence"));
  assert.deepEqual(reasoningLines(EVENTS, 0), []);
  assert.deepEqual(reasoningLines({}, 6), []);
});

test("probeTimeline pairs tool results and truncates output", () => {
  const out = probeTimeline(EVENTS);
  assert.equal(out.tools.length, 2);
  const [bash, read] = out.tools;
  assert.equal(bash.i, 3);
  assert.equal(bash.name, "bash");
  assert.equal(bash.summary, "run smoke");
  assert.equal(bash.command, "python -m eval");
  assert.equal(bash.output.length, 400);
  assert.equal(bash.isError, false);
  assert.equal(read.name, "read");
  assert.equal(read.command, null);
  assert.equal(read.output, "boom");
  assert.equal(read.isError, true);
});

test("probeTimeline returns the last result event", () => {
  const out = probeTimeline(EVENTS);
  assert.equal(out.result.text, "finished");
  assert.equal(out.result.meta.turns, 3);
  assert.equal(probeTimeline({ events: [] }).result, null);
});

/* ---------------------------------------------------------------------------
 * Dream
 * ------------------------------------------------------------------------- */

test("dreamModel marks a missing payload pending with empty helpers", () => {
  for (const payload of [null, {}, { status: "pending" }, { status: "missing" }]) {
    const out = dreamModel(payload);
    assert.equal(out.pending, true);
    assert.deepEqual(out.worlds, []);
    assert.deepEqual(out.candidates, []);
    assert.equal(out.world("w0"), null);
    assert.equal(out.tree("w0"), null);
    assert.deepEqual(out.rounds("m0", "w0", 0.5), []);
  }
});

test("dreamModel normalises a ready payload", () => {
  const out = dreamModel(DREAM);
  assert.equal(out.pending, false);
  assert.equal(out.worlds.length, 1);
  assert.equal(out.candidates.length, 2);
  assert.equal(out.verdict.selected_slot, "m1_llm");
  assert.equal(out.heldout.delta_vs_incumbent, 0.05);
  assert.equal(out.agent.backend, "pi");
  assert.equal(out.protocol.m, 4);
  assert.equal(out.world("w0").nodes, 4);
  assert.equal(out.world().id, "w0");
  assert.equal(out.tree("w0").nodes.length, 1);
  assert.equal(out.rounds("m1_llm", "w0", 0.5).length, 2);
});

test("dreamModel misses unknown worlds, trees and slots", () => {
  const out = dreamModel(DREAM);
  assert.equal(out.world("w9"), null);
  assert.equal(out.tree("w9"), null);
  assert.deepEqual(out.rounds("nope", "w0", 0.5), []);
  assert.deepEqual(out.rounds("m1_llm", "w0", 0.25), []);
  assert.equal(dreamModel({ status: "ready" }).curves, null);
});

test("dreamFrame stops at the requested round", () => {
  const frame = dreamFrame(DREAM, "w0", 0.5, 1);
  assert.equal(frame.last, 2);
  assert.equal(frame.rounds, 1);
  const [incumbent, variant] = frame.paths;
  assert.equal(incumbent.slot, "m0_incumbent");
  assert.deepEqual(incumbent.cells, ["b0_a0"]);
  assert.equal(incumbent.probes, 1);
  almost(incumbent.attainment, 0.2);
  assert.deepEqual(variant.cells, ["b0_a1"]);
  assert.equal(variant.probes, 1);
  assert.equal(variant.attainment, null);
});

test("dreamFrame accumulates probes and reads attainment at the last used point", () => {
  const frame = dreamFrame(DREAM, "w0", 0.5, 2);
  assert.equal(frame.rounds, 2);
  const variant = frame.paths[1];
  assert.deepEqual(variant.cells, ["b0_a1", "b1_a0", "b1_a1"]);
  assert.equal(variant.probes, 3);
  almost(variant.attainment, 0.35);
});

test("dreamFrame defaults to the last round and tolerates an unknown world", () => {
  const frame = dreamFrame(DREAM, "w0", 0.5);
  assert.equal(frame.rounds, 2);
  assert.equal(frame.paths[1].probes, 3);
  const empty = dreamFrame(DREAM, "w9", 0.5);
  assert.equal(empty.last, 0);
  assert.equal(empty.rounds, 0);
  assert.deepEqual(empty.paths.map((path) => path.cells), [[], []]);
  assert.deepEqual(dreamFrame(null, "w0", 0.5).paths, []);
});

/* ---------------------------------------------------------------------------
 * Changed
 * ------------------------------------------------------------------------- */

test("changedModel names the incumbent, the selected slot and the delta", () => {
  const out = changedModel(STATE, DREAM);
  assert.equal(out.incumbent.slot, "m0_incumbent");
  assert.equal(out.selected, "m1_llm");
  almost(out.delta, 0.12);
  almost(out.heldoutDelta, 0.05);
  assert.equal(out.variants.length, 2);
  assert.equal(out.variants[0].meanV, 0.4);
  assert.equal(out.variants[1].deltaVsBest, 0.1);
  assert.equal(out.variants[1].attempts, 2);
});

test("changedModel falls back to the cycle policy dev", () => {
  const state = {
    server: {
      data: {
        cycles: [{ cycle_id: "c1", policy_dev: { status: "accepted", incumbent_v: 0.3, selected_v: 0.45 } }],
      },
    },
  };
  const out = changedModel(state, { candidates: [] });
  almost(out.delta, 0.15);
  assert.equal(out.selected, null);
  assert.equal(out.incumbent, null);
});

test("changedModel returns a null delta without numbers", () => {
  const out = changedModel({}, null);
  assert.equal(out.delta, null);
  assert.equal(out.heldoutDelta, null);
  assert.deepEqual(out.variants, []);
});

/* ---------------------------------------------------------------------------
 * Diff
 * ------------------------------------------------------------------------- */

test("diffLines marks unchanged lines", () => {
  assert.deepEqual(diffLines("a\nb\nc", "a\nb\nc"), [
    { kind: "same", text: "a" },
    { kind: "same", text: "b" },
    { kind: "same", text: "c" },
  ]);
  assert.deepEqual(diffLines("a\n", "a"), [{ kind: "same", text: "a" }]);
});

test("diffLines reports a deletion and an addition", () => {
  assert.deepEqual(diffLines("a\nb\nc", "a\nc"), [
    { kind: "same", text: "a" },
    { kind: "del", text: "b" },
    { kind: "same", text: "c" },
  ]);
  assert.deepEqual(diffLines("a\nc", "a\nb\nc"), [
    { kind: "same", text: "a" },
    { kind: "add", text: "b" },
    { kind: "same", text: "c" },
  ]);
  assert.deepEqual(diffLines("", "x"), [{ kind: "add", text: "x" }]);
});

test("diffLines caps the output at 500 lines", () => {
  const before = Array.from({ length: 600 }, (_, index) => `old ${index}`).join("\n");
  const after = Array.from({ length: 600 }, (_, index) => `new ${index}`).join("\n");
  const out = diffLines(before, after);
  assert.equal(out.length, 500);
  assert.equal(out[0].kind, "del");
  assert.equal(out[0].text, "old 0");
});

console.log(`# tests ${passed + failures.length}`);
console.log(`# pass ${passed}`);
console.log(`# fail ${failures.length}`);
if (failures.length) process.exitCode = 1;
