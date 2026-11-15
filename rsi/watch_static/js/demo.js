/* Demo adapter over the sanitized fixtures. Same shape as the live API, plus a scripted
   stream so ?demo=1 exercises the transcript without a server.
   Scenarios: ?demo=1 (running), &scenario=stopped|error|empty. */

const CHUNK = 4;
const TICK_MS = 90;

export async function createDemo() {
  const [state, bundle] = await Promise.all([
    fetch("fixture.json", { cache: "no-store" }).then((r) => r.json()),
    fetch("fixture_events.json", { cache: "no-store" }).then((r) => r.json()),
  ]);
  const scenario = new URLSearchParams(location.search).get("scenario") || "running";
  const data = applyScenario(state, scenario);
  const live = new Set(bundle.live || []);

  function eventsFor(source) {
    return bundle.events[source] || [];
  }

  return {
    state: async () => ({ ...data, now: new Date().toISOString() }),
    events: async (source, after = 0) => {
      const events = eventsFor(source);
      const slice = events.slice(after);
      return { events: slice, next: after + slice.length, done: !live.has(source) };
    },
    artifact: async (source, name) => {
      const text = bundle.artifacts?.[source]?.[name];
      if (text == null) {
        const error = new Error("artifact not found");
        error.status = 404;
        throw error;
      }
      return { name, text };
    },
    context: async (source) => {
      const payload = bundle.context?.[source];
      if (!payload) {
        const error = new Error("context not found");
        error.status = 404;
        throw error;
      }
      return payload;
    },
    stream: ({ source, after = 0, onEvent, onStatus }) => {
      const events = eventsFor(source);
      let cursor = after;
      let timer = null;
      let closed = false;
      const isLive = live.has(source);
      const still = new URLSearchParams(location.search).has("still");
      onStatus && onStatus("open");
      onEvent({ type: "hello", payload: { v: 1, source, cursor, interval_ms: TICK_MS } });
      if (still) {
        if (events.length > cursor) onEvent({ type: "events", events: events.slice(cursor), next: events.length });
        onEvent({ type: "done", reason: isLive ? "score" : "settled", next: events.length });
        return { close() {} };
      }
      const pump = () => {
        if (closed) return;
        if (cursor >= events.length) {
          onEvent({ type: "done", reason: isLive ? "score" : "settled", next: cursor });
          return;
        }
        const batch = events.slice(cursor, cursor + CHUNK);
        cursor += batch.length;
        onEvent({ type: "events", events: batch, next: cursor });
        timer = setTimeout(pump, TICK_MS);
      };
      timer = setTimeout(pump, TICK_MS);
      return {
        close() {
          closed = true;
          if (timer) clearTimeout(timer);
        },
      };
    },
  };
}

function applyScenario(state, scenario) {
  if (scenario === "empty") {
    return {
      ...state,
      study_dir: "empty-study",
      loop: {
        status: "idle",
        stop_reason: null,
        iteration: null,
        stage: null,
        cycles_target: null,
        elapsed_s: null,
        agent_cost_usd: 0,
        best_score: null,
        running: false,
        probe_budget: null,
      },
      gpu: null,
      cycles: [],
    };
  }
  if (scenario === "stopped" || scenario === "error") {
    const loop = {
      ...state.loop,
      status: scenario,
      running: false,
      stage: null,
      stop_reason: scenario === "error" ? "agent backend exited 1" : "stopped by operator",
    };
    const cycles = state.cycles.map((cycle) => ({
      ...cycle,
      status: cycle.status === "running" ? "stopped" : cycle.status,
      nodes: cycle.nodes.map((node) =>
        node.status === "agent" || node.status === "evaluating"
          ? { ...node, status: "stopped", elapsed_s: null }
          : node,
      ),
    }));
    return { ...state, loop, cycles };
  }
  return state;
}
