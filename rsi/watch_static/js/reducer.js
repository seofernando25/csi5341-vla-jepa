/* The one reducer. Pure: no DOM, no fetch, no storage. Every transition is listed in
   docs/watch_state.md and covered by tests/test_rsi_watch_store.py. */

import { parseSource } from "./selectors.js";
import { defaultView, isView } from "./views/meta.js";

export const MAX_TABS = 12;

export function makeInitialState({ theme = "system" } = {}) {
  return {
    server: { status: "idle", data: null, at: null, seq: 0, error: null },
    transport: {
      source: null,
      status: "idle",
      cursor: 0,
      done: false,
      reason: null,
      frozen: false,
      lastEventAt: null,
      retries: 0,
      error: null,
    },
    transcript: { source: null, events: [], next: 0, status: "idle", error: null, at: null },
    artifacts: { byKey: {} },
    context: { bySource: {} },
    ui: {
      cycleId: null,
      tabs: [],
      activeId: null,
      theme,
      sidebar: { open: true, width: 264 },
      inspector: { open: true, width: 272 },
      outline: { open: true },
      collapsedCycles: [],
      treeCursor: null,
      filter: "all",
      search: { open: false, q: "", focus: 0, at: 0 },
      palette: { open: false, q: "", at: 0 },
      shortcuts: { open: false },
      follow: true,
      drawer: null,
      focusBlock: null,
    },
  };
}

function tabFor(source, view) {
  const parsed = parseSource(source);
  if (!parsed) return null;
  return {
    id: source,
    kind: parsed.kind,
    cycle: parsed.cycle,
    target: parsed.target,
    view: view && isView(parsed.kind, view) ? view : defaultView(parsed.kind),
  };
}

function pickCycle(data) {
  const list = data?.cycles || [];
  return list.find((c) => c.status === "running")?.cycle_id || list[list.length - 1]?.cycle_id || null;
}

function freezeTransport(transport, status = "closed") {
  if (transport.status === "idle") return transport;
  return { ...transport, frozen: true, status: transport.status === "open" ? status : transport.status };
}

export function reducer(state, action) {
  switch (action.type) {
    case "state/loading":
      return { ...state, server: { ...state.server, status: state.server.data ? "ready" : "loading" } };

    case "state/loaded": {
      const data = action.data || {};
      const cycles = data.cycles || [];
      let cycleId = state.ui.cycleId;
      if (!cycleId || !cycles.some((c) => c.cycle_id === cycleId)) cycleId = pickCycle(data);
      let transport = state.transport;
      const ended = ["stopped", "done", "error"].includes(data.loop?.status);
      if (ended && transport.status !== "idle" && !transport.frozen) {
        transport = freezeTransport(transport);
      }
      return {
        ...state,
        server: { status: "ready", data, at: action.at, seq: action.seq ?? state.server.seq, error: null },
        transport,
        ui: { ...state.ui, cycleId },
      };
    }

    case "tick":
      return { ...state, ui: { ...state.ui, tickAt: action.at } };

    case "state/error":
      return { ...state, server: { ...state.server, status: "error", error: action.error, at: action.at } };

    case "transport/attach":
      return {
        ...state,
        transport: {
          ...state.transport,
          source: action.source,
          status: "connecting",
          cursor: action.after,
          done: false,
          reason: null,
          frozen: false,
          error: null,
        },
      };

    case "transport/detach":
      if (action.source && action.source !== state.transport.source) return state;
      return {
        ...state,
        transport: { ...state.transport, status: "idle", frozen: false, done: false },
      };

    case "transport/hello": {
      const payload = action.payload || {};
      const source = payload.source || state.transport.source;
      if (state.transport.source && source !== state.transport.source) return state;
      return {
        ...state,
        transport: {
          ...state.transport,
          source,
          status: "open",
          cursor: typeof payload.cursor === "number" ? payload.cursor : state.transport.cursor,
          retries: 0,
          error: null,
        },
      };
    }

    case "transport/events": {
      if (action.source !== state.transport.source) return state;
      if (state.transport.frozen) return state;
      const events = action.events || [];
      const transcript =
        state.transcript.source === action.source
          ? {
              ...state.transcript,
              events: state.transcript.events.concat(events),
              next: action.next,
              status: "ready",
              at: action.at,
            }
          : {
              source: action.source,
              events: events.slice(),
              next: action.next,
              status: "ready",
              error: null,
              at: action.at,
            };
      return {
        ...state,
        transcript,
        transport: {
          ...state.transport,
          cursor: action.next,
          lastEventAt: action.at,
          status: "open",
        },
      };
    }

    case "transport/reset": {
      if (action.source && action.source !== state.transport.source) return state;
      if (state.transport.frozen) return state;
      return {
        ...state,
        transcript: { ...state.transcript, events: [], next: 0, at: action.at },
        transport: { ...state.transport, cursor: 0, reason: action.reason || null },
      };
    }

    case "transport/done": {
      if (action.source && state.transport.source && action.source !== state.transport.source) {
        return state;
      }
      return {
        ...state,
        transport: {
          ...state.transport,
          status: action.reason === "gone" ? "gone" : "closed",
          done: true,
          reason: action.reason || "settled",
          frozen: true,
          cursor: action.next ?? state.transport.cursor,
        },
      };
    }

    case "transport/error":
      if (action.source && state.transport.source && action.source !== state.transport.source) {
        return state;
      }
      return {
        ...state,
        transport: { ...state.transport, status: "error", error: action.error, retries: state.transport.retries + 1 },
      };

    case "transport/closed": {
      if (action.source && state.transport.source && action.source !== state.transport.source) {
        return state;
      }
      return { ...state, transport: { ...state.transport, status: "closed", frozen: true } };
    }

    case "transcript/loading":
      return {
        ...state,
        transcript: {
          source: action.source,
          events: [],
          next: 0,
          status: "loading",
          error: null,
          at: action.at,
        },
      };

    case "transcript/loaded": {
      const same = state.transcript.source === action.source;
      if (same && state.transcript.events.length > (action.events || []).length) return state;
      return {
        ...state,
        transcript: {
          source: action.source,
          events: action.events || [],
          next: action.next ?? 0,
          status: "ready",
          error: null,
          at: action.at,
        },
      };
    }

    case "transcript/error":
      if (action.source !== state.transcript.source) return state;
      return { ...state, transcript: { ...state.transcript, status: "error", error: action.error } };

    case "ui/open": {
      const tab = tabFor(action.source, action.view);
      if (!tab) return state;
      const existing = state.ui.tabs.find((t) => t.id === tab.id);
      let tabs;
      if (existing) {
        tabs = state.ui.tabs.map((t) =>
          t.id === tab.id ? { ...t, view: action.view && isView(t.kind, action.view) ? action.view : t.view } : t,
        );
      } else {
        tabs = [...state.ui.tabs, tab];
        if (tabs.length > MAX_TABS) {
          const drop = tabs.find((t) => t.id !== state.ui.activeId);
          tabs = tabs.filter((t) => t !== drop);
        }
      }
      return {
        ...state,
        ui: {
          ...state.ui,
          tabs,
          activeId: tab.id,
          cycleId: tab.cycle,
          treeCursor: tab.id,
          follow: true,
          focusBlock: null,
          drawer: null,
        },
      };
    }

    case "ui/close": {
      const index = state.ui.tabs.findIndex((t) => t.id === action.id);
      if (index < 0) return state;
      const tabs = state.ui.tabs.filter((t) => t.id !== action.id);
      let activeId = state.ui.activeId;
      if (activeId === action.id) {
        const next = tabs[index] || tabs[index - 1] || null;
        activeId = next ? next.id : null;
      }
      return { ...state, ui: { ...state.ui, tabs, activeId } };
    }

    case "ui/activate": {
      const tab = state.ui.tabs.find((t) => t.id === action.id);
      if (!tab) return state;
      return {
        ...state,
        ui: { ...state.ui, activeId: tab.id, cycleId: tab.cycle, treeCursor: tab.id, follow: true },
      };
    }

    case "ui/view": {
      const tab = state.ui.tabs.find((t) => t.id === state.ui.activeId);
      if (!tab || !isView(tab.kind, action.view)) return state;
      return {
        ...state,
        ui: {
          ...state.ui,
          tabs: state.ui.tabs.map((t) => (t.id === tab.id ? { ...t, view: action.view } : t)),
        },
      };
    }

    case "ui/cycle": {
      if (!state.server.data?.cycles.some((c) => c.cycle_id === action.cycleId)) return state;
      return { ...state, ui: { ...state.ui, cycleId: action.cycleId } };
    }

    case "ui/theme":
      return { ...state, ui: { ...state.ui, theme: action.theme } };

    case "ui/sidebar":
      return { ...state, ui: { ...state.ui, sidebar: { ...state.ui.sidebar, ...action.patch } } };

    case "ui/inspector":
      return { ...state, ui: { ...state.ui, inspector: { ...state.ui.inspector, ...action.patch } } };

    case "ui/outline":
      return { ...state, ui: { ...state.ui, outline: { open: action.open } } };

    case "ui/treeToggle": {
      const has = state.ui.collapsedCycles.includes(action.cycleId);
      return {
        ...state,
        ui: {
          ...state.ui,
          collapsedCycles: has
            ? state.ui.collapsedCycles.filter((id) => id !== action.cycleId)
            : [...state.ui.collapsedCycles, action.cycleId],
        },
      };
    }

    case "ui/treeCursor":
      return { ...state, ui: { ...state.ui, treeCursor: action.source } };

    case "ui/follow":
      return { ...state, ui: { ...state.ui, follow: action.on } };

    case "ui/filter":
      return { ...state, ui: { ...state.ui, filter: action.filter } };

    case "ui/search": {
      const search = { ...state.ui.search, ...action.patch };
      if (action.patch && "q" in action.patch) {
        search.focus = 0;
        search.at = action.at;
      }
      return { ...state, ui: { ...state.ui, search } };
    }

    case "ui/focusBlock":
      return {
        ...state,
        ui: { ...state.ui, focusBlock: { index: action.index, at: action.at }, follow: false },
      };

    case "ui/palette":
      return { ...state, ui: { ...state.ui, palette: { ...state.ui.palette, ...action.patch } } };

    case "ui/shortcuts":
      return { ...state, ui: { ...state.ui, shortcuts: { open: action.open } } };

    case "ui/drawer":
      return { ...state, ui: { ...state.ui, drawer: action.open } };

    case "artifact/loading": {
      const key = `${action.source}|${action.name}`;
      const prev = state.artifacts.byKey[key];
      if (prev && prev.status === "ready") return state;
      return {
        ...state,
        artifacts: {
          byKey: { ...state.artifacts.byKey, [key]: { status: "loading", text: null, at: action.at, error: null } },
        },
      };
    }

    case "artifact/loaded": {
      const key = `${action.source}|${action.name}`;
      const prev = state.artifacts.byKey[key];
      if (prev && prev.at > action.at) return state;
      return {
        ...state,
        artifacts: {
          byKey: {
            ...state.artifacts.byKey,
            [key]: { status: "ready", text: action.text, at: action.at, error: null },
          },
        },
      };
    }

    case "artifact/error": {
      const key = `${action.source}|${action.name}`;
      const prev = state.artifacts.byKey[key];
      if (prev && prev.at > action.at) return state;
      return {
        ...state,
        artifacts: {
          byKey: {
            ...state.artifacts.byKey,
            [key]: { status: "error", text: null, at: action.at, error: action.error },
          },
        },
      };
    }

    case "context/loading": {
      const prev = state.context.bySource[action.source];
      if (prev && prev.status === "ready") return state;
      return {
        ...state,
        context: {
          bySource: {
            ...state.context.bySource,
            [action.source]: { status: "loading", payload: null, at: action.at, error: null },
          },
        },
      };
    }

    case "context/loaded": {
      const prev = state.context.bySource[action.source];
      if (prev && prev.at > action.at) return state;
      return {
        ...state,
        context: {
          bySource: {
            ...state.context.bySource,
            [action.source]: { status: "ready", payload: action.payload, at: action.at, error: null },
          },
        },
      };
    }

    case "context/error": {
      const prev = state.context.bySource[action.source];
      if (prev && prev.at > action.at) return state;
      return {
        ...state,
        context: {
          bySource: {
            ...state.context.bySource,
            [action.source]: { status: "error", payload: null, at: action.at, error: action.error },
          },
        },
      };
    }

    default:
      return state;
  }
}
