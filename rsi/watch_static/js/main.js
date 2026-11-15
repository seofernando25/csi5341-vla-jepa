/* Bootstrap: store, data loading, routing, layout and view mounting. */

import { createStore } from "./store.js";
import { makeInitialState, reducer } from "./reducer.js";
import { act } from "./actions.js";
import * as sel from "./selectors.js";
import * as fmt from "./format.js";
import * as dom from "./dom.js";
import { createApi } from "./api.js";
import { openStream } from "./sse.js";
import { parseHash, writeHash, parseStory, storyHash } from "./router.js";
import { mountStory } from "./story/index.js";
import { mountView } from "./views/index.js";
import { mount as mountTitlebar } from "./components/titlebar.js";
import { mountTabs, mountViewbar } from "./components/tabs.js";
import { mount as mountStatusbar } from "./components/statusbar.js";
import { mount as mountInspector } from "./components/inspector.js";
import { mount as mountTree } from "./views/tree.js";
import { mount as mountPolicy } from "./views/policy.js";
import { mount as mountOutline } from "./views/outline.js";
import { mountPalette } from "./components/palette.js";
import { mountShortcuts } from "./components/shortcuts.js";
import { installKeyboard } from "./keyboard.js";

const DEMO = new URLSearchParams(location.search).get("demo") === "1";
const POLL_MS = 2000;
const TICK_MS = 1000;

const stored = safeTheme();
const store = createStore(reducer, makeInitialState({ theme: stored }));
let api = null;
let story = null;
let mode = "workbench";

const ctx = {
  store,
  dispatch: store.dispatch,
  act,
  sel,
  fmt,
  dom,
  api: null,
  go: (source, view) => {
    store.dispatch(act.open(source, view));
    const tab = sel.activeTab(store.getState());
    if (!tab) return;
    writeHash(tab);
    applyMode("workbench");
  },
};

const regions = {};
const mounted = { view: null, key: null, outline: false };
let stream = null;
let streamSource = null;
const inflight = new Set();
const failed = new Set();
let lastStateAt = 0;
let lastStateError = null;

/* --- mount the shell ---------------------------------------------------- */
function mountRegions() {
  regions.titlebar = mountTitlebar(document.getElementById("titlebar"), ctx);
  regions.tabs = mountTabs(document.getElementById("tabbar"), ctx);
  regions.viewbar = mountViewbar(document.getElementById("viewbar"), ctx);
  regions.statusbar = mountStatusbar(document.getElementById("statusbar"), ctx);
  regions.tree = mountTree(document.getElementById("tree-pane"), ctx);
  regions.policy = mountPolicy(document.getElementById("policy-pane"), ctx);
  regions.inspector = mountInspector(document.getElementById("inspector"), ctx);
  regions.outline = mountOutline(document.getElementById("outline"), ctx);
  regions.palette = mountPalette(document.getElementById("overlay-root"), ctx);
  regions.shortcuts = mountShortcuts(document.getElementById("overlay-root"), ctx);
}

/* --- data loading ------------------------------------------------------- */
const VIEW_ARTIFACTS = {
  node: {
    proposal: ["proposal"],
    diff: ["diff"],
    solution: ["solution"],
    evaluation: ["score", "error"],
    transcript: [],
    context: [],
  },
  policy: {
    prompt: ["seed"],
    diff: ["policy", "seed"],
    replay: ["curve"],
    verdict: ["verdict"],
  },
};

function ensureArtifact(source, name) {
  const key = sel.artifactKey(source, name);
  const entry = store.getState().artifacts.byKey[key];
  if (entry && (entry.status === "ready" || entry.status === "loading")) return;
  if (inflight.has(key) || failed.has(key)) return;
  inflight.add(key);
  store.dispatch(act.artifactLoading(source, name));
  api
    .artifact(source, name)
    .then((payload) => store.dispatch(act.artifactLoaded(source, name, payload.text)))
    .catch((error) => {
      if (error.status === 404) failed.add(key);
      store.dispatch(act.artifactError(source, name, error.message));
    })
    .finally(() => inflight.delete(key));
}

function ensureContext(source) {
  const entry = sel.contextFor(store.getState(), source);
  if (entry && (entry.status === "ready" || entry.status === "loading")) return;
  const key = `ctx|${source}`;
  if (inflight.has(key) || failed.has(key)) return;
  inflight.add(key);
  store.dispatch(act.contextLoading(source));
  api
    .context(source)
    .then((payload) => store.dispatch(act.contextLoaded(source, payload)))
    .catch((error) => {
      if (error.status === 404) failed.add(key);
      store.dispatch(act.contextError(source, error.message));
    })
    .finally(() => inflight.delete(key));
}

function ensureTranscript(source) {
  const state = store.getState();
  if (state.transcript.source === source && state.transcript.status === "error") return;
  if (state.transcript.source === source) {
    if (state.transcript.status === "ready" && !stream && !state.transport.frozen) {
      openFor(source, state.transcript.next);
    }
    return;
  }
  if (inflight.has(`ev|${source}`) || failed.has(`ev|${source}`)) return;
  inflight.add(`ev|${source}`);
  store.dispatch(act.transcriptLoading(source));
  api
    .events(source, 0)
    .then((payload) => {
      store.dispatch(act.transcriptLoaded(source, payload.events || [], payload.next || 0));
      if (payload.done) {
        const reason = source.startsWith("node:") ? "score" : "verdict";
        store.dispatch(act.done(source, reason, payload.next || 0));
      } else {
        openFor(source, payload.next || 0);
      }
    })
    .catch((error) => {
      if (error.status === 404) failed.add(`ev|${source}`);
      store.dispatch(act.transcriptError(source, error.message));
    })
    .finally(() => inflight.delete(`ev|${source}`));
}

function openFor(source, after) {
  if (stream && streamSource === source) return;
  closeStream();
  streamSource = source;
  store.dispatch(act.attach(source, after));
  const handlers = {
    onEvent: (frame) => onStream(source, frame),
    onStatus: (status) => {
      if (status === "reconnecting") store.dispatch(act.transportError(source, "reconnecting"));
    },
  };
  stream = api.demo ? api.stream({ source, after, ...handlers }) : openStream({ source, after, ...handlers });
}

function onStream(source, frame) {
  if (frame.type === "hello") {
    store.dispatch(act.hello(frame.payload));
  } else if (frame.type === "events") {
    store.dispatch(act.events(source, frame.events, frame.next));
  } else if (frame.type === "reset") {
    store.dispatch(act.reset(source, frame.reason));
  } else if (frame.type === "done") {
    store.dispatch(act.done(source, frame.reason, frame.next));
    closeStream();
  } else if (frame.type === "state") {
    const seq = frame.payload?.seq;
    if (seq != null && seq > store.getState().server.seq) {
      store.dispatch(act.stateLoaded(frame.payload.state, Date.now(), seq));
    }
  } else if (frame.type === "error") {
    store.dispatch(act.transportError(source, frame.error));
  }
}

function closeStream() {
  if (stream) stream.close();
  stream = null;
  streamSource = null;
}

/* What the active tab needs right now. */
function effects(state) {
  const tab = sel.activeTab(state);
  if (!tab) return;
  const kind = tab.kind;
  const view = tab.view;
  if (kind === "node" && view === "transcript") {
    ensureTranscript(tab.id);
    return;
  }
  if (streamSource) {
    store.dispatch(act.detach(streamSource));
    closeStream();
  }
  if (kind === "node") {
    for (const name of VIEW_ARTIFACTS.node[view] || []) ensureArtifact(tab.id, name);
    if (view === "context") ensureContext(tab.id);
  } else {
    for (const name of VIEW_ARTIFACTS.policy[view] || []) ensureArtifact(tab.id, name);
    if (view === "prompt") ensureContext(tab.id);
  }
}

/* --- routing ------------------------------------------------------------ */
function applyMode(next) {
  if (mode === next) return;
  mode = next;
  document.body.dataset.mode = mode;
  document.getElementById("app").hidden = mode === "story";
  document.getElementById("story-root").hidden = mode !== "story";
  const state = store.getState();
  if (mode === "story" && story) story.update(state, null);
  else render(state, null);
}

function routeFromHash(state) {
  const storyRoute = parseStory(location.hash);
  if (storyRoute) {
    applyMode("story");
    if (story) story.go(storyRoute.chapter);
    return true;
  }
  applyMode("workbench");
  const route = parseHash(location.hash);
  if (!route) return false;
  const source = sel.sourceOf(route.kind, route.cycle, route.target);
  store.dispatch(act.open(source, route.view || undefined));
  return true;
}

/* --- theme -------------------------------------------------------------- */
function safeTheme() {
  try {
    const value = localStorage.getItem("watch.theme");
    return value === "dark" || value === "light" || value === "system" ? value : "system";
  } catch {
    return "system";
  }
}

function applyTheme(theme) {
  const dark = theme === "dark" || (theme === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  document.documentElement.dataset.themeChoice = theme;
  try {
    localStorage.setItem("watch.theme", theme);
  } catch {
    /* private mode: the choice lasts for this page only */
  }
}

/* --- layout ------------------------------------------------------------- */
function applyLayout(state) {
  const body = document.getElementById("body");
  const narrow = matchMedia("(max-width: 900px)").matches;
  const drawer = state.ui.drawer;
  body.dataset.sidebar = state.ui.sidebar.open || drawer === "tree" ? "open" : "closed";
  body.dataset.inspector = state.ui.inspector.open || drawer === "inspector" ? "open" : "closed";
  body.dataset.drawer = narrow && drawer ? drawer : "";
  document.getElementById("drawer-scrim").hidden = !(narrow && drawer);
  document.documentElement.style.setProperty("--sidebar-w", `${state.ui.sidebar.width}px`);
  document.documentElement.style.setProperty("--inspector-w", `${state.ui.inspector.width}px`);
  document.getElementById("splitter")?.setAttribute("aria-valuenow", String(state.ui.sidebar.width));
  document.getElementById("splitter-inspector")?.setAttribute("aria-valuenow", String(state.ui.inspector.width));
  const editorBody = document.getElementById("editor-body");
  const tab = sel.activeTab(state);
  const showOutline = state.ui.outline.open && tab?.kind === "node" && tab.view === "transcript";
  editorBody.dataset.outline = state.ui.search.open ? "search" : showOutline ? "open" : "closed";
}

function installSplitters() {
  for (const [id, key] of [
    ["splitter", "sidebar"],
    ["splitter-inspector", "inspector"],
  ]) {
    const handle = document.getElementById(id);
    if (!handle) continue;
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      handle.dataset.dragging = "true";
      const startX = event.clientX;
      const start = store.getState().ui[key].width;
      const move = (moveEvent) => {
        const delta = key === "sidebar" ? moveEvent.clientX - startX : startX - moveEvent.clientX;
        const width = Math.max(180, Math.min(480, start + delta));
        store.dispatch(key === "sidebar" ? act.sidebar({ width }) : act.inspector({ width }));
      };
      const up = () => {
        handle.dataset.dragging = "false";
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
    handle.addEventListener("keydown", (event) => {
      const delta = event.key === "ArrowLeft" ? -16 : event.key === "ArrowRight" ? 16 : 0;
      if (!delta) return;
      event.preventDefault();
      const width = store.getState().ui[key].width + delta;
      store.dispatch(key === "sidebar" ? act.sidebar({ width }) : act.inspector({ width }));
    });
  }
}

/* --- view mounting ------------------------------------------------------ */
function mountActiveView(state) {
  const tab = sel.activeTab(state);
  const host = document.getElementById("view-host");
  const key = tab ? `${tab.id}|${tab.view}` : "empty";
  if (mounted.key === key) return;
  if (mounted.view) mounted.view.destroy();
  host.replaceChildren();
  mounted.key = key;
  mounted.view = null;
  if (!tab) {
    const cycles = state.server.data?.cycles || [];
    const text = cycles.length ? "No editor open" : "No cycles recorded";
    host.append(dom.el("div", { class: "empty" }, [dom.el("strong", {}, text)]));
    return;
  }
  const root = dom.el("div", { class: "view-root", dataset: { view: tab.view } });
  host.append(root);
  mounted.view = mountView(tab.kind, tab.view, root, ctx);
}

/* --- render loop -------------------------------------------------------- */
function render(state, prev) {
  if (state.ui.theme !== prev?.ui.theme) applyTheme(state.ui.theme);
  if (mode === "story") {
    if (story) story.update(state, prev);
    return;
  }
  applyLayout(state);
  mountActiveView(state);
  for (const region of Object.values(regions)) region.update(state, prev);
  if (mounted.view) mounted.view.update(state, prev);
  const snapshot = state;
  queueMicrotask(() => effects(snapshot));
  if (state.server.status === "error" && state.server.error !== lastStateError) {
    lastStateError = state.server.error;
    dom.toast(`state unavailable: ${state.server.error}`, "err");
  }
  if (state.server.status === "ready") lastStateError = null;
  const tab = sel.activeTab(state);
  document.title = tab ? `${sel.tabTitle(state, tab)} — Dream-RSI` : "Dream-RSI workbench";
  if (tab) writeHash(tab, { replace: !prev?.ui.activeId });
}

/* --- boot --------------------------------------------------------------- */
async function boot() {
  api = await createApi({ demo: DEMO });
  ctx.api = api;
  mountRegions();
  story = mountStory(document.getElementById("story-root"), ctx);
  installSplitters();
  installKeyboard(store, { act, go: ctx.go });
  store.subscribe(render);

  document.getElementById("drawer-scrim").addEventListener("click", () => store.dispatch(act.drawer(null)));
  window.addEventListener("hashchange", () => routeFromHash(store.getState()));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (store.getState().ui.theme === "system") applyTheme("system");
  });
  matchMedia("(max-width: 900px)").addEventListener("change", () => applyLayout(store.getState()));

  applyTheme(store.getState().ui.theme);
  document.body.dataset.mode = mode;
  render(store.getState(), null);

  const poll = async () => {
    try {
      const data = await api.state();
      lastStateAt = Date.now();
      store.dispatch(act.stateLoaded(data, lastStateAt));
    } catch (error) {
      store.dispatch(act.stateError(error.message));
    }
  };

  await poll();
  setInterval(poll, POLL_MS);
  setInterval(() => store.dispatch({ type: "tick", at: Date.now() }), TICK_MS);
  routeFromHash(store.getState());
  if (mode === "workbench" && !sel.activeTab(store.getState())) {
    const source = sel.defaultSource(store.getState());
    if (source) store.dispatch(act.open(source));
  }
}

boot().catch((error) => {
  console.error(error);
  dom.toast(`page failed to start: ${error.message}`, "err");
});
