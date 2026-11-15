/* Command palette: fuzzy jump to nodes, policy steps, views and commands. */

import { el, clear, icon, copyText, toast } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";
import { viewsFor } from "../views/meta.js";
import { formatHash } from "../router.js";

const MAX_RESULTS = 40;
const IDLE_RESULTS = 12;
const IDLE_TABS = 4;
const BOUNDARY = /[\s:/_.-]/;
const FILTERS = ["all", "messages", "reasoning", "tools", "errors"];
const GROUP_LABEL = { command: "Commands", view: "Views", node: "Nodes", policy: "Policy steps" };
const GROUP_TEXT = { command: "command", view: "view", node: "nodes", policy: "policy" };
const FOOT = [
  ["↑↓", "move"],
  ["↵", "open"],
  ["esc", "close"],
];

export function mountPalette(root, ctx) {
  const { store, dispatch, act } = ctx;
  const input = el("input", {
    class: "palette-input",
    type: "text",
    placeholder: "Search nodes, policy steps, views, commands",
    autocomplete: "off",
    spellcheck: "false",
    role: "combobox",
    "aria-label": "Search",
    "aria-controls": "palette-list",
    "aria-expanded": "false",
    "aria-autocomplete": "list",
  });
  const list = el("div", { class: "palette-list", role: "listbox", id: "palette-list", "aria-label": "Results" });
  const overlay = el(
    "div",
    { class: "overlay palette-overlay", role: "dialog", "aria-modal": "true", "aria-label": "Command palette" },
    [
      el("div", { class: "dialog palette-dialog" }, [
        el("div", { class: "palette-search" }, [icon("search", 14), input, el("kbd", { class: "kbd" }, "esc")]),
        list,
        el(
          "div",
          { class: "palette-foot" },
          FOOT.map(([key, label]) =>
            el("span", { class: "palette-foot-item" }, [el("kbd", { class: "kbd" }, key), label]),
          ),
        ),
      ]),
    ],
  );
  overlay.hidden = true;
  root.append(overlay);

  let open = false;
  let activeIndex = 0;
  let results = [];
  let rows = [];
  let lastFocus = null;
  let lastQ = "";
  let pendingScroll = false;
  let renderSig = "";

  input.addEventListener("input", () => dispatch(act.palette({ open: true, q: input.value })));
  input.addEventListener("focusout", () => {
    if (!open) return;
    requestAnimationFrame(() => {
      if (open && document.hasFocus() && !overlay.contains(document.activeElement)) {
        input.focus({ preventScroll: true });
      }
    });
  });
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();
  });

  function close() {
    dispatch(act.palette({ open: false }));
  }

  function show(state) {
    lastFocus = document.activeElement;
    overlay.hidden = false;
    input.value = state.ui.palette.q || "";
    lastQ = input.value;
    activeIndex = 0;
    pendingScroll = true;
    renderSig = "";
    document.addEventListener("keydown", onDocKey, true);
    input.focus({ preventScroll: true });
    input.select();
  }

  function hide() {
    overlay.hidden = true;
    document.removeEventListener("keydown", onDocKey, true);
    clear(list);
    results = [];
    rows = [];
    renderSig = "";
    const target = lastFocus;
    lastFocus = null;
    if (target && target.isConnected && typeof target.focus === "function") {
      try {
        target.focus({ preventScroll: true });
      } catch {
        target.focus();
      }
    }
  }

  function update(state) {
    const isOpen = Boolean(state.ui.palette.open);
    if (isOpen !== open) {
      open = isOpen;
      if (open) show(state);
      else hide();
    }
    if (!open) return;
    const q = state.ui.palette.q || "";
    if (q !== lastQ) {
      lastQ = q;
      if (input.value !== q) input.value = q;
      activeIndex = 0;
      pendingScroll = true;
    }
    const sig = signature(state, q);
    if (sig === renderSig) return;
    renderSig = sig;
    render(state, q);
  }

  function render(state, q) {
    const items = collect(state);
    const query = q.trim();
    let entries;
    if (query) {
      entries = [];
      for (const item of items) {
        const score = matchScore(query, item);
        if (score != null) entries.push({ item, score });
      }
      entries.sort((a, b) => a.score - b.score || a.item.label.localeCompare(b.item.label));
      entries = entries.slice(0, MAX_RESULTS);
    } else {
      entries = idle(state, items).map((item) => ({ item, score: 0 }));
    }
    const groups = [];
    const byId = new Map();
    for (const entry of entries) {
      let group = byId.get(entry.item.group);
      if (!group) {
        group = { id: entry.item.group, entries: [] };
        byId.set(entry.item.group, group);
        groups.push(group);
      }
      group.entries.push(entry);
    }
    results = groups.flatMap((group) => group.entries);
    if (activeIndex >= results.length) activeIndex = Math.max(0, results.length - 1);
    clear(list);
    rows = [];
    input.setAttribute("aria-expanded", results.length ? "true" : "false");
    if (!results.length) {
      list.append(el("div", { class: "palette-empty", role: "presentation" }, query ? "No matches" : "Nothing to jump to"));
      input.removeAttribute("aria-activedescendant");
      return;
    }
    let index = 0;
    for (const group of groups) {
      list.append(el("div", { class: "palette-group", role: "presentation" }, GROUP_LABEL[group.id] || group.id));
      for (const entry of group.entries) {
        const row = rowFor(entry.item, index);
        rows.push(row);
        list.append(row);
        index += 1;
      }
    }
    paint();
    if (pendingScroll) {
      pendingScroll = false;
      scrollActive();
    }
  }

  function rowFor(item, index) {
    const row = el(
      "div",
      { class: "palette-row", role: "option", id: `palette-opt-${index}`, "aria-selected": "false" },
      [
        el("span", { class: "palette-label" }, item.label),
        el("span", { class: "palette-meta" }, [
          item.hint ? el("span", { class: "palette-hint" }, item.hint) : null,
          item.key ? el("kbd", { class: "kbd palette-key" }, item.key) : null,
        ]),
      ],
    );
    row.addEventListener("mousemove", () => {
      if (activeIndex === index) return;
      activeIndex = index;
      paint();
    });
    row.addEventListener("click", () => run(index));
    return row;
  }

  function paint() {
    for (let i = 0; i < rows.length; i += 1) {
      rows[i].setAttribute("aria-selected", i === activeIndex ? "true" : "false");
    }
    const row = rows[activeIndex];
    if (row) input.setAttribute("aria-activedescendant", row.id);
    else input.removeAttribute("aria-activedescendant");
  }

  function move(delta) {
    if (!results.length) return;
    activeIndex = (activeIndex + delta + results.length) % results.length;
    paint();
    scrollActive();
  }

  function scrollActive() {
    const row = rows[activeIndex];
    if (row && typeof row.scrollIntoView === "function") row.scrollIntoView({ block: "nearest" });
  }

  function run(index) {
    const entry = results[index];
    if (!entry) return;
    close();
    entry.item.run();
  }

  function onDocKey(event) {
    if (!open || event.defaultPrevented || event.isComposing) return;
    let handled = true;
    if (event.key === "ArrowDown") move(1);
    else if (event.key === "ArrowUp") move(-1);
    else if (event.key === "PageDown") move(8);
    else if (event.key === "PageUp") move(-8);
    else if (event.key === "Enter") run(activeIndex);
    else if (event.key === "Escape") close();
    else handled = false;
    if (handled) {
      event.preventDefault();
      event.stopPropagation();
    }
  }

  function collect(state) {
    const items = commands();
    const tab = sel.activeTab(state);
    if (tab) {
      for (const view of viewsFor(tab.kind)) {
        items.push({
          group: "view",
          label: view.label,
          hint: sel.tabTitle(state, tab),
          key: view.key,
          run: () => dispatch(act.view(view.id)),
        });
      }
    }
    for (const cycle of sel.cycles(state)) {
      const iteration = cycle.iteration ?? cycle.cycle_id;
      for (const node of cycle.nodes || []) {
        const source = sel.sourceOf("node", cycle.cycle_id, node.cell_id);
        const outcome =
          node.score != null
            ? fmt.pct(node.score)
            : node.fail_class
              ? fmt.failLabel(node.fail_class, true)
              : fmt.nodeStatusLabel(node.status);
        items.push({
          group: "node",
          label: node.cell_id === "root" ? "baseline" : node.cell_id,
          hint: `cycle ${iteration} · ${outcome}`,
          source,
          run: () => ctx.go(source),
        });
      }
      for (const step of cycle.policy_dev?.steps || []) {
        const source = sel.sourceOf("policy", cycle.cycle_id, String(step.step));
        items.push({
          group: "policy",
          label: `step ${step.step}`,
          hint: `cycle ${iteration} · ${fmt.verdictLabel(step.status)}${step.mean_v == null ? "" : ` · ${fmt.num(step.mean_v)}`}`,
          source,
          run: () => ctx.go(source),
        });
      }
    }
    return items;
  }

  function commands() {
    const state = store.getState();
    const dark = document.documentElement.dataset.theme === "dark";
    const toggle = (label, hint, run) => ({ label, hint, run });
    const out = [
      toggle("Toggle theme", dark ? "to light" : "to dark", () =>
        dispatch(act.theme(document.documentElement.dataset.theme === "dark" ? "light" : "dark")),
      ),
      toggle("Toggle sidebar", state.ui.sidebar.open ? "hide" : "show", () =>
        dispatch(act.sidebar({ open: !store.getState().ui.sidebar.open })),
      ),
      toggle("Toggle inspector", state.ui.inspector.open ? "hide" : "show", () =>
        dispatch(act.inspector({ open: !store.getState().ui.inspector.open })),
      ),
      toggle("Toggle outline", state.ui.outline.open ? "hide" : "show", () =>
        dispatch(act.outline(!store.getState().ui.outline.open)),
      ),
      toggle("Follow transcript", state.ui.follow ? "on" : "off", () =>
        dispatch(act.follow(!store.getState().ui.follow)),
      ),
    ];
    for (const filter of FILTERS) {
      out.push(
        toggle(`Filter: ${filter}`, state.ui.filter === filter ? "current" : "", () =>
          dispatch(act.filter(filter)),
        ),
      );
    }
    out.push(
      { label: "Copy link to this view", hint: "clipboard", run: copyLink },
      { label: "Open live node", hint: "running", run: openLive },
      { label: "Next node", hint: "tree", key: "]", run: () => step(1) },
      { label: "Previous node", hint: "tree", key: "[", run: () => step(-1) },
    );
    return out.map((item) => ({ group: "command", ...item }));
  }

  function idle(state, items) {
    const tabs = new Set(state.ui.tabs.map((tab) => tab.id));
    const out = [];
    for (const item of items) {
      if (item.source && tabs.has(item.source) && out.length < IDLE_TABS) out.push(item);
    }
    for (const item of items) {
      if (item.group === "command" && out.length < IDLE_RESULTS) out.push(item);
    }
    for (const item of items) {
      if (item.group === "view" && out.length < IDLE_RESULTS) out.push(item);
    }
    return out;
  }

  function step(direction) {
    const source = sel.nextSource(store.getState(), direction);
    if (source) ctx.go(source);
  }

  function openLive() {
    const state = store.getState();
    for (const cycle of sel.cycles(state)) {
      const node = (cycle.nodes || []).find((entry) => sel.isNodeLive(state, entry));
      if (node) {
        ctx.go(sel.sourceOf("node", cycle.cycle_id, node.cell_id));
        return;
      }
    }
    toast("no live node");
  }

  function copyLink() {
    const tab = sel.activeTab(store.getState());
    const hash = tab ? formatHash(tab) : location.hash;
    copyText(`${location.origin}${location.pathname}${hash}`).then((ok) =>
      toast(ok ? "link copied" : "copy failed"),
    );
  }

  function signature(state, q) {
    const tab = sel.activeTab(state);
    const cycles = state.server.data?.cycles || [];
    const nodes = cycles.map(
      (cycle) =>
        `${cycle.cycle_id}:${cycle.iteration}:${cycle.status}:` +
        (cycle.nodes || []).map((node) => `${node.cell_id}:${node.status}:${node.score}:${node.fail_class}`).join(",") +
        ":" +
        (cycle.policy_dev?.steps || []).map((step) => `${step.step}:${step.status}:${step.mean_v}`).join(","),
    );
    return [
      q,
      state.ui.filter,
      state.ui.follow ? 1 : 0,
      state.ui.sidebar.open ? 1 : 0,
      state.ui.inspector.open ? 1 : 0,
      state.ui.outline.open ? 1 : 0,
      document.documentElement.dataset.theme,
      tab ? `${tab.id}|${tab.view}` : "",
      state.ui.tabs.map((item) => item.id).join(","),
      nodes.join("|"),
    ].join("~");
  }

  update(store.getState());
  return {
    update,
    destroy() {
      document.removeEventListener("keydown", onDocKey, true);
      overlay.remove();
    },
  };
}

function fuzzy(query, text) {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  let index = 0;
  let previous = -1;
  let score = 0;
  for (const ch of q) {
    if (ch === " ") continue;
    const at = t.indexOf(ch, index);
    if (at < 0) return null;
    score += at - previous;
    if (at === 0 || BOUNDARY.test(t[at - 1])) score -= 2;
    previous = at;
    index = at + 1;
  }
  return score;
}

function matchScore(query, item) {
  const label = fuzzy(query, item.label);
  const hint = fuzzy(query, `${item.hint || ""} ${GROUP_TEXT[item.group] || ""}`);
  if (label == null) return hint == null ? null : hint + 8;
  if (hint == null) return label;
  return Math.min(label, hint + 8);
}
