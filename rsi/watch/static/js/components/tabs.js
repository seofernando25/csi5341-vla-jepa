/* Editor tabs and the view switcher for the active tab. */

import { el, clear, icon, setText } from "../dom.js";
import * as sel from "../selectors.js";
import { viewsFor } from "../views/meta.js";

export function mountTabs(root, ctx) {
  const { dispatch, act } = ctx;
  const list = el("div", { class: "tabs", role: "tablist", "aria-label": "Open editors" });
  const actions = el("div", { class: "tabbar-actions" });
  root.append(list, actions);

  const outlineBtn = el("button", { class: "icon-btn", type: "button", "aria-label": "Outline" }, icon("list", 14));
  outlineBtn.addEventListener("click", () => {
    dispatch(act.outline(!ctx.store.getState().ui.outline.open));
  });
  actions.append(outlineBtn);

  let sig = "";

  function render(state) {
    const tabs = state.ui.tabs;
    const active = state.ui.activeId;
    const next = JSON.stringify([tabs.map((t) => [t.id, t.view]), active]);
    if (next === sig) return;
    sig = next;
    clear(list);
    for (const tab of tabs) {
      const selected = tab.id === active;
      const close = el("button", {
        class: "tab-close",
        type: "button",
        tabIndex: -1,
        "aria-label": `Close ${sel.tabTitle(state, tab)}`,
      }, "×");
      close.addEventListener("click", (event) => {
        event.stopPropagation();
        dispatch(act.close(tab.id));
      });
      const node = el(
        "div",
        {
          class: "tab",
          role: "tab",
          tabIndex: selected ? 0 : -1,
          "aria-selected": selected ? "true" : "false",
          "aria-controls": "view-host",
          dataset: { id: tab.id },
        },
        [
          icon(tab.kind === "policy" ? "refresh" : tab.target === "root" ? "dot" : "file", 12),
          el("span", { class: "tab-label" }, sel.tabTitle(state, tab)),
          close,
        ],
      );
      node.addEventListener("click", () => dispatch(act.activate(tab.id)));
      node.addEventListener("auxclick", (event) => {
        if (event.button === 1) dispatch(act.close(tab.id));
      });
      node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          dispatch(act.activate(tab.id));
          return;
        }
        const step = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
        if (!step) return;
        event.preventDefault();
        const index = tabs.findIndex((item) => item.id === tab.id);
        const nextTab = tabs[(index + step + tabs.length) % tabs.length];
        dispatch(act.activate(nextTab.id));
        requestAnimationFrame(() => list.querySelector(`[data-id="${nextTab.id}"]`)?.focus());
      });
      list.append(node);
    }
    outlineBtn.setAttribute("aria-pressed", state.ui.outline.open ? "true" : "false");
  }

  render(ctx.store.getState());
  return { update: render, destroy() {} };
}

export function mountViewbar(root, ctx) {
  const { dispatch, act } = ctx;
  root.setAttribute("role", "tablist");
  root.setAttribute("aria-label", "Views");
  let sig = "";

  function render(state) {
    const tab = sel.activeTab(state);
    const next = tab ? `${tab.id}|${tab.view}` : "none";
    if (next === sig) return;
    sig = next;
    clear(root);
    if (!tab) return;
    const views = viewsFor(tab.kind);
    for (const view of views) {
      const selected = tab.view === view.id;
      const button = el(
        "button",
        {
          class: "viewtab",
          type: "button",
          role: "tab",
          "aria-selected": selected ? "true" : "false",
          "aria-controls": "view-host",
          dataset: { view: view.id },
        },
        [el("span", {}, view.label), el("span", { class: "key" }, view.key)],
      );
      button.addEventListener("click", () => dispatch(act.view(view.id)));
      root.append(button);
    }
    root.append(el("span", { class: "viewbar-spacer" }));
    const copy = el("button", { class: "copy-btn", type: "button", title: "Copy a link to this view" }, [
      icon("external", 12),
      "link",
    ]);
    copy.addEventListener("click", async () => {
      const url = `${location.origin}${location.pathname}${location.hash}`;
      try {
        await navigator.clipboard.writeText(url);
        setText(copy.lastChild, "copied");
        setTimeout(() => setText(copy.lastChild, "link"), 1200);
      } catch {
        setText(copy.lastChild, "failed");
      }
    });
    root.append(copy);
  }

  render(ctx.store.getState());
  return { update: render, destroy() {} };
}
