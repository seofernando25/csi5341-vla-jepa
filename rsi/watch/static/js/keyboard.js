/* Global keyboard map. One listener; every shortcut is listed in the sheet. */

import * as sel from "./selectors.js";
import { viewsFor } from "./views/meta.js";

export function installKeyboard(store, { act, go }) {
  const onKey = (event) => {
    if (event.isComposing) return;
    const state = store.getState();
    const paletteOpen = state.ui.palette.open;
    const shortcutsOpen = state.ui.shortcuts.open;
    const modalOpen = paletteOpen || shortcutsOpen;
    const mod = event.metaKey || event.ctrlKey;
    const key = event.key;

    if (key === "Escape") {
      if (paletteOpen) {
        event.preventDefault();
        store.dispatch(act.palette({ open: false }));
        return;
      }
      if (shortcutsOpen) {
        event.preventDefault();
        store.dispatch(act.shortcuts(false));
        return;
      }
      if (state.ui.drawer) {
        event.preventDefault();
        store.dispatch(act.drawer(null));
        return;
      }
      const search = document.getElementById("outline-search");
      if (search && document.activeElement === search) {
        event.preventDefault();
        search.blur();
        store.dispatch(act.search({ open: false }));
        return;
      }
      if (state.ui.search.open) {
        event.preventDefault();
        store.dispatch(act.search({ open: false }));
      }
      return;
    }

    if (event.defaultPrevented) return;

    if (mod && !event.shiftKey && !event.altKey && key.toLowerCase() === "k") {
      event.preventDefault();
      if (event.repeat) return;
      if (paletteOpen) {
        store.dispatch(act.palette({ open: false }));
      } else {
        if (shortcutsOpen) store.dispatch(act.shortcuts(false));
        store.dispatch(act.palette({ open: true, q: "" }));
      }
      return;
    }

    if (modalOpen) return;
    if (isTyping(event.target)) return;
    if (mod || event.altKey) return;

    if (key === "?") {
      event.preventDefault();
      store.dispatch(act.shortcuts(true));
      return;
    }

    if (/^[1-6]$/.test(key)) {
      const tab = sel.activeTab(state);
      if (!tab) return;
      const view = viewsFor(tab.kind)[Number(key) - 1];
      if (view) {
        event.preventDefault();
        store.dispatch(act.view(view.id));
      }
      return;
    }

    if (key === "[" || key === "]") {
      const source = sel.nextSource(state, key === "]" ? 1 : -1);
      if (source) {
        event.preventDefault();
        go(source);
      }
      return;
    }

    if (key === "/") {
      event.preventDefault();
      store.dispatch(act.search({ open: true }));
      focusSearch();
    }
  };

  function focusSearch() {
    const input = document.getElementById("outline-search");
    if (input && input.offsetParent != null) {
      input.focus();
      input.select();
      return;
    }
    store.dispatch(act.outline(true));
    requestAnimationFrame(() => {
      const next = document.getElementById("outline-search");
      if (next) {
        next.focus();
        next.select();
      }
    });
  }

  window.addEventListener("keydown", onKey);
  return () => window.removeEventListener("keydown", onKey);
}

function isTyping(target) {
  if (!target) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable === true;
}
