/* Shortcut sheet: the one place the whole key map is listed. */

import { el, icon, modLabel } from "../dom.js";

export function mountShortcuts(root, ctx) {
  const { dispatch, act } = ctx;
  const titleId = "shortcuts-title";
  const rows = [
    { keys: [modLabel(), "K"], label: "Command palette" },
    { keys: ["1–6"], label: "Switch view" },
    { keys: ["[", "]"], label: "Previous / next node" },
    { keys: ["/"], label: "Search transcript" },
    { keys: ["?"], label: "This sheet" },
    { keys: ["Esc"], label: "Close palette, sheet or drawer" },
    { keys: ["j", "k"], label: "Move in the tree" },
    { keys: ["Enter"], label: "Open selected node" },
    { keys: ["↑", "↓"], label: "Move in the tree" },
    { keys: ["Shift", "Enter"], label: "Previous search hit" },
  ];
  const closeBtn = el(
    "button",
    { class: "icon-btn", type: "button", title: "Close", "aria-label": "Close" },
    icon("close", 14),
  );
  const body = el(
    "tbody",
    {},
    rows.map((row) =>
      el("tr", {}, [
        el("th", { scope: "row" }, [
          el(
            "span",
            { class: "shortcuts-keys" },
            row.keys.map((key) => el("kbd", { class: "kbd" }, key)),
          ),
        ]),
        el("td", {}, row.label),
      ]),
    ),
  );
  const dialog = el(
    "div",
    {
      class: "dialog shortcuts-dialog",
      tabIndex: -1,
    },
    [
      el("div", { class: "dialog-head" }, [
        el("span", { class: "dialog-title", id: titleId }, "Keyboard shortcuts"),
        el("span", { class: "shortcuts-spacer" }),
        closeBtn,
      ]),
      el("div", { class: "dialog-body" }, [el("table", { class: "shortcuts-table" }, [body])]),
    ],
  );
  const overlay = el(
    "div",
    {
      class: "overlay shortcuts-overlay",
      role: "dialog",
      "aria-modal": "true",
      "aria-labelledby": titleId,
    },
    [dialog],
  );
  overlay.hidden = true;
  root.append(overlay);

  let open = false;
  let lastFocus = null;

  function close() {
    dispatch(act.shortcuts(false));
  }

  closeBtn.addEventListener("click", close);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();
  });

  function onKey(event) {
    if (!open || event.key !== "Escape" || event.defaultPrevented || event.isComposing) return;
    event.preventDefault();
    event.stopPropagation();
    close();
  }

  function show() {
    lastFocus = document.activeElement;
    overlay.hidden = false;
    document.addEventListener("keydown", onKey, true);
    dialog.focus({ preventScroll: true });
  }

  function hide() {
    overlay.hidden = true;
    document.removeEventListener("keydown", onKey, true);
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
    const isOpen = Boolean(state.ui.shortcuts.open);
    if (isOpen === open) return;
    open = isOpen;
    if (open) show();
    else hide();
  }

  update(ctx.store.getState());
  return {
    update,
    destroy() {
      document.removeEventListener("keydown", onKey, true);
      overlay.remove();
    },
  };
}
