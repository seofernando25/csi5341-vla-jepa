/* Solution: the scored file, line-numbered and highlighted. */

import { el, clear, copyButton, icon } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";
import { highlightLines } from "../highlight.js";

export function mount(root, ctx) {
  const view = el("div", { class: "code-view solution-view", dataset: { wrap: "false" } });
  const head = el("div", { class: "code-head" });
  const scroll = el("div", { class: "code-scroll" });
  view.append(head, scroll);
  root.append(view);

  let signature = "";
  let wrap = false;

  function paintHead(node, text, lines) {
    clear(head);
    const wrapButton = el(
      "button",
      { class: "btn", type: "button", title: "Toggle soft wrap", "aria-pressed": wrap ? "true" : "false" },
      [icon("list", 12), "Wrap"],
    );
    wrapButton.addEventListener("click", () => {
      wrap = !wrap;
      view.dataset.wrap = wrap ? "true" : "false";
      wrapButton.setAttribute("aria-pressed", wrap ? "true" : "false");
    });
    const commit = node?.commit ? fmt.shortId(node.commit, 8) : null;
    head.append(
      el("span", { class: "code-path mono" }, "solution.py"),
      el("span", { class: "code-meta" }, ["commit ", el("span", { class: "mono" }, commit || "—")]),
      el("span", { class: "code-meta num" }, `${fmt.plural(lines, "line")} · ${fmt.chars(text.length)} chars`),
      el("span", { class: "spacer" }),
      wrapButton,
      copyButton(() => text, "Copy"),
    );
  }

  function paintBody(rows) {
    clear(scroll);
    const lines = el("div", { class: "code-lines" });
    rows.forEach((fragment, index) => {
      lines.append(
        el("div", { class: "code-line" }, [
          el("span", { class: "ln num", "aria-hidden": "true" }, String(index + 1)),
          el("span", { class: "code-text" }, fragment),
        ]),
      );
    });
    scroll.append(lines);
  }

  function empty(text) {
    clear(head);
    clear(scroll);
    scroll.append(el("p", { class: "empty" }, [el("strong", {}, text)]));
  }

  function render(state) {
    const source = sel.activeSource(state);
    const entry = source ? sel.artifact(state, source, "solution") : null;
    const node = source ? sel.nodeForSource(state, source) : null;
    const next = `${source}|${entry?.status}|${entry?.at}|${node?.commit}`;
    if (next === signature) return;
    signature = next;

    if (!entry || entry.status === "loading") {
      clear(head);
      clear(scroll);
      return scroll.append(el("p", { class: "empty" }, [el("span", { class: "spinner" }), "Loading solution…"]));
    }
    if (entry.status === "error") {
      if (notFound(entry.error)) return empty("No solution recorded.");
      clear(head);
      clear(scroll);
      return scroll.append(
        el("p", { class: "empty" }, [el("strong", {}, "Solution unavailable"), entry.error]),
      );
    }
    const text = entry.text || "";
    if (!text.trim()) return empty("The scored file is empty.");
    const rows = highlightLines(text, "python");
    paintHead(node, text, rows.length);
    paintBody(rows);
  }

  render(ctx.store.getState());
  return { update: render, destroy() {} };
}

function notFound(message) {
  return /not found|404/i.test(String(message || ""));
}
