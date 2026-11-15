/* Proposal: the node's written proposal, rendered as prose. */

import { el, clear, copyButton } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";
import { renderMarkdown } from "../markdown.js";

export function mount(root, ctx) {
  root.classList.add("proposal-root");
  const head = el("header", { class: "proposal-head" });
  const doc = el("article", { class: "md proposal-doc" });
  const scroller = el("div", { class: "proposal-scroll", tabIndex: 0 }, doc);
  root.append(el("div", { class: "proposal-view" }, [head, scroller]));

  let raw = "";
  let sig = "";
  const copy = copyButton(() => raw, "Copy raw");
  copy.title = "Copy the proposal markdown";

  function chips(state, node) {
    const out = [];
    if (!node) return out;
    if (node.score != null) {
      const tone = node.fail_class === "ok" ? "ok" : node.fail_class ? "err" : null;
      out.push(el("span", { class: "pill num", dataset: tone ? { tone } : {} }, `score ${fmt.pct(node.score)}`));
    }
    if (node.successes != null && node.episodes != null) {
      out.push(el("span", { class: "pill num" }, `smoke ${node.successes}/${node.episodes}`));
    }
    const phase = sel.nodePhase(state, node);
    if (phase !== "done") {
      const tone = phase === "agent" || phase === "evaluating" ? "live" : "warn";
      out.push(el("span", { class: "pill", dataset: { tone } }, fmt.nodeStatusLabel(phase)));
    }
    return out;
  }

  function renderHead(state, node, source) {
    const cell = node?.cell_id || (source ? source.split(":").pop() : "");
    const headline = node?.headline || "No headline recorded";
    clear(head);
    head.append(
      el("div", { class: "proposal-title" }, [
        el("div", { class: "proposal-eyebrow" }, [
          el("span", {}, "Proposal"),
          cell ? el("span", { class: "proposal-cell mono" }, cell) : null,
        ]),
        el("h1", { class: "proposal-headline", title: headline }, headline),
      ]),
      el("div", { class: "proposal-chips" }, chips(state, node)),
      el("div", { class: "proposal-actions" }, copy),
    );
  }

  function renderBody(entry) {
    clear(doc);
    if (!entry || entry.status === "loading") {
      doc.append(el("p", { class: "empty" }, "Loading proposal…"));
      return;
    }
    if (entry.status !== "ready" || !raw.trim()) {
      doc.append(
        el("div", { class: "empty" }, [
          el("strong", {}, "No proposal was recorded for this node."),
          entry.error || "",
        ]),
      );
      return;
    }
    doc.append(renderMarkdown(raw));
  }

  function render(state) {
    const source = sel.activeSource(state);
    const node = source ? sel.nodeForSource(state, source) : null;
    const entry = source ? sel.artifact(state, source, "proposal") : null;
    const next = [
      source,
      entry?.status,
      entry?.at,
      entry?.error,
      entry?.text?.length,
      node?.cell_id,
      node?.headline,
      node?.score,
      node?.successes,
      node?.episodes,
      node?.status,
      node?.fail_class,
    ].join("|");
    if (next === sig) return;
    sig = next;
    raw = entry?.status === "ready" ? entry.text || "" : "";
    copy.disabled = !raw.trim();
    renderHead(state, node, source);
    renderBody(entry);
  }

  render(ctx.store.getState());
  return { update: render, destroy() {} };
}
